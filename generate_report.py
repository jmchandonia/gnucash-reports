#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_report.py
Interactive financial report (HTML) from a GnuCash file.

Features
--------
* Reads native GnuCash compressed XML (.gnucash).
* Monthly income (bank vs retirement) – line chart with hover details.
* Monthly expenses (education, housing, other) – line chart with hover details.
* Net cash flow (bank income – all expenses) – line chart (7‑month MA).
* Account values (bank, investment, retirement) by month – stacked bars.
"""

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import urllib.request

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1️⃣  CONFIGURATION – adjust only if your account names use different keywords
# ----------------------------------------------------------------------
ACCOUNT_TYPE_PATTERNS = {
    "bank":        re.compile(r"Bank Accounts:", re.I),
    "retirement":  re.compile(r"Retirement|IRA|401k", re.I),
    "investment":  re.compile(r"Investments?|Brokerage|Securities", re.I),
}
INCOME_ACCOUNT_PATTERN = re.compile(r"^Income:", re.I)
EXPENSE_ACCOUNT_PATTERN = re.compile(r"^Expenses:", re.I)
EXPENSE_CATEGORY_PATTERNS = {
    "education": re.compile(r"Education|School|Tuition|Books", re.I),
    "housing":   re.compile(r"Housing|Rent|Mortgage|Utilities|Property", re.I),
}
# ----------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip spaces, lower‑case, replace spaces & dots with underscores.
    Returns the DataFrame with renamed columns.
    """
    new_names = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace(".", "")
        new_names[col] = clean
    df = df.rename(columns=new_names)

    # Debug print – helps you see what the script sees
    print("\n=== Detected columns after normalisation ===")
    for orig, clean in new_names.items():
        print(f"  {orig!r} → {clean!r}")
    print("============================================\n")
    return df


def _parse_fraction(value: str) -> float:
    if value is None or value == "":
        return 0.0
    if "/" in value:
        num, denom = value.split("/", 1)
        return float(num) / float(denom)
    return float(value)


def load_gnucash(gnucash_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read a GnuCash native file (compressed XML) and create the canonical
    columns that the rest of the pipeline expects:
        date, full_account_name, amount (float), value (float)
    """
    try:
        with gzip.open(gnucash_path, "rb") as fh:
            tree = ET.parse(fh)
    except OSError:
        tree = ET.parse(gnucash_path)

    root = tree.getroot()
    ns = {
        "gnc": "http://www.gnucash.org/XML/gnc",
        "act": "http://www.gnucash.org/XML/act",
        "trn": "http://www.gnucash.org/XML/trn",
        "split": "http://www.gnucash.org/XML/split",
        "ts": "http://www.gnucash.org/XML/ts",
        "cmdty": "http://www.gnucash.org/XML/cmdty",
        "price": "http://www.gnucash.org/XML/price",
    }

    accounts = {}
    parents = {}
    account_commodity = {}
    for acct in root.findall(".//gnc:account", ns):
        guid = acct.findtext("act:id", default="", namespaces=ns)
        name = acct.findtext("act:name", default="", namespaces=ns)
        parent = acct.findtext("act:parent", default="", namespaces=ns)
        space = acct.findtext("act:commodity/cmdty:space", default="", namespaces=ns)
        mnemonic = acct.findtext("act:commodity/cmdty:id", default="", namespaces=ns)
        accounts[guid] = name
        parents[guid] = parent
        account_commodity[guid] = {"space": space, "id": mnemonic}

    full_name_cache = {}

    def full_name(guid: str) -> str:
        if guid in full_name_cache:
            return full_name_cache[guid]
        name = accounts.get(guid, "")
        parent = parents.get(guid, "")
        if parent and accounts.get(parent):
            parent_name = full_name(parent)
            combined = f"{parent_name}:{name}" if parent_name else name
        else:
            combined = name
        if combined.startswith("Root Account:"):
            combined = combined.replace("Root Account:", "", 1)
        full_name_cache[guid] = combined
        return combined

    rows = []
    for trn in root.findall(".//gnc:transaction", ns):
        trn_id = trn.findtext("trn:id", default="", namespaces=ns)
        desc = trn.findtext("trn:description", default="", namespaces=ns)
        date_str = trn.findtext("trn:date-posted/ts:date", default="", namespaces=ns)
        date = pd.to_datetime(date_str, errors="coerce")
        for split in trn.findall("trn:splits/trn:split", ns):
            acct_guid = split.findtext("split:account", default="", namespaces=ns)
            value = _parse_fraction(split.findtext("split:value", default="", namespaces=ns))
            quantity = _parse_fraction(split.findtext("split:quantity", default="", namespaces=ns))
            full_acct = full_name(acct_guid)
            commodity = account_commodity.get(acct_guid, {"space": "", "id": ""})
            rows.append(
                {
                    "date": date,
                    "transaction_id": trn_id,
                    "description": desc,
                    "full_account_name": full_acct,
                    "account_name": full_acct.split(":")[-1] if full_acct else "",
                    "amount_num": quantity,
                    "value_num": value,
                    "commodity_space": commodity.get("space", ""),
                    "commodity_id": commodity.get("id", ""),
                }
            )

    df = pd.DataFrame(rows)
    df["amount"] = df["amount_num"]
    df["value"] = df["value_num"]
    df["period"] = df["date"].dt.to_period("M")
    df["window_period"] = (df["date"] + pd.Timedelta(days=15)).dt.to_period("M")
    df["year"] = df["date"].dt.year
    pricedb = {}
    for price in root.findall(".//gnc:pricedb/price:price", ns):
        space = price.findtext("price:commodity/cmdty:space", default="", namespaces=ns)
        mnemonic = price.findtext("price:commodity/cmdty:id", default="", namespaces=ns)
        date_str = price.findtext("price:time/ts:date", default="", namespaces=ns)
        value_str = price.findtext("price:value", default="", namespaces=ns)
        if not mnemonic or not date_str or not value_str:
            continue
        if space.upper() == "CURRENCY":
            continue
        date = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(date):
            continue
        value = _parse_fraction(value_str)
        pricedb.setdefault(mnemonic, []).append((date, value))
    return df, pricedb


def classify_account(name: str) -> str:
    """Return 'bank', 'retirement', 'investment' or 'other'."""
    for typ, pat in ACCOUNT_TYPE_PATTERNS.items():
        if pat.search(name):
            return typ
    return "other"


def classify_expense(name: str) -> str:
    """Return 'education', 'housing' or 'other'."""
    for cat, pat in EXPENSE_CATEGORY_PATTERNS.items():
        if pat.search(name):
            return cat
    return "other"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three derived columns:
        account_type  – bank / retirement / investment / other
        direction     – income (amount>0) or expense (amount<0)
        expense_category – education / housing / other   (only for expenses)
    """
    df = df.copy()
    df["account_type"] = df["full_account_name"].apply(classify_account)
    df["direction"]    = np.where(df["amount"] > 0, "income", "expense")
    expense_mask = df["full_account_name"].str.match(EXPENSE_ACCOUNT_PATTERN, na=False)
    df["expense_category"] = "other"
    df.loc[expense_mask, "expense_category"] = (
        df.loc[expense_mask, "full_account_name"].apply(classify_expense)
    )
    return df


def _income_transfer_ids(df: pd.DataFrame) -> pd.Index:
    """
    Return transaction_ids where at least one split is an Income account.
    This lets us treat income as transfers from Income:* into Bank/Retirement.
    """
    if "transaction_id" not in df.columns:
        return pd.Index([])
    income_tx = (
        df[df["full_account_name"].str.match(INCOME_ACCOUNT_PATTERN, na=False)]
        ["transaction_id"]
        .dropna()
        .unique()
    )
    return pd.Index(income_tx)


def _income_amount(row: pd.Series) -> float:
    """
    Use value for retirement splits (shares priced in USD),
    and amount for bank splits (already USD).
    """
    if row["account_type"] == "retirement":
        return row["value"]
    return row["amount"]


def _expense_amount(row: pd.Series) -> float:
    """Use signed value so refunds/returns offset expenses."""
    return row["value"]


def _summarize_transactions(
    group: pd.DataFrame,
    amount_col: str,
    threshold: float = 500.0,
) -> str:
    """
    Summarize items over the threshold with date and amount,
    and bucket the rest as "N other ($total)".
    """
    cols = ["date", "description", "full_account_name", amount_col]
    g = group[cols].copy()
    over = g[g[amount_col].abs() > threshold].sort_values("date")
    under = g[g[amount_col].abs() <= threshold]
    parts = []
    for _, row in over.iterrows():
        desc = row["description"] or row["full_account_name"]
        date_str = row["date"].strftime("%b %d") if pd.notnull(row["date"]) else ""
        amount = int(round(row[amount_col]))
        parts.append(f"{date_str} {desc} (${amount})")
    if not under.empty:
        total = under[amount_col].sum()
        total = int(round(total))
        parts.append(f"{len(under)} other (${total})")
    return "; ".join(parts) if parts else "No transactions"


# ----------------------------------------------------------------------
# 2️⃣  AGGREGATIONS (monthly & yearly)
# ----------------------------------------------------------------------
def monthly_income(df):
    income_tx = _income_transfer_ids(df)
    inc = df[df["account_type"].isin(["bank", "retirement"])].copy()
    if not income_tx.empty:
        inc = inc[inc["transaction_id"].isin(income_tx)]
    inc["income_amt"] = inc.apply(_income_amount, axis=1)
    out = (
        inc.groupby(["window_period", "account_type"])["income_amt"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("window_period")
    )
    out = out.rename(columns={"window_period": "period"})
    for col in ("bank", "retirement"):
        if col not in out.columns:
            out[col] = 0.0
    out["total_income"] = out[["bank", "retirement"]].sum(axis=1)
    summaries = (
        inc.groupby(["window_period", "account_type"])
        .apply(_summarize_transactions, amount_col="income_amt")
        .reset_index(name="summary")
    )
    summaries = summaries.pivot(
        index="window_period",
        columns="account_type",
        values="summary",
    ).reset_index()
    summaries = summaries.rename(
        columns={
            "window_period": "period",
            "bank": "bank_summary",
            "retirement": "retirement_summary",
        }
    )
    out = out.merge(summaries, on="period", how="left")
    for col in ("bank_summary", "retirement_summary"):
        if col not in out.columns:
            out[col] = ""
    return out


def monthly_expenses(df):
    exp = df[df["full_account_name"].str.match(EXPENSE_ACCOUNT_PATTERN, na=False)].copy()
    exp["signed_amount"] = exp.apply(_expense_amount, axis=1)
    out = (
        exp.groupby(["window_period", "expense_category"])["signed_amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("window_period")
    )
    out = out.rename(columns={"window_period": "period"})
    for col in ("education", "housing"):
        if col not in out.columns:
            out[col] = 0.0
    out["other"] = out.drop(columns="period").sum(axis=1) - out["education"] - out["housing"]
    out["total_exp"] = out[["education", "housing", "other"]].sum(axis=1)
    summaries = (
        exp.groupby(["window_period", "expense_category"])
        .apply(_summarize_transactions, amount_col="signed_amount")
        .reset_index(name="summary")
    )
    summaries = summaries.pivot(
        index="window_period",
        columns="expense_category",
        values="summary",
    ).reset_index()
    summaries = summaries.rename(
        columns={
            "window_period": "period",
            "education": "education_summary",
            "housing": "housing_summary",
            "other": "other_summary",
        }
    )
    out = out.merge(summaries, on="period", how="left")
    for col in ("education_summary", "housing_summary", "other_summary"):
        if col not in out.columns:
            out[col] = ""
    return out


def _recent_average(df: pd.DataFrame, columns: list[str], end_period: pd.Period, months: int = 12) -> dict:
    start_period = end_period - (months - 1)
    periods = pd.period_range(start_period, end_period, freq="M")
    recent = df[df["period"] <= end_period].set_index("period").reindex(periods)
    recent = recent.fillna(0).reset_index().rename(columns={"index": "period"})
    averages = {}
    for col in columns:
        if col in recent.columns and not recent.empty:
            averages[col] = float(recent[col].mean())
        else:
            averages[col] = 0.0
    return averages


def _education_projection(
    periods: pd.PeriodIndex,
    exp_df: pd.DataFrame,
    end_period: pd.Period,
    target: float = 18000.0,
) -> pd.Series:
    proj = pd.Series(0.0, index=periods)
    if exp_df.empty:
        exp_df = pd.DataFrame(columns=["period", "education"])
    for year in sorted(set(p.year for p in periods)):
        for start_month, end_month in ((1, 4), (8, 11)):
            window_periods = [
                p for p in periods
                if p.year == year and start_month <= p.month <= end_month
            ]
            if not window_periods:
                continue
            mask = (
                (exp_df["period"] <= end_period)
                & (exp_df["period"].dt.year == year)
                & (exp_df["period"].dt.month.between(start_month, end_month))
            )
            actual_sum = exp_df.loc[mask, "education"].sum()
            remaining = max(0.0, target - actual_sum)
            per_month = remaining / len(window_periods)
            proj.loc[window_periods] = per_month
    return proj


def build_projections(
    inc_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    end_period: pd.Period,
    months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = pd.period_range(end_period + 1, end_period + months, freq="M")
    inc_avg = _recent_average(inc_df, ["bank", "retirement"], end_period)
    exp_avg = _recent_average(exp_df, ["housing", "other"], end_period)
    proj_inc = pd.DataFrame(
        {
            "period": periods,
            "bank": inc_avg["bank"],
            "retirement": inc_avg["retirement"],
            "bank_summary": "Projection",
            "retirement_summary": "Projection",
        }
    )
    education_proj = _education_projection(periods, exp_df, end_period)
    proj_exp = pd.DataFrame(
        {
            "period": periods,
            "education": education_proj.values,
            "housing": exp_avg["housing"],
            "other": exp_avg["other"],
            "education_summary": "Projection",
            "housing_summary": "Projection",
            "other_summary": "Projection",
        }
    )
    proj_exp["total_exp"] = proj_exp[["education", "housing", "other"]].sum(axis=1)
    return proj_inc, proj_exp


def _net_window_details(net_df: pd.DataFrame, window: int = 6) -> pd.Series:
    """
    Build hover details for net cash flow: month-by-month income/expense/net
    within a centered window.
    """
    details = []
    periods = net_df["period"].tolist()
    for idx, period in enumerate(periods):
        start_idx = max(0, idx - window)
        end_idx = min(len(periods) - 1, idx + window)
        rows = net_df.iloc[start_idx : end_idx + 1]
        parts = []
        for _, row in rows.iterrows():
            if pd.isna(row["net_bank_flow"]):
                continue
            label = row["period"].strftime("%b %Y")
            income = int(round(row["bank"])) if not pd.isna(row["bank"]) else 0
            exp = int(round(row["total_exp"])) if not pd.isna(row["total_exp"]) else 0
            net = int(round(row["net_bank_flow"]))
            parts.append(f"{label} +{income} -{exp} = {net}")
        details.append("<br>".join(parts) if parts else "")
    return pd.Series(details, index=net_df.index)


def build_net_projection(proj_inc: pd.DataFrame, proj_exp: pd.DataFrame) -> pd.DataFrame:
    if proj_inc is None or proj_exp is None or proj_inc.empty or proj_exp.empty:
        return pd.DataFrame(columns=["period", "net_bank_flow", "summary"])
    merged = pd.merge(
        proj_inc[["period", "bank"]],
        proj_exp[["period", "total_exp"]],
        on="period",
        how="inner",
    )
    merged["net_bank_flow"] = merged["bank"] - merged["total_exp"]
    merged["summary"] = "Projection"
    return merged


def build_values_projection(
    values_df: pd.DataFrame,
    proj_net_df: pd.DataFrame,
    end_period: pd.Period,
) -> pd.DataFrame:
    if values_df.empty or proj_net_df.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "bank",
                "investment",
                "retirement",
                "bank_summary",
                "investment_summary",
                "retirement_summary",
            ]
        )
    last_actual = values_df[values_df["period"] <= end_period]
    if last_actual.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "bank",
                "investment",
                "retirement",
                "bank_summary",
                "investment_summary",
                "retirement_summary",
            ]
        )
    last_row = last_actual.sort_values("period").iloc[-1]
    liquidation = last_row["investment"] + last_row["retirement"]
    start_bank = last_row["bank"] + liquidation
    proj = proj_net_df.sort_values("period").copy()
    proj["bank"] = start_bank + proj["net_bank_flow"].cumsum()
    proj["investment"] = 0.0
    proj["retirement"] = 0.0
    proj["bank_summary"] = "Projection (liquidated investments)"
    proj["investment_summary"] = "Projection (sold)"
    proj["retirement_summary"] = "Projection (sold)"
    return proj[
        [
            "period",
            "bank",
            "investment",
            "retirement",
            "bank_summary",
            "investment_summary",
            "retirement_summary",
        ]
    ]


def build_yearly_account_table(values_df: pd.DataFrame, proj_values_df: pd.DataFrame | None) -> list[dict]:
    if values_df.empty and (proj_values_df is None or proj_values_df.empty):
        return []
    combined = values_df.copy()
    if proj_values_df is not None and not proj_values_df.empty:
        combined = pd.concat([combined, proj_values_df], ignore_index=True)
    combined = combined.dropna(subset=["period"])
    combined["year"] = combined["period"].dt.year
    year_end = (
        combined.sort_values("period")
        .groupby("year")
        .tail(1)
        .sort_values("year")
        .reset_index(drop=True)
    )
    rows = []
    prev = None
    for _, row in year_end.iterrows():
        total = row["bank"] + row["investment"] + row["retirement"]
        if prev is None:
            bank_change = investment_change = retirement_change = total_change = 0.0
        else:
            bank_change = row["bank"] - prev["bank"]
            investment_change = row["investment"] - prev["investment"]
            retirement_change = row["retirement"] - prev["retirement"]
            total_change = total - (prev["bank"] + prev["investment"] + prev["retirement"])
        rows.append(
            {
                "year": int(row["year"]),
                "bank_value": int(round(row["bank"])),
                "bank_change": int(round(bank_change)),
                "investment_value": int(round(row["investment"])),
                "investment_change": int(round(investment_change)),
                "retirement_value": int(round(row["retirement"])),
                "retirement_change": int(round(retirement_change)),
                "total_value": int(round(total)),
                "total_change": int(round(total_change)),
            }
        )
        prev = row
    return rows


def _summarize_account_values(group: pd.DataFrame, threshold: float = 500.0) -> str:
    """
    Summarize per-account values above a threshold, with remaining grouped as Other.
    """
    g = group[["full_account_name", "balance"]].copy()
    g["account_label"] = g["full_account_name"].str.split(":").str[-1].fillna("")
    over = g[g["balance"].abs() > threshold].sort_values("balance", ascending=False)
    under = g[g["balance"].abs() <= threshold]
    parts = []
    for _, row in over.iterrows():
        amount = int(round(row["balance"]))
        parts.append(f"{row['account_label']} (${amount})")
    if not under.empty:
        total = int(round(under["balance"].sum()))
        parts.append(f"Other (${total})")
    return "; ".join(parts) if parts else ""


def _load_price_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_price_cache(cache_path: Path, cache: dict) -> None:
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _fetch_yahoo_prices(symbol: str, start: datetime, end: datetime) -> list[tuple[pd.Timestamp, float]]:
    period1 = int(start.replace(tzinfo=timezone.utc).timestamp())
    period2 = int(end.replace(tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval=1d&period1={period1}&period2={period2}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    result = payload.get("chart", {}).get("result", [])
    if not result:
        return []
    timestamps = result[0].get("timestamp", [])
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
    prices = []
    for ts_val, close in zip(timestamps, closes):
        if close is None:
            continue
        date = pd.to_datetime(ts_val, unit="s").tz_localize(None)
        prices.append((date, float(close)))
    return prices


def _symbol_for_commodity(space: str, mnemonic: str) -> str:
    if not mnemonic:
        return ""
    if space.upper() in ("NYSE", "NASDAQ", "AMEX", "US"):
        return mnemonic
    return mnemonic


def _price_series_for_commodity(
    space: str,
    mnemonic: str,
    periods: pd.PeriodIndex,
    split_prices: list[tuple[pd.Timestamp, float]],
    pricedb_prices: list[tuple[pd.Timestamp, float]],
    cache_path: Path,
) -> pd.Series:
    symbol = _symbol_for_commodity(space, mnemonic)
    if not symbol:
        return pd.Series(index=periods, dtype=float)
    cache = _load_price_cache(cache_path)
    cache_key = symbol.upper()
    cached_prices = []
    for item in cache.get(cache_key, []):
        try:
            cached_prices.append((pd.to_datetime(item[0]), float(item[1])))
        except (ValueError, TypeError):
            continue

    def _naive(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts

    all_prices = [
        (_naive(pd.to_datetime(d)), float(p))
        for d, p in (split_prices + pricedb_prices + cached_prices)
        if d is not None
    ]
    if all_prices:
        all_prices = sorted(all_prices, key=lambda x: x[0])
    start = periods.min().to_timestamp(how="start").tz_localize(None)
    end = periods.max().to_timestamp(how="end").tz_localize(None)
    need_fetch = True
    if all_prices:
        need_fetch = all_prices[0][0] > start or all_prices[-1][0] < end
    if need_fetch:
        fetched = _fetch_yahoo_prices(symbol, start.to_pydatetime(), end.to_pydatetime())
        if fetched:
            all_prices = sorted(all_prices + fetched, key=lambda x: x[0])
            cache[cache_key] = [(d.strftime("%Y-%m-%d"), p) for d, p in all_prices]
            _save_price_cache(cache_path, cache)

    if not all_prices:
        return pd.Series(index=periods, dtype=float)

    series = pd.Series(
        data=[p for _, p in all_prices],
        index=pd.to_datetime([d for d, _ in all_prices]),
    ).sort_index()
    series = series.groupby(series.index).last()
    period_ends = periods.to_timestamp(how="end")
    price_at_period = series.reindex(period_ends, method="ffill")
    price_at_period.index = periods
    return price_at_period


def net_cash_flow(inc_df, exp_df):
    merged = pd.merge(
        inc_df[["period", "bank"]],
        exp_df[["period", "total_exp"]],
        on="period",
        how="outer",
    )
    if merged.empty:
        merged["net_bank_flow"] = pd.Series(dtype=float)
        merged["net_bank_flow_ma13"] = pd.Series(dtype=float)
        merged["net_window_details"] = pd.Series(dtype=str)
        return merged

    start = merged["period"].min()
    end = merged["period"].max()
    full_periods = pd.period_range(start, end, freq="M")
    merged = merged.set_index("period").reindex(full_periods).rename_axis("period").reset_index()
    merged = merged.sort_values("period")
    net = merged["bank"].fillna(0) - merged["total_exp"].fillna(0)
    no_data = merged["bank"].isna() & merged["total_exp"].isna()
    net = net.mask(no_data)
    merged["net_bank_flow"] = net
    merged["net_bank_flow_ma13"] = (
        merged["net_bank_flow"]
        .rolling(window=13, min_periods=1, center=True)
        .mean()
    )
    merged["net_window_details"] = _net_window_details(merged, window=6)
    return merged


def monthly_account_values(df, pricedb: dict, cache_path: Path | None = None):
    if df.empty:
        return pd.DataFrame(columns=["period", "bank", "investment", "retirement"])

    if cache_path is None:
        cache_path = Path("price_cache.json")

    df_sorted = df.sort_values("date").copy()
    df_sorted["price"] = np.where(
        df_sorted["amount"] != 0,
        (df_sorted["value"] / df_sorted["amount"]).abs(),
        np.nan,
    )
    df_sorted["shares"] = df_sorted.groupby("full_account_name")["amount"].cumsum()
    df_sorted["last_price"] = df_sorted.groupby("full_account_name")["price"].ffill()
    df_sorted["cash_balance"] = df_sorted.groupby("full_account_name")["value"].cumsum()
    df_sorted["balance"] = df_sorted["cash_balance"]

    per_account_shares = (
        df_sorted.groupby(
            ["window_period", "full_account_name", "account_type", "commodity_id", "commodity_space"]
        )["shares"]
        .last()
        .reset_index()
    )
    per_account_cash = (
        df_sorted.groupby(["window_period", "full_account_name", "account_type"])["cash_balance"]
        .last()
        .reset_index()
    )
    full_periods = pd.period_range(
        df_sorted["window_period"].min(),
        df_sorted["window_period"].max(),
        freq="M",
    )
    account_types = (
        df_sorted.groupby("full_account_name")[["account_type", "commodity_id", "commodity_space"]]
        .first()
        .to_dict(orient="index")
    )
    filled = []
    for acct, info in account_types.items():
        acct_type = info.get("account_type")
        commodity_id = info.get("commodity_id", "")
        commodity_space = info.get("commodity_space", "")
        cash_df = per_account_cash[per_account_cash["full_account_name"] == acct].copy()
        cash_df = cash_df.set_index("window_period").reindex(full_periods)
        cash_df["cash_balance"] = cash_df["cash_balance"].ffill().fillna(0.0)
        cash_df["account_type"] = acct_type
        cash_df["full_account_name"] = acct
        cash_df = cash_df.reset_index().rename(columns={"index": "window_period"})

        if acct_type in ("investment", "retirement") and commodity_id:
            split_prices = (
                df_sorted[
                    (df_sorted["commodity_id"] == commodity_id)
                    & (df_sorted["price"].notna())
                ][["date", "price"]]
                .values.tolist()
            )
            pricedb_prices = pricedb.get(commodity_id, [])
            price_series = _price_series_for_commodity(
                commodity_space,
                commodity_id,
                full_periods,
                split_prices,
                pricedb_prices,
                cache_path,
            )
            shares_df = per_account_shares[per_account_shares["full_account_name"] == acct].copy()
            shares_df = shares_df.set_index("window_period").reindex(full_periods)
            shares_df["shares"] = shares_df["shares"].ffill().fillna(0.0)
            shares_df = shares_df.reset_index().rename(columns={"index": "window_period"})
            merged = shares_df.merge(
                price_series.rename("price").rename_axis("window_period").reset_index(),
                on="window_period",
                how="left",
            )
            merged["balance"] = merged["shares"] * merged["price"]
            merged = merged[["window_period", "balance"]]
            cash_df = cash_df.merge(merged, on="window_period", how="left")
            cash_df["balance"] = cash_df["balance"].fillna(cash_df["cash_balance"])
        else:
            cash_df["balance"] = cash_df["cash_balance"]

        filled.append(cash_df[["window_period", "account_type", "full_account_name", "balance"]])
    per_account_full = pd.concat(filled, ignore_index=True)

    summaries = (
        per_account_full.groupby(["window_period", "account_type"])
        .apply(_summarize_account_values)
        .reset_index(name="summary")
    )

    values = (
        per_account_full.groupby(["window_period", "account_type"])["balance"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("window_period")
    )
    values = values.rename(columns={"window_period": "period"})
    for col in ("bank", "investment", "retirement"):
        if col not in values.columns:
            values[col] = 0.0
    summaries = summaries.pivot(
        index="window_period",
        columns="account_type",
        values="summary",
    ).reset_index()
    summaries = summaries.rename(
        columns={
            "window_period": "period",
            "bank": "bank_summary",
            "investment": "investment_summary",
            "retirement": "retirement_summary",
        }
    )
    values = values.merge(summaries, on="period", how="left")
    for col in ("bank_summary", "investment_summary", "retirement_summary"):
        if col not in values.columns:
            values[col] = ""
    return values


def latest_balances(df):
    """
    Return the latest *value* for each top‑level account type.
    This is not plotted, but keeping the function makes the rest of the code unchanged.
    """
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    bal = {}
    for typ in ("bank", "investment", "retirement"):
        bal[typ] = latest.loc[latest["account_type"] == typ, "value"].sum()
    return bal


# ----------------------------------------------------------------------
# 3️⃣  Build Plotly JSON
# ----------------------------------------------------------------------
def build_plotly_data(
    inc_df,
    exp_df,
    net_df,
    values_df,
    balances,
    proj_inc_df=None,
    proj_exp_df=None,
    proj_net_df=None,
    proj_values_df=None,
    yearly_table=None,
):
    # Monthly Income
    inc_traces = [
        {
            "x": inc_df["period"].astype(str).tolist(),
            "y": inc_df["bank"].tolist(),
            "mode": "lines+markers",
            "name": "Bank Income",
            "customdata": inc_df["bank_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Bank Income: %{y:.0f}<br>"
                "Tx: %{customdata}<extra></extra>"
            ),
        },
        {
            "x": inc_df["period"].astype(str).tolist(),
            "y": inc_df["retirement"].tolist(),
            "mode": "lines+markers",
            "name": "Retirement Income",
            "customdata": inc_df["retirement_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Retirement Income: %{y:.0f}<br>"
                "Tx: %{customdata}<extra></extra>"
            ),
        },
    ]
    if proj_inc_df is not None and not proj_inc_df.empty:
        inc_traces.extend(
            [
                {
                    "x": proj_inc_df["period"].astype(str).tolist(),
                    "y": proj_inc_df["bank"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Bank Income",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_inc_df["bank_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Bank Income: %{y:.0f}<br>"
                        "Tx: %{customdata}<extra></extra>"
                    ),
                },
                {
                    "x": proj_inc_df["period"].astype(str).tolist(),
                    "y": proj_inc_df["retirement"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Retirement Income",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_inc_df["retirement_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Retirement Income: %{y:.0f}<br>"
                        "Tx: %{customdata}<extra></extra>"
                    ),
                },
            ]
        )

    # Monthly Expenses
    exp_traces = [
        {
            "x": exp_df["period"].astype(str).tolist(),
            "y": exp_df["education"].tolist(),
            "mode": "lines+markers",
            "name": "Education",
            "customdata": exp_df["education_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Education: %{y:.0f}<br>"
                "Tx: %{customdata}<extra></extra>"
            ),
        },
        {
            "x": exp_df["period"].astype(str).tolist(),
            "y": exp_df["housing"].tolist(),
            "mode": "lines+markers",
            "name": "Housing",
            "customdata": exp_df["housing_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Housing: %{y:.0f}<br>"
                "Tx: %{customdata}<extra></extra>"
            ),
        },
        {
            "x": exp_df["period"].astype(str).tolist(),
            "y": exp_df["other"].tolist(),
            "mode": "lines+markers",
            "name": "Other",
            "customdata": exp_df["other_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Other: %{y:.0f}<br>"
                "Tx: %{customdata}<extra></extra>"
            ),
        },
    ]
    if proj_exp_df is not None and not proj_exp_df.empty:
        exp_traces.extend(
            [
                {
                    "x": proj_exp_df["period"].astype(str).tolist(),
                    "y": proj_exp_df["education"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Education",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_exp_df["education_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Education: %{y:.0f}<br>"
                        "Tx: %{customdata}<extra></extra>"
                    ),
                },
                {
                    "x": proj_exp_df["period"].astype(str).tolist(),
                    "y": proj_exp_df["housing"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Housing",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_exp_df["housing_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Housing: %{y:.0f}<br>"
                        "Tx: %{customdata}<extra></extra>"
                    ),
                },
                {
                    "x": proj_exp_df["period"].astype(str).tolist(),
                    "y": proj_exp_df["other"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Other",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_exp_df["other_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Other: %{y:.0f}<br>"
                        "Tx: %{customdata}<extra></extra>"
                    ),
                },
            ]
        )

    # Net Cash Flow
    net_trace = [
        {
            "x": net_df["period"].astype(str).tolist(),
            "y": net_df["net_bank_flow_ma13"].tolist(),
            "mode": "lines+markers",
            "name": "Net Bank Flow (13-mo MA)",
            "line": {"color": "#ff9900"},
            "customdata": net_df["net_window_details"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Net (MA): %{y:.0f}<br>"
                "%{customdata}<extra></extra>"
            ),
        }
    ]
    if proj_net_df is not None and not proj_net_df.empty:
        net_trace.append(
            {
                "x": proj_net_df["period"].astype(str).tolist(),
                "y": proj_net_df["net_bank_flow"].tolist(),
                "mode": "lines+markers",
                "name": "Projected Net Bank Flow",
                "line": {"dash": "dot", "color": "#ff9900"},
                "opacity": 0.6,
                "customdata": proj_net_df["summary"].fillna("").tolist(),
                "hovertemplate": (
                    "Month: %{x}<br>"
                    "Projected Net: %{y:.0f}<br>"
                    "%{customdata}<extra></extra>"
                ),
            }
        )

    # Account values by month (bank, investment, retirement) – stacked bars
    months = values_df["period"].astype(str).tolist()
    value_traces = [
        {
            "x": months,
            "y": values_df["bank"].tolist(),
            "mode": "lines+markers",
            "name": "Bank Value",
            "customdata": values_df["bank_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Bank: %{y:.0f}<br>"
                "%{customdata}<extra></extra>"
            ),
        },
        {
            "x": months,
            "y": values_df["investment"].tolist(),
            "mode": "lines+markers",
            "name": "Investment Value",
            "customdata": values_df["investment_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Investment: %{y:.0f}<br>"
                "%{customdata}<extra></extra>"
            ),
        },
        {
            "x": months,
            "y": values_df["retirement"].tolist(),
            "mode": "lines+markers",
            "name": "Retirement Value",
            "customdata": values_df["retirement_summary"].fillna("").tolist(),
            "hovertemplate": (
                "Month: %{x}<br>"
                "Retirement: %{y:.0f}<br>"
                "%{customdata}<extra></extra>"
            ),
        },
    ]
    if proj_values_df is not None and not proj_values_df.empty:
        value_traces.extend(
            [
                {
                    "x": proj_values_df["period"].astype(str).tolist(),
                    "y": proj_values_df["bank"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Bank Value",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_values_df["bank_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Bank: %{y:.0f}<br>"
                        "%{customdata}<extra></extra>"
                    ),
                },
                {
                    "x": proj_values_df["period"].astype(str).tolist(),
                    "y": proj_values_df["investment"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Investment Value",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_values_df["investment_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Investment: %{y:.0f}<br>"
                        "%{customdata}<extra></extra>"
                    ),
                },
                {
                    "x": proj_values_df["period"].astype(str).tolist(),
                    "y": proj_values_df["retirement"].tolist(),
                    "mode": "lines+markers",
                    "name": "Projected Retirement Value",
                    "line": {"dash": "dot"},
                    "opacity": 0.6,
                    "customdata": proj_values_df["retirement_summary"].fillna("").tolist(),
                    "hovertemplate": (
                        "Month: %{x}<br>"
                        "Projected Retirement: %{y:.0f}<br>"
                        "%{customdata}<extra></extra>"
                    ),
                },
            ]
        )

    return {
        "income": inc_traces,
        "expenses": exp_traces,
        "netflow": net_trace,
        "values": value_traces,
        "yearly_account_table": yearly_table or [],
    }


# ----------------------------------------------------------------------
# 4️⃣  Write a single self‑contained HTML file
# ----------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Financial Report – Line Charts Only</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{font-family:Arial,Helvetica,sans-serif; margin:20px;}}
  .chart {{margin-bottom:45px;}}
  table {{border-collapse:collapse;width:100%;margin-top:20px;}}
  th, td {{border:1px solid #ddd;padding:8px;text-align:center;}}
  th {{background:#f2f2f2;}}
</style>
</head>
<body>

<h1>Financial Report – Line Charts Only</h1>

<div id="income" class="chart"></div>
<div id="expenses" class="chart"></div>
<div id="netflow" class="chart"></div>
<div id="values" class="chart"></div>

<h2>Yearly Account Totals</h2>
<div id="yearly_account_table"></div>

<script>
  const plotData = {plot_json};

  // ---- Income ----
  Plotly.newPlot('income', plotData.income, {{
    title: 'Monthly Income – Bank vs Retirement',
    xaxis: {{title:'Month'}},
    yaxis: {{title:'USD'}},
    legend: {{orientation:'h'}}
  }});

  // ---- Expenses ----
  Plotly.newPlot('expenses', plotData.expenses, {{
    title: 'Monthly Expenses – Education, Housing, Other',
    xaxis: {{title:'Month'}},
    yaxis: {{title:'USD'}},
    legend: {{orientation:'h'}}
  }});

  // ---- Net Cash Flow ----
  Plotly.newPlot('netflow', plotData.netflow, {{
    title: 'Net Cash Flow (Bank Income – All Expenses)',
    xaxis: {{title:'Month'}},
    yaxis: {{title:'USD'}}
  }});

  // ---- Account Values by Year ----
  Plotly.newPlot('values', plotData.values, {{
    title: 'Account Values by Month',
    xaxis: {{title:'Month'}},
    yaxis: {{title:'USD'}}
  }});

  // ---- Yearly Account Totals ----
  const yearlyTbl = document.createElement('table');
  if (plotData.yearly_account_table.length === 0) {{
    document.getElementById('yearly_account_table').textContent = 'No yearly data available.';
  }} else {{
    const hdr = yearlyTbl.insertRow();
    const cols = Object.keys(plotData.yearly_account_table[0]);
    cols.forEach(c => {{
      const th = document.createElement('th');
      th.textContent = c;
      hdr.appendChild(th);
    }});
    plotData.yearly_account_table.forEach(row => {{
      const tr = yearlyTbl.insertRow();
      cols.forEach(c => {{
        const td = tr.insertCell();
        td.textContent = row[c];
      }});
    }});
    document.getElementById('yearly_account_table').appendChild(yearlyTbl);
  }}

</script>

</body>
</html>
"""

def write_html(out_path: Path, plot_json: dict):
    """Save the final HTML file."""
    html = HTML_TEMPLATE.format(plot_json=json.dumps(plot_json))
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Report written to: {out_path.resolve()}\n")


# ----------------------------------------------------------------------
# 5️⃣  Main driver
# ----------------------------------------------------------------------
def main(gnucash_file: str, out_file: str = "report.html", start_year: int = 2021, end_year: int = None):
    gnucash_path = Path(gnucash_file)
    out_path = Path(out_file)

    # Load & enrich data
    df_raw, pricedb = load_gnucash(gnucash_path)
    df_all = enrich(df_raw)
    end_year_is_default = end_year is None
    if end_year_is_default:
        end_period = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")
    else:
        end_period = pd.Period(f"{end_year}-12", freq="M")
    start_period = pd.Period(f"{start_year}-01", freq="M")
    df = df_all[
        df_all["window_period"].between(start_period, end_period)
    ]

    # Monthly aggregations
    inc_month = monthly_income(df)
    exp_month = monthly_expenses(df)
    net_month = net_cash_flow(inc_month, exp_month)
    values_month = monthly_account_values(df_all, pricedb)
    values_month = values_month[
        (values_month["period"] >= start_period) & (values_month["period"] <= end_period)
    ]
    proj_inc = proj_exp = None
    proj_net = proj_values = None
    if end_year_is_default:
        proj_inc, proj_exp = build_projections(inc_month, exp_month, end_period, months=12)
        proj_net = build_net_projection(proj_inc, proj_exp)
        proj_values = build_values_projection(values_month, proj_net, end_period)

    # Yearly aggregation and balances
    balances  = latest_balances(df)   # kept for completeness

    # Build Plotly JSON and write HTML
    yearly_table = build_yearly_account_table(values_month, proj_values)
    plot_json = build_plotly_data(
        inc_month,
        exp_month,
        net_month,
        values_month,
        balances,
        proj_inc_df=proj_inc,
        proj_exp_df=proj_exp,
        proj_net_df=proj_net,
        proj_values_df=proj_values,
        yearly_table=yearly_table,
    )
    write_html(out_path, plot_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an interactive financial HTML report."
    )
    parser.add_argument("gnucash", help="Path to the GnuCash .gnucash file")
    parser.add_argument("-o", "--output", default="report.html",
                        help="Filename for the HTML report (default: report.html)")
    parser.add_argument("--start-year", type=int, default=2021,
                        help="First year to include (default: 2021)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Last year to include (default: current year)")
    args = parser.parse_args()
    main(args.gnucash, args.output, args.start_year, args.end_year)
