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
import sys
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
PLOT_COLORS = {
    "bank": "#1f77b4",
    "retirement": "#ff7f0e",
    "education": "#2ca02c",
    "housing": "#d62728",
    "other": "#7f7f7f",
    "investment": "#9467bd",
    "net": "#ff9900",
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


def _parse_date_arg(value: str, *, default_month: int) -> pd.Period | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", value.strip())
    if not match:
        raise ValueError(f"Expected YYYY, YYYY-MM, or YYYY-MM-DD; got {value!r}")
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else default_month
    day = int(match.group(3)) if match.group(3) else 1
    try:
        datetime(year, month, day)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}") from exc
    return pd.Period(f"{year}-{month:02d}", freq="M")


def _render_progress(label: str, pct: float) -> None:
    pct = min(max(pct, 0.0), 1.0)
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{label} [{bar}] {pct * 100:3.0f}%")
    sys.stdout.flush()


class ProgressTracker:
    def __init__(self, label: str) -> None:
        self.label = label
        self.last_pct = -1.0

    def update(self, pct: float) -> None:
        pct = min(max(pct, 0.0), 1.0)
        if pct <= self.last_pct:
            return
        if pct - self.last_pct < 0.005 and pct < 1.0:
            return
        self.last_pct = pct
        _render_progress(self.label, pct)
        if pct >= 1.0:
            sys.stdout.write("\n")


def _read_gnucash_bytes(path: Path, progress_cb=None) -> bytes:
    total = path.stat().st_size if path.exists() else 0
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        try:
            data = bytearray()
            with path.open("rb") as raw_fh:
                with gzip.GzipFile(fileobj=raw_fh) as gz_fh:
                    while True:
                        chunk = gz_fh.read(1024 * 1024)
                        if not chunk:
                            break
                        data.extend(chunk)
                        if total > 0 and progress_cb is not None:
                            progress_cb(raw_fh.tell() / total)
            return bytes(data)
        except OSError:
            pass
    data = bytearray()
    with path.open("rb") as fh:
        read = 0
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            read += len(chunk)
            if total > 0 and progress_cb is not None:
                progress_cb(read / total)
    return bytes(data)


def load_gnucash(
    gnucash_path: Path,
    progress_tracker: ProgressTracker | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Read a GnuCash native file (compressed XML) and create the canonical
    columns that the rest of the pipeline expects:
        date, full_account_name, amount (float), value (float)
    """
    progress_cb = None
    if progress_tracker is not None:
        progress_cb = lambda pct: progress_tracker.update(0.3 * pct)
    data = _read_gnucash_bytes(gnucash_path, progress_cb=progress_cb)
    if progress_tracker is not None:
        progress_tracker.update(0.35)
    root = ET.fromstring(data)
    if progress_tracker is not None:
        progress_tracker.update(0.4)
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
    account_types = {}
    for acct in root.findall(".//gnc:account", ns):
        guid = acct.findtext("act:id", default="", namespaces=ns)
        name = acct.findtext("act:name", default="", namespaces=ns)
        parent = acct.findtext("act:parent", default="", namespaces=ns)
        act_type = acct.findtext("act:type", default="", namespaces=ns)
        space = acct.findtext("act:commodity/cmdty:space", default="", namespaces=ns)
        mnemonic = acct.findtext("act:commodity/cmdty:id", default="", namespaces=ns)
        accounts[guid] = name
        parents[guid] = parent
        account_types[guid] = act_type
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
            act_type = account_types.get(acct_guid, "")
            rows.append(
                {
                    "date": date,
                    "transaction_id": trn_id,
                    "description": desc,
                    "full_account_name": full_acct,
                    "account_name": full_acct.split(":")[-1] if full_acct else "",
                    "account_act_type": act_type,
                    "amount_num": quantity,
                    "value_num": value,
                    "commodity_space": commodity.get("space", ""),
                    "commodity_id": commodity.get("id", ""),
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
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


def classify_account(name: str, act_type: str) -> str:
    """Return 'bank', 'retirement', 'investment' or 'other'."""
    for typ, pat in ACCOUNT_TYPE_PATTERNS.items():
        if pat.search(name):
            return typ
    act_type = (act_type or "").upper()
    if act_type == "BANK":
        return "bank"
    if act_type == "STOCK":
        return "investment"
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
    df["account_type"] = df.apply(
        lambda row: classify_account(row["full_account_name"], row.get("account_act_type", "")),
        axis=1,
    )
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
    threshold: float,
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
def monthly_income(df, detail_threshold: float):
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
        inc.groupby(["window_period", "account_type"])[
            ["date", "description", "full_account_name", "income_amt"]
        ]
        .apply(_summarize_transactions, amount_col="income_amt", threshold=detail_threshold)
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


def monthly_expenses(df, detail_threshold: float):
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
        exp.groupby(["window_period", "expense_category"])[
            ["date", "description", "full_account_name", "signed_amount"]
        ]
        .apply(_summarize_transactions, amount_col="signed_amount", threshold=detail_threshold)
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
    per_month: float | None = None,
) -> pd.Series:
    proj = pd.Series(0.0, index=periods)
    if per_month is None:
        per_month = target / 4.0
    for year in sorted(set(p.year for p in periods)):
        for start_month, end_month in ((1, 4), (8, 11)):
            window_periods = [
                p for p in periods
                if p.year == year and start_month <= p.month <= end_month
            ]
            if not window_periods:
                continue
            proj.loc[window_periods] = per_month
    return proj


def build_projections(
    inc_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    anchor_period: pd.Period,
    proj_start: pd.Period,
    proj_end: pd.Period,
    retirement_pct: float | None = None,
    education_per_month: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if proj_start is None or proj_end is None or proj_start > proj_end:
        empty = pd.DataFrame(columns=["period"])
        return empty, empty
    periods = pd.period_range(proj_start, proj_end, freq="M")
    inc_avg = _recent_average(inc_df, ["bank", "retirement"], anchor_period)
    exp_avg = _recent_average(exp_df, ["housing", "other"], anchor_period)
    if retirement_pct is not None:
        pct = retirement_pct
        if pct > 1:
            pct = pct / 100.0
        pct = min(max(pct, 0.0), 1.0)
        total_inc = inc_avg["bank"] + inc_avg["retirement"]
        inc_avg["retirement"] = total_inc * pct
        inc_avg["bank"] = total_inc - inc_avg["retirement"]
    proj_inc = pd.DataFrame(
        {
            "period": periods,
            "bank": inc_avg["bank"],
            "retirement": inc_avg["retirement"],
            "bank_summary": "Projection",
            "retirement_summary": "Projection",
        }
    )
    education_proj = _education_projection(periods, exp_df, anchor_period, per_month=education_per_month)
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


def _net_current_details(net_df: pd.DataFrame) -> pd.Series:
    """Single-month hover details for projected net cash flow."""
    details = []
    for _, row in net_df.iterrows():
        label = row["period"].strftime("%b %Y")
        income = int(round(row["bank"])) if not pd.isna(row["bank"]) else 0
        exp = int(round(row["total_exp"])) if not pd.isna(row["total_exp"]) else 0
        net = int(round(row["net_bank_flow"])) if not pd.isna(row["net_bank_flow"]) else 0
        details.append(f"{label} +{income} -{exp} = {net}")
    return pd.Series(details, index=net_df.index)


def build_net_projection(proj_inc: pd.DataFrame, proj_exp: pd.DataFrame) -> pd.DataFrame:
    if proj_inc is None or proj_exp is None or proj_inc.empty or proj_exp.empty:
        return pd.DataFrame(columns=["period", "bank", "total_exp", "net_bank_flow", "net_window_details"])
    merged = pd.merge(
        proj_inc[["period", "bank"]],
        proj_exp[["period", "total_exp"]],
        on="period",
        how="inner",
    )
    merged["net_bank_flow"] = merged["bank"] - merged["total_exp"]
    merged["net_window_details"] = _net_current_details(merged)
    return merged


def build_values_projection(
    values_df: pd.DataFrame,
    proj_net_df: pd.DataFrame,
    proj_inc_df: pd.DataFrame,
    end_period: pd.Period,
    account_details: pd.DataFrame,
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
    bank = float(last_row["bank"])
    investment = float(last_row["investment"])
    retirement = float(last_row["retirement"])
    investment_accounts = []
    if account_details is not None and not account_details.empty:
        end_investments = account_details[
            (account_details["period"] == end_period)
            & (account_details["account_type"] == "investment")
        ].copy()
        if not end_investments.empty:
            end_investments = end_investments.sort_values("balance", ascending=False)
            for _, row in end_investments.iterrows():
                balance = float(row["balance"])
                if balance > 0:
                    investment_accounts.append(
                        {"name": row["full_account_name"], "balance": balance}
                    )
    remaining_investment_total = sum(item["balance"] for item in investment_accounts) or investment
    investment = remaining_investment_total

    proj = proj_net_df.sort_values("period").copy()
    proj["bank"] = np.nan
    proj["investment"] = np.nan
    proj["retirement"] = np.nan
    proj["bank_summary"] = ""
    proj["investment_summary"] = ""
    proj["retirement_summary"] = ""
    retirement_lookup = {}
    if proj_inc_df is not None and not proj_inc_df.empty:
        retirement_lookup = dict(
            zip(proj_inc_df["period"].tolist(), proj_inc_df["retirement"].tolist())
        )

    for idx, row in proj.iterrows():
        net_flow = float(row["net_bank_flow"]) if not pd.isna(row["net_bank_flow"]) else 0.0
        bank += net_flow
        retirement += float(retirement_lookup.get(row["period"], 0.0))
        sold_notes = []
        while bank < 0 and investment_accounts:
            sold = investment_accounts.pop(0)
            sold_amount = sold["balance"]
            bank += sold_amount
            investment -= sold_amount
            sold_notes.append(f"Sold {sold['name']} (${int(round(sold_amount))})")
        proj.at[idx, "bank"] = bank
        proj.at[idx, "investment"] = max(investment, 0.0)
        proj.at[idx, "retirement"] = retirement
        if sold_notes:
            note = "; ".join(sold_notes)
            proj.at[idx, "bank_summary"] = f"Projection ({note})"
            proj.at[idx, "investment_summary"] = f"Projection ({note})"
        else:
            proj.at[idx, "bank_summary"] = "Projection"
            proj.at[idx, "investment_summary"] = "Projection"
        proj.at[idx, "retirement_summary"] = "Projection"

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


def _yearly_income_expense(
    inc_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    proj_inc_df: pd.DataFrame | None,
    proj_exp_df: pd.DataFrame | None,
) -> tuple[dict[int, float], dict[int, float]]:
    inc = inc_df.copy()
    exp = exp_df.copy()
    if proj_inc_df is not None and not proj_inc_df.empty:
        inc = pd.concat([inc, proj_inc_df], ignore_index=True)
    if proj_exp_df is not None and not proj_exp_df.empty:
        exp = pd.concat([exp, proj_exp_df], ignore_index=True)
    if inc.empty:
        inc_totals = {}
    else:
        inc["year"] = inc["period"].dt.year
        inc["total_income"] = inc.get("bank", 0.0) + inc.get("retirement", 0.0)
        inc_totals = inc.groupby("year")["total_income"].sum().to_dict()
    if exp.empty:
        exp_totals = {}
    else:
        exp["year"] = exp["period"].dt.year
        exp_totals = exp.groupby("year")["total_exp"].sum().to_dict()
    return inc_totals, exp_totals


def build_yearly_account_table(
    values_df: pd.DataFrame,
    proj_values_df: pd.DataFrame | None,
    end_period: pd.Period,
    inc_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    proj_inc_df: pd.DataFrame | None,
    proj_exp_df: pd.DataFrame | None,
) -> list[dict]:
    if values_df.empty and (proj_values_df is None or proj_values_df.empty):
        return []
    combined = values_df.copy()
    if proj_values_df is not None and not proj_values_df.empty:
        combined = pd.concat([combined, proj_values_df], ignore_index=True)
    combined = combined.dropna(subset=["period"])
    combined["year"] = combined["period"].dt.year
    combined_sorted = combined.sort_values("period")
    year_groups = combined_sorted.groupby("year")
    year_end = year_groups.tail(1).sort_values("year").reset_index(drop=True)
    year_start = year_groups.head(1).set_index("year")
    inc_totals, exp_totals = _yearly_income_expense(inc_df, exp_df, proj_inc_df, proj_exp_df)
    rows = []
    for _, row in year_end.iterrows():
        start_row = year_start.loc[row["year"]]
        total = row["bank"] + row["investment"] + row["retirement"]
        bank_change = row["bank"] - start_row["bank"]
        investment_change = row["investment"] - start_row["investment"]
        retirement_change = row["retirement"] - start_row["retirement"]
        total_change = total - (start_row["bank"] + start_row["investment"] + start_row["retirement"])
        projected = bool(row["period"] > end_period)
        rows.append(
            {
                "income_total": int(round(inc_totals.get(int(row["year"]), 0.0))),
                "expense_total": int(round(-abs(exp_totals.get(int(row["year"]), 0.0)))),
                "year": int(row["year"]),
                "bank_value": int(round(row["bank"])),
                "bank_change": int(round(bank_change)),
                "investment_value": int(round(row["investment"])),
                "investment_change": int(round(investment_change)),
                "retirement_value": int(round(row["retirement"])),
                "retirement_change": int(round(retirement_change)),
                "total_value": int(round(total)),
                "total_change": int(round(total_change)),
                "projected": projected,
            }
        )
    return rows


def _expense_major_category(name: str) -> str:
    if not name:
        return "Uncategorized"
    if name.startswith("Expenses:"):
        remainder = name.split("Expenses:", 1)[1]
        head = remainder.split(":", 1)[0].strip()
        return head or "Uncategorized"
    return "Uncategorized"


def build_yearly_expense_pies(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    threshold: float = 500.0,
) -> list[dict]:
    exp = df[df["full_account_name"].str.match(EXPENSE_ACCOUNT_PATTERN, na=False)].copy()
    exp = exp[pd.notna(exp["date"])]
    if exp.empty:
        return []
    exp["year"] = exp["date"].dt.year
    exp = exp[exp["year"].between(start_year, end_year)]
    if exp.empty:
        return []
    exp["major_category"] = exp["full_account_name"].apply(_expense_major_category)
    exp["signed_value"] = exp["value"]
    exp["abs_value"] = exp["signed_value"].abs()

    grouped = exp.groupby(["year", "major_category"])["signed_value"].sum().reset_index()

    pies = []
    for year in sorted(grouped["year"].unique()):
        year_groups = grouped[grouped["year"] == year].copy()
        year_groups = year_groups[year_groups["signed_value"] > 0]
        if year_groups.empty:
            continue
        year_groups = year_groups.sort_values("signed_value", ascending=False)
        year_total = float(year_groups["signed_value"].sum())
        labels = []
        values = []
        hovers = []
        for _, row in year_groups.iterrows():
            category = row["major_category"]
            cat_total = float(row["signed_value"])
            labels.append(category)
            values.append(cat_total)
            big_items = exp[
                (exp["year"] == year)
                & (exp["major_category"] == category)
                & (exp["abs_value"] >= threshold)
            ].copy()
            if big_items.empty:
                details = "No expenses >= $1000"
            else:
                big_items = big_items.sort_values("abs_value", ascending=False)
                parts = []
                for _, item in big_items.iterrows():
                    label = item["description"] or item["full_account_name"].split(":")[-1]
                    date_str = item["date"].strftime("%b %d")
                    amount = int(round(item["signed_value"]))
                    amount_str = f"${abs(amount)}"
                    if amount < 0:
                        amount_str = f"(-{amount_str})"
                    parts.append(f"{date_str} {label} ({amount_str})")
                details = "<br>".join(parts)
            pct = 0.0
            if year_total > 0:
                pct = (cat_total / year_total) * 100.0
            hover = (
                f"Category total: ${int(round(cat_total))} ({pct:.0f}%)<br>"
                f"{details}"
            )
            hovers.append(hover)
        pies.append(
            {
                "year": int(year),
                "labels": labels,
                "values": values,
                "hover": hovers,
            }
        )
    return pies


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
    if not mnemonic or not space:
        return ""
    public_spaces = {"NYSE", "NASDAQ", "AMEX", "US"}
    if space.upper() in public_spaces:
        return mnemonic
    return ""

def _price_series_for_commodity(
    space: str,
    mnemonic: str,
    periods: pd.PeriodIndex,
    split_prices: list[tuple[pd.Timestamp, float]],
    pricedb_prices: list[tuple[pd.Timestamp, float]],
    cache_path: Path,
) -> pd.Series:
    symbol = _symbol_for_commodity(space, mnemonic)
    cache = {}
    cached_prices = []
    if symbol:
        cache = _load_price_cache(cache_path)
        cache_key = symbol.upper()
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
    start = periods.min().to_timestamp(how="start").tz_localize(None).floor("s")
    end = periods.max().to_timestamp(how="end").tz_localize(None).ceil("s")
    if symbol and not cached_prices:
        need_fetch = True
        if all_prices:
            need_fetch = all_prices[0][0] > start or all_prices[-1][0] < end
        if need_fetch:
            fetched = _fetch_yahoo_prices(symbol, start.to_pydatetime(), end.to_pydatetime())
            if fetched:
                all_prices = sorted(all_prices + fetched, key=lambda x: x[0])
                cache_key = symbol.upper()
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


def monthly_account_values(
    df,
    pricedb: dict,
    cache_path: Path | None = None,
    return_details: bool = False,
):
    if df.empty:
        values = pd.DataFrame(columns=["period", "bank", "investment", "retirement"])
        if return_details:
            return values, pd.DataFrame(
                columns=["period", "account_type", "full_account_name", "balance"]
            )
        return values

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
        per_account_full.groupby(["window_period", "account_type"])[
            ["full_account_name", "balance"]
        ]
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
    if return_details:
        details = per_account_full.rename(columns={"window_period": "period"})
        return values, details
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
    yearly_expense_pies=None,
):
    # Monthly Income
    inc_traces = [
        {
            "x": inc_df["period"].astype(str).tolist(),
            "y": inc_df["bank"].tolist(),
            "mode": "lines+markers",
            "name": "Bank Income",
            "line": {"color": PLOT_COLORS["bank"]},
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
            "line": {"color": PLOT_COLORS["retirement"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["bank"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["retirement"]},
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
            "line": {"color": PLOT_COLORS["education"]},
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
            "line": {"color": PLOT_COLORS["housing"]},
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
            "line": {"color": PLOT_COLORS["other"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["education"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["housing"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["other"]},
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
            "line": {"color": PLOT_COLORS["net"]},
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
                "line": {"dash": "dot", "color": PLOT_COLORS["net"]},
                "opacity": 0.6,
                "customdata": proj_net_df["net_window_details"].fillna("").tolist(),
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
            "line": {"color": PLOT_COLORS["bank"]},
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
            "line": {"color": PLOT_COLORS["investment"]},
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
            "line": {"color": PLOT_COLORS["retirement"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["bank"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["investment"]},
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
                    "line": {"dash": "dot", "color": PLOT_COLORS["retirement"]},
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
        "yearly_expense_pies": yearly_expense_pies or [],
    }


# ----------------------------------------------------------------------
# 4️⃣  Write a single self‑contained HTML file
# ----------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Financial Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{font-family:Arial,Helvetica,sans-serif; margin:20px;}}
  .chart-section {{margin-bottom:45px;}}
  .chart {{margin-bottom:0;}}
  table {{border-collapse:collapse;width:100%;margin-top:20px;}}
  th, td {{border:1px solid #ddd;padding:8px;text-align:center;}}
  th {{background:#f2f2f2;}}
  td.negative {{color:#b00020;}}
  td.projected {{font-style:italic;}}
</style>
</head>
<body>

<h1>Financial Report</h1>

<div class="chart-section">
  <div id="income" class="chart"></div>
</div>

<div class="chart-section">
  <div id="expenses" class="chart"></div>
</div>

<div class="chart-section">
  <div id="netflow" class="chart"></div>
</div>

<div class="chart-section">
  <div id="values" class="chart"></div>
</div>

<h2>Yearly Expense Categories</h2>
<div id="expense_pies" class="chart"></div>

<h2>Yearly Account Totals</h2>
<p>Values reflect the end of each year, with change shown relative to the first month of that year.</p>
<div id="yearly_account_table"></div>

<script>
  const plotData = {plot_json};
  const gridColor = 'rgba(0,0,0,0.08)';
  const yearLineColor = 'rgba(0,0,0,0.12)';

  function cloneTraces(traces) {{
    return traces.map(trace => Object.assign({{}}, trace));
  }}

  function yearChangeShapes(traces) {{
    const yearStarts = new Set();
    traces.forEach(trace => {{
      (trace.x || []).forEach(x => {{
        if (typeof x === 'string' && x.endsWith('-01')) {{
          yearStarts.add(x);
        }}
      }});
    }});
    return Array.from(yearStarts).sort().map(x => ({{
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: x,
      x1: x,
      y0: 0,
      y1: 1,
      line: {{color: yearLineColor, width: 1}},
    }}));
  }}

  function percentile99(values) {{
    if (!values.length) {{
      return 0;
    }}
    const sorted = values.slice().sort((a, b) => a - b);
    const idx = Math.floor(0.99 * (sorted.length - 1));
    return sorted[idx];
  }}

  function applyOutlierClipping(traces) {{
    const next = cloneTraces(traces);
    const combined = [];
    traces.forEach(trace => {{
      (trace.y || []).forEach(val => {{
        if (typeof val === 'number' && !Number.isNaN(val)) {{
          combined.push(val);
        }}
      }});
    }});
    const z = percentile99(combined);
    if (z <= 0) {{
      return next;
    }}
    const cap = 2 * z;
    next.forEach((trace, idx) => {{
      const originalY = (traces[idx].y || []).slice();
      const originalCustom = traces[idx].customdata || [];
      const marker = Object.assign({{}}, trace.marker || {{}});
      const symbols = originalY.map(val => (val > cap ? 'cross' : 'circle'));
      trace.marker = Object.assign(marker, {{symbol: symbols}});
      trace.customdata = originalY.map((val, i) => [val, originalCustom[i] || '']);
      trace.y = originalY.map(val => (val > cap ? cap : val));
      if (trace.hovertemplate) {{
        trace.hovertemplate = trace.hovertemplate
          .replace(/%{{y(?::[^}}]*)?}}/g, '%{{customdata[0]:.0f}}')
          .replace(/%{{customdata}}/g, '%{{customdata[1]}}');
      }}
    }});
    return next;
  }}

  function buildLayout(title, traces) {{
    const shapes = yearChangeShapes(traces);
    const baseLayout = {{
      title,
      xaxis: {{title: 'Month'}},
      legend: {{orientation: 'h', x: 0.5, xanchor: 'center', y: -0.25, yanchor: 'top'}},
      margin: {{t: 50, b: 80}},
      shapes,
    }};
    baseLayout.yaxis = {{
      title: 'USD',
      showgrid: true,
      gridcolor: gridColor,
      gridwidth: 1,
      autorange: true,
    }};
    return baseLayout;
  }}

  // ---- Income ----
  const incomeTraces = applyOutlierClipping(plotData.income);
  Plotly.newPlot(
    'income',
    incomeTraces,
    buildLayout('Monthly Income – Bank vs Retirement', incomeTraces)
  );

  // ---- Expenses ----
  const expenseTraces = applyOutlierClipping(plotData.expenses);
  Plotly.newPlot(
    'expenses',
    expenseTraces,
    buildLayout('Monthly Expenses – Education, Housing, Other', expenseTraces)
  );

  // ---- Net Cash Flow ----
  const netflowTraces = applyOutlierClipping(plotData.netflow);
  Plotly.newPlot(
    'netflow',
    netflowTraces,
    buildLayout('Net Cash Flow (Bank Income – All Expenses)', netflowTraces)
  );

  // ---- Account Values by Year ----
  const valueTraces = applyOutlierClipping(plotData.values);
  Plotly.newPlot(
    'values',
    valueTraces,
    buildLayout('Account Values by Month', valueTraces)
  );

  // ---- Yearly Expense Pies ----
  const pieContainer = document.getElementById('expense_pies');
  if (!plotData.yearly_expense_pies || plotData.yearly_expense_pies.length === 0) {{
    pieContainer.textContent = 'No yearly expense data available.';
  }} else {{
    const pieTraces = [];
    const annotations = [];
    const cols = plotData.yearly_expense_pies.length;
    const globalTotals = {{}};
    plotData.yearly_expense_pies.forEach(yearData => {{
      yearData.labels.forEach((label, i) => {{
        globalTotals[label] = (globalTotals[label] || 0) + (yearData.values[i] || 0);
      }});
    }});
    const labelOrder = Object.keys(globalTotals).sort((a, b) => globalTotals[b] - globalTotals[a]);
    const palette = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
      '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
      '#dbdb8d', '#9edae5',
    ];
    const colorMap = {{}};
    labelOrder.forEach((label, idx) => {{
      colorMap[label] = palette[idx % palette.length];
    }});
    labelOrder.forEach(label => {{
    pieTraces.push({{
      type: 'scatter',
      mode: 'markers',
      x: [0],
      y: [0],
      name: label,
      marker: {{color: colorMap[label], size: 10, symbol: 'square'}},
      hoverinfo: 'skip',
      showlegend: true,
    }});
  }});
    plotData.yearly_expense_pies.forEach((yearData, idx) => {{
      const ordered = labelOrder.filter(label => yearData.labels.includes(label));
      const values = ordered.map(label => yearData.values[yearData.labels.indexOf(label)] || 0);
      const hover = ordered.map(label => yearData.hover[yearData.labels.indexOf(label)] || '');
      const colors = ordered.map(label => colorMap[label]);
      pieTraces.push({{
        type: 'pie',
        labels: ordered,
        values,
        textinfo: 'none',
        hovertemplate: '%{{label}}<br>%{{customdata}}<extra></extra>',
        customdata: hover,
        marker: {{colors}},
        sort: false,
        domain: {{row: 0, column: idx}},
        name: String(yearData.year),
        showlegend: false,
      }});
      annotations.push({{
        text: String(yearData.year),
        x: (idx + 0.5) / cols,
        y: 1.12,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: {{size: 12}},
      }});
    }});
    Plotly.newPlot('expense_pies', pieTraces, {{
      grid: {{rows: 1, columns: cols}},
      showlegend: true,
      legend: {{x: 1.02, xanchor: 'left', y: 0.5, yanchor: 'middle'}},
      margin: {{t: 40, b: 20, l: 10, r: 140}},
      xaxis: {{visible: false}},
      yaxis: {{visible: false}},
      annotations,
    }});
  }}

  // ---- Yearly Account Totals ----
  const yearlyTbl = document.createElement('table');
  if (plotData.yearly_account_table.length === 0) {{
    document.getElementById('yearly_account_table').textContent = 'No yearly data available.';
  }} else {{
    const hdr = yearlyTbl.insertRow();
    const allCols = Object.keys(plotData.yearly_account_table[0]).filter(c => c !== 'projected');
    const cols = ['year', ...allCols.filter(c => c !== 'year')];
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
        if (row.projected) {{
          td.classList.add('projected');
        }}
        if (typeof row[c] === 'number' && row[c] < 0) {{
          td.classList.add('negative');
        }}
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
    return out_path.resolve()


# ----------------------------------------------------------------------
# 5️⃣  Main driver
# ----------------------------------------------------------------------
def main(
    gnucash_file: str,
    out_file: str = "report.html",
    start: str = "2021",
    end: str | None = None,
    proj_start: str | None = None,
    proj_end: str | None = None,
    detail_threshold: float = 500.0,
    retirement_pct: float | None = None,
    education_per_month: float | None = None,
):
    gnucash_path = Path(gnucash_file)
    out_path = Path(out_file)

    start_period = _parse_date_arg(start, default_month=1)
    if start_period is None:
        raise ValueError("Start date is required.")
    if end is None:
        end_period = pd.Period(datetime.now().strftime("%Y-%m"), freq="M")
    else:
        end_period = _parse_date_arg(end, default_month=12)
    if end_period is None:
        raise ValueError("End date is required.")
    if start_period > end_period:
        raise ValueError("Start date must be on or before end date.")

    proj_start_period = _parse_date_arg(proj_start, default_month=1)
    proj_end_period = _parse_date_arg(proj_end, default_month=12)

    # Load & enrich data
    progress = ProgressTracker("Generating report")
    progress.update(0.02)
    df_raw, pricedb = load_gnucash(gnucash_path, progress_tracker=progress)
    df_all = enrich(df_raw)
    progress.update(0.5)
    end_year_is_default = end is None
    df = df_all[
        df_all["window_period"].between(start_period, end_period)
    ]

    # Monthly aggregations
    inc_month = monthly_income(df, detail_threshold)
    exp_month = monthly_expenses(df, detail_threshold)
    net_month = net_cash_flow(inc_month, exp_month)
    progress.update(0.7)
    values_month, account_details = monthly_account_values(
        df_all,
        pricedb,
        return_details=True,
    )
    progress.update(0.82)
    values_month = values_month[
        (values_month["period"] >= start_period) & (values_month["period"] <= end_period)
    ]
    proj_inc = proj_exp = None
    proj_net = proj_values = None
    if proj_start_period is not None or proj_end_period is not None:
        if proj_start_period is None:
            proj_start_period = end_period + 1
        if proj_end_period is None:
            proj_end_period = proj_start_period + 11
        if proj_start_period > proj_end_period:
            raise ValueError("Projection start must be on or before projection end.")
        proj_inc, proj_exp = build_projections(
            inc_month,
            exp_month,
            end_period,
            proj_start_period,
            proj_end_period,
            retirement_pct=retirement_pct,
            education_per_month=education_per_month,
        )
        proj_net = build_net_projection(proj_inc, proj_exp)
        proj_values = build_values_projection(
            values_month,
            proj_net,
            proj_inc,
            end_period,
            account_details,
        )
    elif end_year_is_default:
        proj_start_period = end_period + 1
        proj_end_period = end_period + 12
        proj_inc, proj_exp = build_projections(
            inc_month,
            exp_month,
            end_period,
            proj_start_period,
            proj_end_period,
            retirement_pct=retirement_pct,
            education_per_month=education_per_month,
        )
        proj_net = build_net_projection(proj_inc, proj_exp)
        proj_values = build_values_projection(
            values_month,
            proj_net,
            proj_inc,
            end_period,
            account_details,
        )
    progress.update(0.9)

    # Yearly aggregation and balances
    balances  = latest_balances(df)   # kept for completeness

    # Build Plotly JSON and write HTML
    yearly_table = build_yearly_account_table(
        values_month,
        proj_values,
        end_period,
        inc_month,
        exp_month,
        proj_inc,
        proj_exp,
    )
    yearly_expense_pies = build_yearly_expense_pies(
        df,
        start_period.year,
        end_period.year,
        threshold=detail_threshold,
    )
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
        yearly_expense_pies=yearly_expense_pies,
    )
    progress.update(0.96)
    report_path = write_html(out_path, plot_json)
    progress.update(1.0)
    print(f"✅ Report written to: {report_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an interactive financial HTML report."
    )
    parser.add_argument("gnucash", help="Path to the GnuCash .gnucash file")
    parser.add_argument("-o", "--output", default="report.html",
                        help="Filename for the HTML report (default: report.html)")
    parser.add_argument("--start", default="2021",
                        help="Start date for report window (YYYY, YYYY-MM, or YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="End date for report window (YYYY, YYYY-MM, or YYYY-MM-DD)")
    parser.add_argument("--proj-start", default=None,
                        help="Start date for projections (YYYY, YYYY-MM, or YYYY-MM-DD)")
    parser.add_argument("--proj-end", default=None,
                        help="End date for projections (YYYY, YYYY-MM, or YYYY-MM-DD)")
    parser.add_argument("--detail-threshold", type=float, default=500.0,
                        help="Detail threshold for transaction summaries (default: 500)")
    parser.add_argument("--proj-retirement-pct", type=float, default=None,
                        help="Retirement share of projected income (0-1 or 0-100)")
    parser.add_argument("--proj-education-per-month", type=float, default=None,
                        help="Projected education expense per active month")
    args = parser.parse_args()
    try:
        main(
            args.gnucash,
            args.output,
            args.start,
            args.end,
            proj_start=args.proj_start,
            proj_end=args.proj_end,
            detail_threshold=args.detail_threshold,
            retirement_pct=args.proj_retirement_pct,
            education_per_month=args.proj_education_per_month,
        )
    except ValueError as exc:
        parser.error(str(exc))
