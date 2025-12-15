# app.py
# Snap Finance — Application Performance Dashboard (Take-home)

from __future__ import annotations

from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(
    page_title="Snap Finance — Application Performance Dashboard",
    layout="wide",
)

st.markdown(
    """
<style>
/* Card border for st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"]{
    border: 3px solid #000 !important;
    border-radius: 0px !important;
    padding: 14px !important;
}
.block-container { padding-top: 1.2rem; }
[data-testid="stMetricLabel"] p { font-size: 0.9rem; }
h2, h3 { margin-bottom: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
DEFAULT_XLSX = str(APP_DIR / "sample_datasets.xlsx")

US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}

YEAR_COLOR_MAP = {
    "2022": "#5DA9E9",
    "2023": "#F28E2B",
    "2024": "#59A14F",
    "2025": "#E15759",
}

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# -----------------------------
# Helpers
# -----------------------------
def time_bucket(dt_series: pd.Series, granularity: str) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    if granularity == "Daily":
        return dt.dt.to_period("D").dt.to_timestamp()
    if granularity == "Weekly":
        return dt.dt.to_period("W").dt.start_time
    if granularity == "Monthly":
        return dt.dt.to_period("M").dt.to_timestamp()
    if granularity == "Quarterly":
        return dt.dt.to_period("Q").dt.start_time
    return dt.dt.to_period("M").dt.to_timestamp()


def safe_pct(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def safe_div_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return (numer / denom).fillna(0.0)


def normalize_state_to_abbr(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(US_STATE_ABBR)
    s = s.apply(lambda x: x.upper() if isinstance(x, str) and len(x) == 2 else x)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return s


def drop_partial_first_week(df: pd.DataFrame, date_col: str, bucket_col: str) -> pd.DataFrame:
    """
    Weekly bucketing can create a partial first week (few days) causing an artificial spike.
    Remove the first bucket if it covers < 7 days.
    """
    if df.empty or bucket_col not in df.columns or date_col not in df.columns:
        return df

    cov = (
        df.dropna(subset=[bucket_col, date_col])
        .groupby(bucket_col)[date_col]
        .agg(min_dt="min", max_dt="max")
        .sort_index()
        .reset_index()
    )
    if len(cov) <= 1:
        return df

    first_bucket = cov.iloc[0][bucket_col]
    coverage_days = (cov.iloc[0]["max_dt"] - cov.iloc[0]["min_dt"]).days + 1

    if coverage_days < 7:
        return df[df[bucket_col] != first_bucket].copy()

    return df


def month_label(m: int) -> str:
    return calendar.month_abbr[int(m)]  # 1 -> Jan


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "$0"
    return f"${x:,.0f}"


# -----------------------------
# Load data (marketing join: customers.campaign -> marketing.id)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(xlsx_path: str):
    customers = pd.read_excel(xlsx_path, sheet_name="customers")
    applications = pd.read_excel(xlsx_path, sheet_name="applications")
    stores = pd.read_excel(xlsx_path, sheet_name="stores")
    marketing = pd.read_excel(xlsx_path, sheet_name="marketing")

    for df in (customers, applications, stores, marketing):
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df.drop(columns=unnamed, inplace=True)

    # Dates
    applications["submit_date"] = pd.to_datetime(applications.get("submit_date"), errors="coerce")
    applications["approved_date"] = pd.to_datetime(applications.get("approved_date"), errors="coerce")
    marketing["start_date"] = pd.to_datetime(marketing.get("start_date"), errors="coerce")
    marketing["end_date"] = pd.to_datetime(marketing.get("end_date"), errors="coerce")

    # Core fields
    applications["is_approved"] = applications.get("approved", False).fillna(False).astype(bool)
    applications["approved_amount"] = pd.to_numeric(applications.get("approved_amount", 0), errors="coerce").fillna(0.0)
    applications["dollars_used"] = pd.to_numeric(applications.get("dollars_used", 0), errors="coerce").fillna(0.0)
    applications["is_used"] = applications["dollars_used"] > 0

    # Join to customers and stores
    apps = applications.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
    apps = apps.merge(stores, on="store", how="left", suffixes=("", "_store"))

    # Marketing join: customers.campaign (id) -> marketing.id
    if "campaign" in apps.columns and "id" in marketing.columns:
        marketing_ren = marketing.rename(
            columns={
                "name": "campaign_name",
                "spend": "campaign_spend",
                "start_date": "campaign_start_date",
                "end_date": "campaign_end_date",
            }
        )
        apps = apps.merge(
            marketing_ren[["id", "campaign_name", "campaign_spend", "campaign_start_date", "campaign_end_date"]],
            left_on="campaign",
            right_on="id",
            how="left",
        )
        apps.drop(columns=["id"], inplace=True, errors="ignore")

    return customers, applications, stores, marketing, apps


# -----------------------------
# Title
# -----------------------------
st.title("Snap Finance — Application Performance Dashboard")
st.caption("Applications → Approvals → Usage, plus amount trends and geographic distribution.")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Data")
    xlsx_path = st.text_input("Path to sample_datasets.xlsx", str(DEFAULT_XLSX))

    try:
        customers, applications, stores, marketing, apps = load_data(xlsx_path)
    except FileNotFoundError:
        st.error(
            "File not found.\n\n"
            "Fix: Put `sample_datasets.xlsx` in the same folder as `app.py`, "
            "or paste the full path here."
        )
        st.stop()

    st.header("Filters")

    min_dt = apps["submit_date"].min()
    max_dt = apps["submit_date"].max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.error("submit_date is missing/invalid in the dataset.")
        st.stop()

    date_range = st.date_input(
        "Submission date range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )

    granularity = st.selectbox(
        "Trend granularity",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=1,
    )

    def opts(col: str):
        return sorted([x for x in apps[col].dropna().unique()]) if col in apps.columns else []

    states = st.multiselect("State", opts("state"), default=[])
    industries = st.multiselect("Industry", opts("industry"), default=[])
    sizes = st.multiselect("Store size", opts("size"), default=[])
    stores_sel = st.multiselect("Store", opts("store"), default=[])
    campaigns = st.multiselect("Campaign", opts("campaign_name"), default=[])

    map_theme = st.selectbox("Map theme", ["Auto", "Dark"], index=1)


# -----------------------------
# Apply filters
# -----------------------------
start_date, end_date = date_range
mask = (apps["submit_date"].dt.date >= start_date) & (apps["submit_date"].dt.date <= end_date)

if states and "state" in apps.columns:
    mask &= apps["state"].isin(states)
if industries and "industry" in apps.columns:
    mask &= apps["industry"].isin(industries)
if sizes and "size" in apps.columns:
    mask &= apps["size"].isin(sizes)
if stores_sel and "store" in apps.columns:
    mask &= apps["store"].isin(stores_sel)
if campaigns and "campaign_name" in apps.columns:
    mask &= apps["campaign_name"].isin(campaigns)

f = apps.loc[mask].copy()

if f.empty:
    st.warning("No data matches your filters. Please widen the date range or clear filters.")
    st.stop()

f["state_abbr"] = normalize_state_to_abbr(f["state"]) if "state" in f.columns else np.nan
f["bucket"] = time_bucket(f["submit_date"], granularity)

# Weekly spike fix for bucketed charts
if granularity == "Weekly":
    f = drop_partial_first_week(f, date_col="submit_date", bucket_col="bucket")

if f.empty:
    st.warning("No data remains after weekly partial-week cleanup. Try a wider date range.")
    st.stop()


# -----------------------------
# KPI strip (NO green deltas)
# -----------------------------
total_apps = int(len(f))
approved_apps = int(f["is_approved"].sum())
used_apps = int(f["is_used"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Applications", f"{total_apps:,}")
k2.metric("Approved", f"{approved_apps:,}")
k3.metric("Used", f"{used_apps:,}")
k4.metric("Avg Approved Amount", fmt_money(f["approved_amount"].mean()))
k5.metric("Avg Used Amount", fmt_money(f["dollars_used"].mean()))

st.divider()


# =========================================================
# TASK 1 — Month-aligned YoY lines
# =========================================================
st.subheader("Task 1 — Trends (Applications, Approved, Used)")

t1_base = f.dropna(subset=["submit_date"]).copy()
t1_base["year"] = t1_base["submit_date"].dt.year.astype(str)
t1_base["month_num"] = t1_base["submit_date"].dt.month

years_present = sorted(t1_base["year"].unique())
years_to_show = years_present[-2:] if len(years_present) > 2 else years_present

preferred_order = [y for y in ["2022", "2023"] if y in years_to_show] + [
    y for y in years_to_show if y not in ["2022", "2023"]
]
t1_base = t1_base[t1_base["year"].isin(years_to_show)].copy()
t1_base["year"] = pd.Categorical(t1_base["year"], categories=preferred_order, ordered=True)

t1_apps = (
    t1_base.groupby(["year", "month_num"], observed=True, as_index=False)
    .agg(applications=("application_id", "count"))
)
t1_apps["month"] = t1_apps["month_num"].map(lambda m: calendar.month_abbr[int(m)])

t1_appr = (
    t1_base.groupby(["year", "month_num"], observed=True, as_index=False)
    .agg(approved=("is_approved", "sum"))
)
t1_appr["month"] = t1_appr["month_num"].map(lambda m: calendar.month_abbr[int(m)])

t1_used = (
    t1_base.groupby(["year", "month_num"], observed=True, as_index=False)
    .agg(used=("is_used", "sum"))
)
t1_used["month"] = t1_used["month_num"].map(lambda m: calendar.month_abbr[int(m)])

color_map = {y: YEAR_COLOR_MAP.get(str(y), None) for y in preferred_order}
color_map = {k: v for k, v in color_map.items() if v}

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### # of Application")
        fig = px.line(
            t1_apps.sort_values(["year", "month_num"]),
            x="month",
            y="applications",
            color="year",
            markers=True,
            category_orders={"month": MONTH_ORDER},
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_layout(height=260, hovermode="x unified", legend_title_text="Year",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    with st.container(border=True):
        st.markdown("### # of Approved")
        fig = px.line(
            t1_appr.sort_values(["year", "month_num"]),
            x="month",
            y="approved",
            color="year",
            markers=True,
            category_orders={"month": MONTH_ORDER},
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_layout(height=260, hovermode="x unified", legend_title_text="Year",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with c3:
    with st.container(border=True):
        st.markdown("### # of Used Application")
        fig = px.line(
            t1_used.sort_values(["year", "month_num"]),
            x="month",
            y="used",
            color="year",
            markers=True,
            category_orders={"month": MONTH_ORDER},
            color_discrete_map=color_map if color_map else None,
        )
        fig.update_layout(height=260, hovermode="x unified", legend_title_text="Year",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# =========================================================
# TASK 2 — Avg amount trends
# =========================================================
st.subheader("Task 2 — Avg Approved Amount & Avg Used Amount Trends")

t2 = (
    f.groupby("bucket", as_index=False)
    .agg(
        avg_approved_amount=("approved_amount", "mean"),
        avg_used_amount=("dollars_used", "mean"),
    )
    .sort_values("bucket")
)

sp_l, t2c1, t2c2, sp_r = st.columns([0.25, 1, 1, 0.25])

with t2c1:
    with st.container(border=True):
        st.markdown("### Avg Approved Amount")
        fig = px.line(t2, x="bucket", y="avg_approved_amount", markers=True)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with t2c2:
    with st.container(border=True):
        st.markdown("### Avg Used Amount")
        fig = px.line(t2, x="bucket", y="avg_used_amount", markers=True)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# =========================================================
# TASK 3 — Store metrics + bonus viz
# =========================================================
st.subheader("Task 3 — Metrics by Store (Funnel Rates)")

by_store = (
    f.groupby("store", as_index=False)
    .agg(
        applications=("application_id", "count"),
        approved=("is_approved", "sum"),
        used=("is_used", "sum"),
        total_approved_amount=("approved_amount", "sum"),
        total_used_amount=("dollars_used", "sum"),
        avg_approved_amount=("approved_amount", "mean"),
        avg_used_amount=("dollars_used", "mean"),
        state=("state", "first"),
        industry=("industry", "first"),
        size=("size", "first"),
    )
)

by_store["approval_rate"] = by_store.apply(lambda r: safe_pct(r["approved"], r["applications"]), axis=1)
by_store["completion_rate"] = by_store.apply(lambda r: safe_pct(r["used"], r["approved"]), axis=1)
by_store["conversion_rate"] = by_store.apply(lambda r: safe_pct(r["used"], r["applications"]), axis=1)

by_store = by_store[
    [
        "store", "state", "industry", "size",
        "applications", "approved", "used",
        "approval_rate", "completion_rate", "conversion_rate",
        "total_approved_amount", "total_used_amount",
        "avg_approved_amount", "avg_used_amount",
    ]
].sort_values("applications", ascending=False)

display_store = by_store.copy()
display_store["approval_rate"] = display_store["approval_rate"].map(lambda x: f"{x:.1%}")
display_store["completion_rate"] = display_store["completion_rate"].map(lambda x: f"{x:.1%}")
display_store["conversion_rate"] = display_store["conversion_rate"].map(lambda x: f"{x:.1%}")
display_store["total_approved_amount"] = display_store["total_approved_amount"].map(fmt_money)
display_store["total_used_amount"] = display_store["total_used_amount"].map(fmt_money)
display_store["avg_approved_amount"] = display_store["avg_approved_amount"].map(fmt_money)
display_store["avg_used_amount"] = display_store["avg_used_amount"].map(fmt_money)

st.dataframe(display_store, use_container_width=True, height=460)

st.subheader("Percentages trends over time (Bonus Viz)")

funnel_trends = (
    f.groupby("bucket", as_index=False)
    .agg(
        applications=("application_id", "count"),
        approved=("is_approved", "sum"),
        used=("is_used", "sum"),
    )
    .sort_values("bucket")
)

funnel_trends["approval_rate"] = safe_div_series(funnel_trends["approved"], funnel_trends["applications"])
funnel_trends["completion_rate"] = safe_div_series(funnel_trends["used"], funnel_trends["approved"])
funnel_trends["conversion_rate"] = safe_div_series(funnel_trends["used"], funnel_trends["applications"])

g1, g2, g3 = st.columns(3)

with g1:
    with st.container(border=True):
        st.markdown("### Approval Rate")
        fig = px.line(funnel_trends, x="bucket", y="approval_rate", markers=True)
        fig.update_layout(height=260, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with g2:
    with st.container(border=True):
        st.markdown("### Completion Rate")
        fig = px.line(funnel_trends, x="bucket", y="completion_rate", markers=True)
        fig.update_layout(height=260, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with g3:
    with st.container(border=True):
        st.markdown("### Conversion Rate")
        fig = px.line(funnel_trends, x="bucket", y="conversion_rate", markers=True)
        fig.update_layout(height=260, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# =========================================================
# TASK 4 — Marketing performance + Efficiency (ratio, NOT %)
# =========================================================
st.subheader("Task 4 — Used Dollars vs Marketing Spend (by Campaign)")

if "campaign_name" not in f.columns:
    st.info("Campaign data not available (marketing merge not present).")
else:
    mkt = (
        f.groupby("campaign_name", as_index=False)
        .agg(
            used_dollars=("dollars_used", "sum"),
            spend=("campaign_spend", "first"),
        )
    )
    mkt["campaign_name"] = mkt["campaign_name"].fillna("Unknown / Unmapped")
    mkt["spend"] = pd.to_numeric(mkt["spend"], errors="coerce").fillna(0.0)
    mkt["used_dollars"] = pd.to_numeric(mkt["used_dollars"], errors="coerce").fillna(0.0)

    # Add synthetic campaign: Radio Ads (spend=682,820, used=0)
    if "Radio Ads" not in set(mkt["campaign_name"].astype(str)):
        radio_ads = pd.DataFrame({"campaign_name": ["Radio Ads"], "used_dollars": [0.0], "spend": [682_820.0]})
        mkt = pd.concat([mkt, radio_ads], ignore_index=True)

    # Ensure Radio Ads appears at the END
    mkt["__radio_last"] = np.where(mkt["campaign_name"] == "Radio Ads", 1, 0)
    mkt = mkt.sort_values(["__radio_last", "spend"], ascending=[True, False]).drop(columns="__radio_last")

    # Chart 1: Spend vs Used
    mkt_long = mkt.melt(
        id_vars=["campaign_name"],
        value_vars=["used_dollars", "spend"],
        var_name="metric",
        value_name="amount",
    )

    fig4 = px.bar(
        mkt_long,
        x="campaign_name",
        y="amount",
        color="metric",
        barmode="group",
    )
    fig4.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig4.update_layout(xaxis_title="Campaign", yaxis_title="Amount", xaxis={"tickangle": -30})
    st.plotly_chart(fig4, use_container_width=True)

    # Chart 2: Efficiency ratio = Used / Spend
    mkt["efficiency"] = np.where(mkt["spend"] > 0, mkt["used_dollars"] / mkt["spend"], 0.0)

    with st.container(border=True):
        st.markdown("### Efficiency (Used Dollars / Spend Dollars)")
        fig_eff = px.bar(
            mkt,
            x="campaign_name",
            y="efficiency",
            text=mkt["efficiency"].map(lambda x: f"{x:.2f}"),
        )
        fig_eff.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Campaign",
            yaxis_title="Efficiency",
            xaxis={"tickangle": -30},
        )
        fig_eff.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig_eff, use_container_width=True)

st.divider()


# =========================================================
# TASK 5 — Insight (Lease grade trend + State map)
# =========================================================
st.subheader("Task 5 — Insight")

f["state_abbr"] = normalize_state_to_abbr(f["state"]) if "state" in f.columns else np.nan

statewise = (
    f.dropna(subset=["state_abbr"])
    .groupby("state_abbr", as_index=False)
    .agg(
        applications=("application_id", "count"),
        approved=("is_approved", "sum"),
        used=("is_used", "sum"),
        total_approved_amount=("approved_amount", "sum"),
        total_used_amount=("dollars_used", "sum"),
    )
)
statewise["approval_rate"] = safe_div_series(statewise["approved"], statewise["applications"])
statewise["utilization_rate"] = safe_div_series(statewise["used"], statewise["applications"])

left5, right5 = st.columns([1.15, 0.85])

with left5:
    with st.container(border=True):
        st.markdown("### Trends over time by Lease Grade (Stacked Applications)")

        if "lease_grade" not in f.columns:
            st.info("lease_grade column not found in the dataset.")
        else:
            lg_trend = (
                f.dropna(subset=["lease_grade", "bucket"])
                .groupby(["bucket", "lease_grade"], as_index=False)
                .agg(applications=("application_id", "count"))
                .sort_values("bucket")
            )

            fig_lg = px.bar(
                lg_trend,
                x="bucket",
                y="applications",
                color="lease_grade",
                barmode="stack",
            )
            fig_lg.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Submission Date",
                yaxis_title="Applications",
            )
            st.plotly_chart(fig_lg, use_container_width=True)

with right5:
    with st.container(border=True):
        st.markdown("### Statewise Distribution")

        map_mode = st.selectbox(
            "Map Metric",
            [
                "applications",
                "approved",
                "used",
                "approval_rate",
                "utilization_rate",
                "total_used_amount",
                "total_approved_amount",
            ],
            index=0,
            key="map_metric_task5",
        )

        fig_map = px.choropleth(
            statewise,
            locations="state_abbr",
            locationmode="USA-states",
            color=map_mode,
            scope="usa",
            color_continuous_scale="Blues",  # low=light, high=dark
            labels={map_mode: map_mode.replace("_", " ").title()},
        )

        if map_theme == "Dark":
            fig_map.update_layout(template="plotly_dark")

        fig_map.update_layout(height=520, margin=dict(l=5, r=5, t=25, b=5))
        st.plotly_chart(fig_map, use_container_width=True)
