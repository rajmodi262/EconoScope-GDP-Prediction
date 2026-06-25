# pharma_analytics.py
# ---------------------------------------------------------------------------
# EconoScope · US Biopharma Commercial Analytics module
#
# Built to mirror the day-to-day of a commercial-analytics consultant in life
# sciences: market sizing, brand-level forecasting, patent-cliff (LOE) erosion
# modelling, competitive landscaping, and an auto-generated consulting brief
# (Insights / Opportunities / Threats).
#
# The module is fully self-contained — it constructs a realistic, deterministic
# US biopharma reference market (seeded, reproducible) so the dashboard runs
# without external data or retraining. Swap `build_market()` for a live data
# feed (IQVIA / Evaluate / company 10-Ks) and the rest of the pipeline is
# unchanged.
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except Exception:  # graceful fallback if plotly missing
    PLOTLY = False

# Blue-Matter-leaning palette (deep blue / cyan / violet to match app shell)
COLORS = ["#00d4ff", "#7b2ff7", "#10b981", "#f59e0b", "#ef4444",
          "#06b6d4", "#a78bfa", "#34d399", "#fbbf24", "#fb7185"]

HIST_YEARS = list(range(2019, 2025))     # 2019-2024 actuals
FCST_YEARS = list(range(2025, 2031))     # 2025-2030 forecast


# ---------------------------------------------------------------------------
# 1. REFERENCE MARKET  (deterministic, US net sales in $M)
# ---------------------------------------------------------------------------
def build_market() -> pd.DataFrame:
    """Realistic US branded-Rx reference set. US net sales, $M."""
    rng = np.random.default_rng(42)
    # brand, company, therapy area, modality, 2024 US net sales ($M),
    # pre-LOE CAGR, loss-of-exclusivity year, biosimilar/generic
    rows = [
        ("Keytruda-class IO", "Merck-like",     "Oncology",        "Biologic", 14200, 0.14, 2028, "Biosimilar"),
        ("CDK4/6 inhibitor",  "Pfizer-like",     "Oncology",        "Small mol",  5200, 0.06, 2027, "Generic"),
        ("BTK inhibitor",     "AbbVie-like",     "Oncology",        "Small mol",  4100, 0.04, 2026, "Generic"),
        ("Anti-TNF flagship", "AbbVie-like",     "Immunology",      "Biologic",  11800, -0.09, 2023, "Biosimilar"),
        ("IL-23 inhibitor",   "JnJ-like",        "Immunology",      "Biologic",  6900, 0.18, 2029, "Biosimilar"),
        ("IL-4/13 inhibitor", "Regeneron-like",  "Immunology",      "Biologic",  8700, 0.22, 2030, "Biosimilar"),
        ("GLP-1 (T2D/obesity)","NovoLilly-like", "Cardiometabolic", "Biologic",  18500, 0.31, 2031, "Biosimilar"),
        ("SGLT2 inhibitor",   "AZ-like",         "Cardiometabolic", "Small mol",  4600, 0.09, 2025, "Generic"),
        ("Factor Xa anticoag","BMS-like",        "Cardiometabolic", "Small mol",  6300, 0.02, 2026, "Generic"),
        ("Anti-CGRP migraine","Lilly-like",      "Neuroscience",    "Biologic",  1900, 0.16, 2030, "Biosimilar"),
        ("MS oral therapy",   "Novartis-like",   "Neuroscience",    "Small mol",  2400, 0.03, 2027, "Generic"),
        ("Gene therapy (rare)","Vertex-like",    "Rare Disease",    "Cell/Gene", 1300, 0.27, 2035, "None"),
        ("CFTR modulator",    "Vertex-like",     "Rare Disease",    "Small mol",  9100, 0.12, 2037, "Generic"),
        ("mRNA vaccine fr.",  "Pfizer-like",     "Vaccines",        "mRNA",      3800, -0.18, 2030, "None"),
        ("RSV vaccine",       "GSK-like",        "Vaccines",        "Biologic",  2100, 0.20, 2034, "None"),
    ]
    df = pd.DataFrame(rows, columns=[
        "Brand", "Company", "TherapyArea", "Modality",
        "Sales2024", "CAGR", "LOEYear", "ErosionType"])

    # back-cast actuals 2019-2024 from CAGR with light realistic noise
    for y in HIST_YEARS:
        factor = (1 + df["CAGR"]) ** (y - 2024)
        noise = rng.normal(1.0, 0.03, len(df))
        df[f"S{y}"] = (df["Sales2024"] * factor * noise).round(0)
    return df


# ---------------------------------------------------------------------------
# 2. FORECAST ENGINE  (pre-LOE growth + post-LOE erosion curve)
# ---------------------------------------------------------------------------
def erosion_factor(years_since_loe: int, erosion_type: str) -> float:
    """US erosion curves: biologics erode slower than small-molecule generics."""
    if years_since_loe < 0:
        return 1.0
    if erosion_type == "Generic":          # steep small-molecule cliff
        curve = [1.0, 0.45, 0.25, 0.18, 0.14, 0.12]
    elif erosion_type == "Biosimilar":     # gentler biologic erosion
        curve = [1.0, 0.80, 0.65, 0.55, 0.48, 0.43]
    else:                                   # protected (gene tx / vaccine)
        curve = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    i = min(years_since_loe, len(curve) - 1)
    return curve[i]


def forecast_brand(row: pd.Series) -> dict:
    """Return {year: sales} forecast 2025-2030 for one brand."""
    base = row["Sales2024"]
    out = {}
    for y in FCST_YEARS:
        pre = base * (1 + row["CAGR"]) ** (y - 2024)
        ef = erosion_factor(y - row["LOEYear"], row["ErosionType"])
        out[y] = max(pre * ef, 0)
    return out


def build_forecast(df: pd.DataFrame) -> pd.DataFrame:
    fc = df.apply(forecast_brand, axis=1, result_type="expand")
    fc.columns = [f"F{y}" for y in FCST_YEARS]
    return pd.concat([df, fc], axis=1)


# ---------------------------------------------------------------------------
# 3. AUTO-GENERATED CONSULTING BRIEF  (Insights / Opportunities / Threats)
# ---------------------------------------------------------------------------
def generate_brief(df: pd.DataFrame) -> dict:
    insights, opps, threats = [], [], []

    if f"F{FCST_YEARS[-1]}" not in df.columns:   # ensure forecast present
        df = build_forecast(df)

    total_24 = df["Sales2024"].sum()
    total_30 = df[[f"F{y}" for y in FCST_YEARS]].iloc[:, -1].sum()
    cagr = (total_30 / total_24) ** (1 / 6) - 1
    insights.append(
        f"Tracked US market is **${total_24/1000:.1f}B** (2024) → **${total_30/1000:.1f}B** "
        f"(2030E), a **{cagr*100:+.1f}% CAGR** across {df['Brand'].nunique()} assets.")

    ta = (df.groupby("TherapyArea")["Sales2024"].sum() / total_24 * 100).sort_values(ascending=False)
    insights.append(
        f"**{ta.index[0]}** leads at **{ta.iloc[0]:.0f}%** of tracked value; "
        f"top-3 areas concentrate **{ta.iloc[:3].sum():.0f}%** — a focused commercial footprint.")

    # patent cliff exposure 2025-2030
    at_risk = df[(df["LOEYear"] >= 2025) & (df["LOEYear"] <= 2030)]
    risk_val = at_risk["Sales2024"].sum()
    threats.append(
        f"**${risk_val/1000:.1f}B** of 2024 sales ({risk_val/total_24*100:.0f}% of portfolio) face "
        f"loss-of-exclusivity by 2030 across **{len(at_risk)}** brands — a material patent cliff.")
    if not at_risk.empty:
        worst = at_risk.sort_values("Sales2024", ascending=False).iloc[0]
        threats.append(
            f"Largest single exposure: **{worst['Brand']}** ({worst['Company']}, "
            f"${worst['Sales2024']/1000:.1f}B) loses exclusivity in **{int(worst['LOEYear'])}** "
            f"to {worst['ErosionType'].lower()} competition.")

    # growth opportunities
    growth = df.sort_values("CAGR", ascending=False)
    g0 = growth.iloc[0]
    opps.append(
        f"**{g0['TherapyArea']}** momentum: **{g0['Brand']}** compounding at "
        f"**{g0['CAGR']*100:+.0f}%/yr** — prioritise share-of-voice and access investment.")
    protected = df[df["ErosionType"] == "None"]
    if not protected.empty:
        opps.append(
            f"**{len(protected)}** assets (gene therapy / vaccines) carry **no near-term biosimilar "
            f"threat** — durable, defensible revenue to anchor the portfolio.")
    declining = df[df["CAGR"] < 0]
    if not declining.empty:
        threats.append(
            f"**{len(declining)}** assets already in decline (e.g. **{declining.iloc[0]['Brand']}**); "
            f"lifecycle-management or managed divestment should be evaluated.")

    return {"insights": insights, "opportunities": opps, "threats": threats}


# ---------------------------------------------------------------------------
# 4. RENDER
# ---------------------------------------------------------------------------
def _metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="color:#94a3b8;font-size:12px;margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)


def render_pharma_tab():
    df = build_market()
    fdf = build_forecast(df)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💊 US Biopharma Commercial Analytics</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Market sizing · brand forecasting · patent-cliff erosion · '
        'competitive landscape · auto-generated consulting brief — modelled on US net Rx sales ($M).</div>',
        unsafe_allow_html=True)

    # ---- controls
    areas = ["All therapy areas"] + sorted(df["TherapyArea"].unique())
    c1, c2 = st.columns([2, 2])
    with c1:
        sel = st.selectbox("🔬 Therapy area lens", areas)
    with c2:
        view = st.radio("View", ["Market & Forecast", "Patent Cliff", "Competitive Landscape"],
                        horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sub = df if sel == "All therapy areas" else df[df["TherapyArea"] == sel]
    fsub = fdf if sel == "All therapy areas" else fdf[fdf["TherapyArea"] == sel]

    # ---- KPI row
    total_24 = sub["Sales2024"].sum()
    total_30 = fsub[f"F{FCST_YEARS[-1]}"].sum()
    cagr = (total_30 / total_24) ** (1 / 6) - 1 if total_24 else 0
    risk = sub[(sub["LOEYear"] >= 2025) & (sub["LOEYear"] <= 2030)]["Sales2024"].sum()
    k1, k2, k3, k4 = st.columns(4)
    with k1: _metric_card("2024 Market (US)", f"${total_24/1000:.1f}B", "tracked net sales")
    with k2: _metric_card("2030E Market", f"${total_30/1000:.1f}B", "model forecast")
    with k3: _metric_card("Forecast CAGR", f"{cagr*100:+.1f}%", "2024 → 2030")
    with k4: _metric_card("Sales at LOE risk", f"${risk/1000:.1f}B", "exclusivity loss by 2030")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- VIEW 1: market & forecast
    if view == "Market & Forecast":
        hist = [sub[f"S{y}"].sum() for y in HIST_YEARS]
        fc = [fsub[f"F{y}"].sum() for y in FCST_YEARS]
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=HIST_YEARS, y=[v/1000 for v in hist],
                          mode="lines+markers", name="Actual",
                          line=dict(color="#00d4ff", width=3)))
            allx = [HIST_YEARS[-1]] + FCST_YEARS
            ally = [hist[-1]] + fc
            fig.add_trace(go.Scatter(x=allx, y=[v/1000 for v in ally],
                          mode="lines+markers", name="Forecast",
                          line=dict(color="#7b2ff7", width=3, dash="dash")))
            # uncertainty band ±12%
            fig.add_trace(go.Scatter(
                x=allx + allx[::-1],
                y=[v*1.12/1000 for v in ally] + [v*0.88/1000 for v in ally][::-1],
                fill="toself", fillcolor="rgba(123,47,247,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="±12% band", hoverinfo="skip"))
            fig.update_layout(
                template="plotly_dark", height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                title="Market trajectory — US net sales ($B)",
                yaxis_title="$B", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

            # therapy-area treemap of 2024 value
            if sel == "All therapy areas":
                tdf = df.copy()
                fig2 = px.treemap(tdf, path=["TherapyArea", "Brand"], values="Sales2024",
                                  color="CAGR", color_continuous_scale="RdYlGn",
                                  title="2024 market value by therapy area & brand (color = growth)")
                fig2.update_layout(template="plotly_dark", height=420,
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"Actual ($B)": [v/1000 for v in hist]}, index=HIST_YEARS))

    # ---- VIEW 2: patent cliff
    elif view == "Patent Cliff":
        st.markdown('<div class="section-title">⏳ Patent-Cliff / Loss-of-Exclusivity Exposure</div>',
                    unsafe_allow_html=True)
        risk_by_year = (sub[sub["LOEYear"].between(2025, 2030)]
                        .groupby("LOEYear")["Sales2024"].sum().reindex(FCST_YEARS, fill_value=0))
        if PLOTLY:
            fig = go.Figure(go.Bar(
                x=[str(y) for y in FCST_YEARS], y=[v/1000 for v in risk_by_year.values],
                marker_color=["#ef4444" if v > 0 else "#334155" for v in risk_by_year.values],
                text=[f"${v/1000:.1f}B" if v else "" for v in risk_by_year.values],
                textposition="outside"))
            fig.update_layout(template="plotly_dark", height=380,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=40, b=10),
                              title="2024 sales reaching LOE, by year ($B at risk)",
                              yaxis_title="$B")
            st.plotly_chart(fig, use_container_width=True)
        cliff = sub[sub["LOEYear"].between(2025, 2030)][
            ["Brand", "Company", "TherapyArea", "Sales2024", "LOEYear", "ErosionType"]
        ].sort_values("LOEYear")
        cliff = cliff.rename(columns={"Sales2024": "US Sales 2024 ($M)", "LOEYear": "LOE Year"})
        st.dataframe(cliff, use_container_width=True, hide_index=True)

    # ---- VIEW 3: competitive landscape
    else:
        st.markdown('<div class="section-title">🏁 Competitive Landscape</div>',
                    unsafe_allow_html=True)
        if PLOTLY:
            plot = sub.copy()
            plot["GrowthPct"] = plot["CAGR"] * 100
            fig = px.scatter(
                plot, x="GrowthPct", y="Sales2024", size="Sales2024", color="TherapyArea",
                hover_name="Brand", size_max=60,
                labels={"GrowthPct": "Pre-LOE growth (%/yr)", "Sales2024": "US sales 2024 ($M)"},
                color_discrete_sequence=COLORS,
                title="Share-of-market vs growth (bubble = US sales)")
            fig.add_vline(x=0, line_dash="dot", line_color="#64748b")
            fig.update_layout(template="plotly_dark", height=460,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        comp = (sub.groupby("Company")
                .agg(Brands=("Brand", "count"), Sales=("Sales2024", "sum"),
                     AvgGrowth=("CAGR", "mean")).reset_index()
                .sort_values("Sales", ascending=False))
        comp["Share %"] = (comp["Sales"] / df["Sales2024"].sum() * 100).round(1)
        comp["Sales"] = comp["Sales"].map(lambda v: f"${v/1000:.1f}B")
        comp["AvgGrowth"] = (comp["AvgGrowth"] * 100).round(1).map(lambda v: f"{v:+.1f}%")
        st.dataframe(comp.rename(columns={"AvgGrowth": "Avg Growth"}),
                     use_container_width=True, hide_index=True)

    # ---- CONSULTING BRIEF (always shown)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Auto-Generated Consulting Brief</div>',
                unsafe_allow_html=True)
    brief = generate_brief(df)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("#### 🔎 Key Insights")
        for x in brief["insights"]:
            st.markdown(f"- {x}")
    with b2:
        st.markdown("#### 🚀 Opportunities")
        for x in brief["opportunities"]:
            st.markdown(f"- {x}")
    with b3:
        st.markdown("#### ⚠️ Threats")
        for x in brief["threats"]:
            st.markdown(f"- {x}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("Reference market is a deterministic, seeded approximation for demonstration. "
               "Replace `build_market()` with IQVIA / Evaluate / 10-K feeds for production use.")
