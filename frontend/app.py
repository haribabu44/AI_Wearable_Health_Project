"""
AI Wearable Health Dashboard  —  Professional Streamlit Frontend v2
====================================================================
Improvements over v1:
  - API key authentication sent with every request
  - Proper error handling and user-facing error messages
  - Typed response parsing — no raw tuple indexing
  - Sidebar configuration panel (API URL, API key)
  - Health score colour coded with severity label
  - Model metrics tab showing training performance
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from fpdf import FPDF

# ── Config ────────────────────────────────────────────────────────────────────
#DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api")
DEFAULT_API_URL = os.getenv(
    "API_URL",
    "https://ai-wearable-health-project-2.onrender.com/api"
)
DEFAULT_API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
METRICS_PATH    = Path(__file__).parent.parent / "ml" / "saved_models" / "training_metrics.json"

st.set_page_config(
    page_title="AI Health Monitor",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#e6edf3!important}
.stApp{background-color:#0f172a}
[data-testid="stSidebar"]{background:#020617;border-right:1px solid #1e293b}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
input,textarea{background-color:#020617!important;color:#f8fafc!important;border:1px solid #334155!important;border-radius:8px!important}
.stNumberInput input,.stTextInput input{background-color:#020617!important;color:#f8fafc!important}
label{color:#cbd5f5!important;font-weight:500}
button[kind="primary"]{background:linear-gradient(135deg,#22c55e,#4ade80);color:black!important;border-radius:10px;font-weight:600}
button[kind="secondary"]{background:#1e293b;color:#e2e8f0!important;border-radius:10px}
.metric-card{background:#020617;border:1px solid #1e293b;padding:20px;border-radius:14px;text-align:center}
.metric-card h4{color:#94a3b8;font-size:.8rem;margin:0}
.metric-card h1{color:#f8fafc;font-size:2rem;margin:4px 0 0 0}
.alert-danger{background:#2d1515;border-left:4px solid #ef4444;color:#fecaca;padding:10px;border-radius:6px;margin:4px 0}
.alert-warn{background:#2d2a12;border-left:4px solid #facc15;color:#fde68a;padding:10px;border-radius:6px;margin:4px 0}
.alert-ok{background:#052e16;border-left:4px solid #22c55e;color:#bbf7d0;padding:10px;border-radius:6px;margin:4px 0}
.risk-high{background:#2d1515;border:1px solid #ef4444;border-radius:10px;padding:14px;margin:6px 0}
.risk-mod{background:#2d2a12;border:1px solid #facc15;border-radius:10px;padding:14px;margin:6px 0}
.risk-low{background:#052e16;border:1px solid #22c55e;border-radius:10px;padding:14px;margin:6px 0}
button[data-baseweb="tab"]{color:#cbd5f5!important;font-weight:600}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-thumb{background:#334155;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:10px 0 5px 0'>
  <h1 style='color:#64ffda;font-size:2.2rem;margin:0'>🩺 AI Health Monitor</h1>
  <p style='color:#8892b0;font-size:.95rem;margin-top:6px'>
    Smart Wearable — Real-time Disease Risk & Vitals Dashboard v2.0
  </p>
</div>
<hr style='border:none;border-top:1px solid #2e3650;margin:14px 0 20px 0'>
""", unsafe_allow_html=True)


# ── Sidebar — configuration + inputs ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

    st.markdown("---")
    st.markdown("### 👤 Patient Info")
    patient_name = st.text_input("Patient name", placeholder="e.g. Arjun Sharma")
    age  = st.number_input("Age (years)",     min_value=1,   max_value=120, value=35)
    sex  = st.selectbox("Sex", ["Female", "Male"])
    bmi  = st.number_input("BMI",             min_value=10.0, max_value=70.0, value=24.5, step=0.1)

    st.markdown("---")
    st.markdown("### 📡 Wearable Vitals")
    steps       = st.number_input("Steps today",          min_value=0,    max_value=50000, value=6000, step=100)
    temperature = st.number_input("Body temperature (°C)", min_value=34.0, max_value=43.0, value=36.7, step=0.1, format="%.1f")
    spo2        = st.number_input("SpO2 (%)",              min_value=50.0, max_value=100.0, value=98.0, step=0.5, format="%.1f")
    glucose     = st.number_input("Glucose (mg/dL)",       min_value=20.0, max_value=600.0, value=95.0, step=1.0)
    bp          = st.number_input("Systolic BP (mmHg)",    min_value=50.0, max_value=250.0, value=118.0, step=1.0)
    heart_rate  = st.number_input("Resting HR (bpm)",      min_value=20.0, max_value=250.0, value=72.0, step=1.0)

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Vitals", type="primary", use_container_width=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def call_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{api_url}/predict", json=payload,
                          headers=_headers(), timeout=15)
        if r.status_code == 401:
            st.error("❌ Invalid API key. Check the API Key field in the sidebar.")
            return None
        if r.status_code == 429:
            st.warning("⏳ Rate limit reached. Please wait a moment.")
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at {api_url}. Is it running?")
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
    return None


def call_history(limit: int = 100) -> list[dict]:
    try:
        r = requests.get(f"{api_url}/history", params={"limit": limit},
                         headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json().get("records", [])
    except Exception:
        return []


# ── Prediction flow ───────────────────────────────────────────────────────────
if predict_btn:
    payload = {
        "patient_name": patient_name or None,
        "steps":        float(steps),
        "temperature":  float(temperature),
        "spo2":         float(spo2),
        "glucose":      float(glucose),
        "bp":           float(bp),
        "heart_rate":   float(heart_rate),
        "bmi":          float(bmi),
        "age":          int(age),
        "sex":          1 if sex == "Male" else 0,
    }
    with st.spinner("Running inference…"):
        result = call_predict(payload)
    if result:
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_vitals, tab_disease, tab_history, tab_metrics = st.tabs(
    ["📊 Vitals", "🧬 Disease Risk", "📋 History", "🎯 Model Metrics"]
)


# ─── TAB 1: Vitals ────────────────────────────────────────────────────────────
with tab_vitals:
    if not result:
        st.info("Submit vitals using the sidebar to see results.")
    else:
        score = result["health_score"]
        hr    = result["predicted_heart_rate"]
        anom  = result["anomaly_status"]

        # Score colour
        if score >= 85:
            score_color = "#22c55e"
        elif score >= 60:
            score_color = "#facc15"
        else:
            score_color = "#ef4444"

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, unit in [
            (c1, "Health Score",    f"{score}/100", ""),
            (c2, "Predicted HR",    f"{hr:.0f}",    "bpm"),
            (c3, "Anomaly Status",  anom,            ""),
            (c4, "SpO2",            f"{spo2:.1f}",  "%"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <h4>{label}</h4>
                  <h1 style="color:{score_color if label=='Health Score' else '#f8fafc'}">{val}</h1>
                  <p style="color:#64748b;margin:0;font-size:.8rem">{unit}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Alerts
        st.markdown("#### 🔔 Clinical Alerts")
        for alert in result["alerts"]:
            if "🚨" in alert:
                css = "alert-danger"
            elif "⚠️" in alert or "🔥" in alert or "🧊" in alert:
                css = "alert-warn"
            else:
                css = "alert-ok"
            st.markdown(f'<div class="{css}">{alert}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Radar chart
        inputs = result.get("inputs", {})
        cats   = ["Steps", "Temperature", "SpO2", "Glucose", "BP", "Heart Rate"]
        vals_raw   = [inputs.get("steps",0), inputs.get("temperature",0),
                      inputs.get("spo2",0),  inputs.get("glucose",0),
                      inputs.get("bp",0),     inputs.get("heart_rate",0)]
        # Normalise 0-1 for radar
        norms  = [min(steps/10000,1), min((temperature-34)/9,1), min(spo2/100,1),
                  min(glucose/300,1), min(bp/200,1), min(heart_rate/200,1)]
        fig = go.Figure(go.Scatterpolar(
            r=norms + [norms[0]], theta=cats + [cats[0]],
            fill="toself", line_color="#64ffda",
            fillcolor="rgba(100,255,218,0.1)",
        ))
        fig.update_layout(
            polar=dict(bgcolor="#0f172a",
                       radialaxis=dict(visible=True, range=[0,1], color="#475569"),
                       angularaxis=dict(color="#94a3b8")),
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", margin=dict(t=20,b=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB 2: Disease Risk ──────────────────────────────────────────────────────
with tab_disease:
    if not result:
        st.info("Submit vitals using the sidebar to see disease risk results.")
    else:
        disease_preds = result.get("disease_predictions", {})
        if not disease_preds:
            st.warning("No disease models loaded. Run `python ml/train_all.py` first.")
        else:
            st.markdown("#### 🧬 Multi-Disease Risk Assessment")
            st.caption("Powered by GradientBoosting classifiers trained on 10,000 clinically-grounded samples")

            for disease, info in disease_preds.items():
                prob  = info["probability"]
                risk  = info["risk"]
                interp = info["interpretation"]
                pct   = int(prob * 100)
                css   = "risk-high" if pct >= 65 else ("risk-mod" if pct >= 40 else "risk-low")
                icon  = "🔴" if pct >= 65 else ("🟡" if pct >= 40 else "🟢")
                st.markdown(f"""
                <div class="{css}">
                  <b>{icon} {disease}</b>
                  <span style="float:right;font-weight:700">{pct}%</span><br>
                  <small style="color:#94a3b8">{interp}</small>
                  <div style="background:#1e293b;border-radius:6px;height:6px;margin-top:8px">
                    <div style="width:{pct}%;background:{'#ef4444' if pct>=65 else ('#facc15' if pct>=40 else '#22c55e')};height:6px;border-radius:6px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Probability bar chart
            names  = list(disease_preds.keys())
            probs  = [disease_preds[n]["probability"] * 100 for n in names]
            colors = ["#ef4444" if p >= 65 else ("#facc15" if p >= 40 else "#22c55e") for p in probs]
            fig2 = go.Figure(go.Bar(
                x=probs, y=names, orientation="h",
                marker_color=colors, text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
            ))
            fig2.add_vline(x=40, line_dash="dot", line_color="#facc15",
                           annotation_text="Moderate risk threshold")
            fig2.add_vline(x=65, line_dash="dot", line_color="#ef4444",
                           annotation_text="High risk threshold")
            fig2.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", xaxis=dict(range=[0,100], color="#475569"),
                yaxis=dict(color="#94a3b8"), height=300, margin=dict(t=10,b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # PDF report
            st.markdown("---")
            if st.button("📄 Download PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                font_path = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"
                if font_path.exists():
                    pdf.add_font("DejaVu", "", str(font_path))
                    pdf.set_font("DejaVu", size=12)
                else:
                    pdf.set_font("Helvetica", size=12)

                pdf.set_font(size=18)
                pdf.cell(0, 12, "AI Health Monitor — Clinical Report", ln=True, align="C")
                pdf.set_font(size=10)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
                if result.get("patient_name"):
                    pdf.cell(0, 8, f"Patient: {result['patient_name']}", ln=True, align="C")
                pdf.ln(6)

                pdf.set_font(size=13)
                pdf.cell(0, 10, "Vitals Summary", ln=True)
                pdf.set_font(size=10)
                for k, v in (result.get("inputs") or {}).items():
                    pdf.cell(0, 7, f"  {k.replace('_',' ').title()}: {v}", ln=True)

                pdf.ln(4)
                pdf.set_font(size=13)
                pdf.cell(0, 10, f"Health Score: {result['health_score']}/100", ln=True)
                pdf.cell(0, 10, f"Anomaly Status: {result['anomaly_status']}", ln=True)

                pdf.ln(4)
                pdf.set_font(size=13)
                pdf.cell(0, 10, "Disease Risk Probabilities", ln=True)
                pdf.set_font(size=10)
                for disease, info in disease_preds.items():
                    tag = "HIGH" if info["probability"] >= 0.65 else ("MOD" if info["probability"] >= 0.40 else "LOW")
                    pdf.cell(0, 7,
                             f"  {disease}: {info['probability']*100:.1f}%  [{tag}]  {info['interpretation']}",
                             ln=True)
                pdf.ln(4)
                pdf.set_font(size=9)
                pdf.cell(0, 6, "DISCLAIMER: This report is for informational purposes only. "
                               "Consult a qualified physician for medical advice.", ln=True)

                pdf_bytes = pdf.output()
                st.download_button(
                    label="💾 Save PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )


# ─── TAB 3: History ───────────────────────────────────────────────────────────
with tab_history:
    st.markdown("#### 📋 Recent Assessments")
    records = call_history(limit=100)
    if not records:
        st.info("No records yet — run a prediction first.")
    else:
        df_hist = pd.DataFrame(records)

        # Parse JSON columns
        if "disease_predictions" in df_hist.columns:
            df_hist["disease_predictions"] = df_hist["disease_predictions"].apply(
                lambda x: x if isinstance(x, str) else json.dumps(x)
            )
        if "alerts" in df_hist.columns:
            df_hist["alerts"] = df_hist["alerts"].apply(
                lambda x: x if isinstance(x, str) else json.dumps(x)
            )

        # Health score trend
        if "health_score" in df_hist.columns and "timestamp" in df_hist.columns:
            fig3 = px.line(
                df_hist.sort_values("timestamp"),
                x="timestamp", y="health_score",
                title="Health Score Over Time",
                color_discrete_sequence=["#64ffda"],
            )
            fig3.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", height=280,
                xaxis=dict(color="#475569"), yaxis=dict(color="#94a3b8", range=[0,105]),
            )
            st.plotly_chart(fig3, use_container_width=True)

        cols_to_show = [c for c in ["timestamp","patient_name","health_score",
                                     "anomaly_status","predicted_heart_rate",
                                     "steps","glucose","bp","spo2"] if c in df_hist.columns]
        st.dataframe(df_hist[cols_to_show], use_container_width=True, hide_index=True)

        csv_bytes = df_hist.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", data=csv_bytes,
                           file_name="health_history.csv", mime="text/csv")


# ─── TAB 4: Model Metrics ────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("#### 🎯 Model Training Performance")
    st.caption("Metrics generated by `ml/train_all.py` — refreshed on each training run")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        # Heart rate metrics
        if "heart_rate" in metrics:
            m = metrics["heart_rate"]
            st.markdown("**Heart Rate Regressor** (RandomForest + StandardScaler)")
            c1, c2, c3 = st.columns(3)
            c1.metric("CV MAE", f"{m['cv_mae']:.2f} bpm")
            c2.metric("Test MAE", f"{m['test_mae']:.2f} bpm")
            c3.metric("Test R²", f"{m['r2']:.4f}")

        st.markdown("---")

        # Disease metrics
        disease_keys = [k for k in metrics if k not in ("heart_rate", "anomaly")]
        if disease_keys:
            st.markdown("**Disease Classifiers** (GradientBoosting + StandardScaler)")
            rows = []
            for k in disease_keys:
                m = metrics[k]
                rows.append({
                    "Disease":    k.replace("_"," ").title(),
                    "CV AUC":     f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}",
                    "Test AUC":   f"{m['test_auc']:.3f}",
                    "Prevalence": f"{m['pos_rate']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("AUC > 0.85 = excellent discriminative power for a binary classifier")

        if "anomaly" in metrics:
            m = metrics["anomaly"]
            st.markdown(f"**Anomaly Detector** — trained on {m['trained_on_healthy_n']:,} "
                        f"healthy records (contamination={m['contamination']})")
    else:
        st.warning("No metrics file found. Run `python ml/train_all.py` to train all models.")
        st.code("cd ai-wearable-pro\npython ml/train_all.py", language="bash")
