"""
🩺 AI Diabetes Prediction System — Final Year Project
Polished multi-page Streamlit app with auth, history, AI explanations,
personalised diet plans, admin analytics, and PDF reports.
"""
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import database as db
import ai_helper
from pdf_report import build_pdf

# ────────────────────────── Setup ──────────────────────────
load_dotenv()
ROOT = Path(__file__).parent
st.set_page_config(
    page_title="GlucoCare Health Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────── Theme / CSS ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] {
  visibility:hidden !important;
  height:0 !important;
}

:root {
  --brand:#0d47a1; --brand2:#1976d2; --brand3:#42a5f5;
  --ok:#2e7d32; --warn:#f9a825; --bad:#c62828;
  --glass: rgba(255,255,255,0.7);
}

.main .block-container { padding-top: 1.2rem; max-width: 1280px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0d47a1 0%,#1565c0 60%, #1976d2 100%);
}
section[data-testid="stSidebar"] * { color: #fff !important; }
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.10); color:#fff !important;
  border:1px solid rgba(255,255,255,.20); width:100%; text-align:left;
  border-radius:12px; padding:.65rem 1rem; margin:.2rem 0; font-weight:500;
  transition: all .25s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.25); border-color:#fff;
  transform: translateX(4px);
}

/* ── Animations ── */
@keyframes fadeUp { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
@keyframes pulse  { 0%,100% {transform:scale(1);} 50% {transform:scale(1.03);} }
@keyframes shimmer{ 0% {background-position:-1000px 0;} 100%{background-position:1000px 0;} }
.fade-up   { animation: fadeUp .55s ease-out; }
.pulse-anim{ animation: pulse 2.5s ease-in-out infinite; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg,#0d47a1 0%,#1976d2 60%,#42a5f5 100%);
  color:#fff; padding:2.6rem 2.2rem; border-radius:20px; margin-bottom:1.5rem;
  box-shadow:0 12px 40px rgba(13,71,161,.35);
  animation: fadeUp .55s ease-out;
  position: relative; overflow: hidden;
}
.hero::before {
  content:""; position:absolute; top:-50%; right:-10%; width:400px; height:400px;
  background: radial-gradient(circle, rgba(255,255,255,.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero h1 { color:#fff; margin:0; font-size:2.5rem; font-weight:800; letter-spacing:-.02em; }
.hero p  { color:rgba(255,255,255,.95); font-size:1.12rem; margin-top:.6rem; }

/* ── Auth screen ── */
.auth-topbar {
  display:flex; align-items:center; gap:.75rem; margin:.35rem 0 1.2rem;
}
.auth-mark {
  width:46px; height:46px; border-radius:14px;
  display:flex; align-items:center; justify-content:center;
  background:#e8f5f3; color:#00695c; font-size:1.45rem;
  border:1px solid #c8e6e0;
}
.auth-name { font-size:1.05rem; font-weight:800; color:#f8fbff; }
.auth-name span { display:block; font-size:.78rem; font-weight:600; color:rgba(248,251,255,.74); margin-top:.08rem; }
.auth-brand {
  min-height:560px; border-radius:22px; padding:2.2rem;
  background: linear-gradient(135deg,#f3fbf8 0%,#eaf4ff 58%,#fff8eb 100%);
  border:1px solid rgba(25,118,210,.16);
  box-shadow:0 18px 50px rgba(15,35,60,.10);
  color:#172033; overflow:hidden;
}
.auth-eyebrow {
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.38rem .75rem; border-radius:999px;
  background:rgba(0,105,92,.10); color:#00695c;
  font-size:.8rem; font-weight:800; letter-spacing:.04em;
  text-transform:uppercase;
}
.auth-brand h1 {
  color:#10233f; font-size:clamp(2rem,3.6vw,3rem);
  line-height:1.04; margin:1.2rem 0 .9rem; font-weight:800;
}
.auth-brand p {
  color:#46566c; font-size:1.05rem; line-height:1.7;
  max-width:620px; margin:0 0 1.5rem;
}
.auth-point-grid {
  display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
  gap:.8rem; margin:1.45rem 0;
}
.auth-point {
  background:rgba(255,255,255,.72); border:1px solid rgba(25,118,210,.14);
  border-radius:14px; padding:.9rem; min-height:96px;
}
.auth-point strong { display:block; color:#10233f; font-size:.95rem; margin-bottom:.35rem; }
.auth-point span { color:#607086; font-size:.86rem; line-height:1.45; }
.auth-note {
  border-left:4px solid #00897b; background:rgba(255,255,255,.78);
  padding:1rem 1.1rem; border-radius:12px; color:#526174;
  font-size:.92rem; line-height:1.55;
}
.auth-card-head { margin-bottom:1rem; }
.auth-card-head h2 { color:#f8fbff; margin:.1rem 0 .25rem; font-size:1.75rem; font-weight:800; }
.auth-card-head p { color:rgba(248,251,255,.72); margin:0; font-size:.95rem; }
.auth-secure {
  background:#f5faf9; border:1px solid #d7ebe6; border-radius:12px;
  color:#315c54; padding:.85rem 1rem; font-size:.88rem; line-height:1.45;
  margin-top:1rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:rgba(255,255,255,.96) !important;
  border:1px solid rgba(25,118,210,.16) !important;
  border-radius:22px !important;
  box-shadow:0 18px 50px rgba(15,35,60,.12) !important;
}
div[data-testid="stForm"] {
  border:0; padding:0; background:transparent;
}
div[data-testid="stTextInput"] input {
  border-radius:12px; min-height:44px;
  background:#fff !important; color:#172033 !important;
  border:1px solid #d8e2ee !important;
}
div[data-testid="stTextInput"] input::placeholder { color:#8a97a8 !important; }
div[data-testid="stTextInput"] label p { color:#f8fbff !important; font-weight:700; }
div[data-testid="stTabs"] [role="tab"] p { color:#526174 !important; font-weight:800; }
div[data-testid="stTabs"] [aria-selected="true"] p { color:#0d47a1 !important; }
.stFormSubmitButton > button {
  background: linear-gradient(135deg,#0d47a1 0%,#1976d2 100%) !important;
  color:#fff !important; border:none !important; border-radius:12px !important;
  min-height:46px; box-shadow:0 8px 18px rgba(13,71,161,.26) !important;
  font-weight:800 !important;
}
.stFormSubmitButton > button:hover {
  transform: translateY(-1px);
  box-shadow:0 12px 24px rgba(13,71,161,.32) !important;
}
div[data-testid="stTabs"] button {
  font-weight:700;
}
@media (max-width: 900px) {
  .auth-brand { min-height:auto; padding:1.6rem; }
  .auth-point-grid { grid-template-columns:1fr; }
}

/* ── Stat cards ── */
.stat-card {
  background: linear-gradient(135deg,#0d47a1 0%,#1976d2 100%);
  color:#fff; padding:1.4rem 1.25rem; border-radius:16px;
  box-shadow:0 6px 22px rgba(13,71,161,.25);
  transition: transform .25s ease, box-shadow .25s ease;
  animation: fadeUp .55s ease-out;
}
.stat-card:hover { transform: translateY(-4px); box-shadow:0 12px 32px rgba(13,71,161,.4); }
.stat-card h3 { color:#fff; margin:0; font-size:.8rem;
  text-transform:uppercase; letter-spacing:.08em; opacity:.85; font-weight:600; }
.stat-card .v { font-size:2.1rem; font-weight:800; margin-top:.4rem; line-height:1.1; }
.stat-card .sub { font-size:.78rem; opacity:.8; margin-top:.25rem; }

/* Glass card */
.glass {
  background: var(--glass);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,.5);
  border-radius:16px; padding:1.4rem;
  box-shadow:0 4px 18px rgba(0,0,0,.06);
  animation: fadeUp .55s ease-out;
  margin-bottom: 1rem;
}

/* Result badges */
.result-banner {
  padding:1.5rem; border-radius:16px; text-align:center;
  font-size:1.4rem; font-weight:700; animation: fadeUp .55s ease-out;
  box-shadow:0 6px 20px rgba(0,0,0,.08);
}
.result-pos { background: linear-gradient(135deg,#ffebee 0%, #ffcdd2 100%); color:#c62828; }
.result-neg { background: linear-gradient(135deg,#e8f5e9 0%, #c8e6c9 100%); color:#2e7d32; }

.footer { text-align:center; color:#777; padding:2rem 0 1rem;
  border-top:1px solid #eee; margin-top:3rem; font-size:.85rem; }

/* Streamlit native widgets */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg,#0d47a1 0%,#1976d2 100%);
  border:none; box-shadow:0 4px 14px rgba(13,71,161,.3);
  transition: all .25s ease; font-weight:600;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px); box-shadow:0 8px 22px rgba(13,71,161,.45);
}
div[data-testid="stMetric"] {
  background: white; padding: 1rem; border-radius: 12px;
  border:1px solid #eee; box-shadow:0 2px 8px rgba(0,0,0,.04);
}
div[data-testid="stMetric"] * { color:#172033 !important; }
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color:#526174 !important;
  white-space: normal !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color:#172033 !important;
  font-size: clamp(1.35rem, 2.2vw, 1.9rem) !important;
  line-height: 1.15 !important;
}

/* Admin badge */
.admin-pill {
  display:inline-block; background: #ffd54f; color:#5d4037;
  padding:.18rem .6rem; border-radius:999px; font-size:.7rem;
  font-weight:700; letter-spacing:.05em; margin-left:.4rem;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────── Cached resources ────────────────
@st.cache_resource
def load_model():
    with open(ROOT / "trained_model.sav", "rb") as f: return pickle.load(f)

@st.cache_resource
def load_metadata():
    with open(ROOT / "model_metadata.json") as f: return json.load(f)

@st.cache_data
def load_dataset():
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    return pd.read_csv(ROOT / "diabetes.csv", header=None, names=cols)

MODEL = load_model()
META = load_metadata()
FEATURES = META["feature_columns"]

# ────────────────────────── Session ─────────────────────────
for k, v in [("user", None), ("page", "Home"),
             ("last_prediction", None), ("ai_explanation", None),
             ("diet_plan", None)]:
    if k not in st.session_state: st.session_state[k] = v

# ────────────────────────── Auth ────────────────────────────
def auth_screen():
    st.markdown("""
    <div class="auth-topbar">
      <div class="auth-mark">🩺</div>
      <div class="auth-name">GlucoCare Health Portal<span>Secure health awareness portal</span></div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown("""
        <section class="auth-brand">
          <div class="auth-eyebrow">Health Awareness</div>
          <h1>Check diabetes risk and keep your records organized.</h1>
          <p>
            Sign in to save assessments, follow changes over time, and download
            clear reports for your personal health records.
          </p>
          <div class="auth-point-grid">
            <div class="auth-point">
              <strong>Risk Assessment</strong>
              <span>Enter basic health measurements and receive an easy-to-read risk result.</span>
            </div>
            <div class="auth-point">
              <strong>Personal History</strong>
              <span>Review your previous assessments and monitor changes over time.</span>
            </div>
            <div class="auth-point">
              <strong>Download Reports</strong>
              <span>Keep a simple report for your records or future doctor visits.</span>
            </div>
            <div class="auth-point">
              <strong>Health Guidance</strong>
              <span>Read general diabetes awareness tips for prevention and lifestyle care.</span>
            </div>
          </div>
          <div class="auth-note">
            This website is for awareness only. It does not replace medical
            testing, diagnosis, or advice from a qualified healthcare professional.
          </div>
        </section>
        """, unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("""
            <div class="auth-card-head">
              <h2>Welcome</h2>
              <p>Sign in to continue, or create a new account.</p>
            </div>
            """, unsafe_allow_html=True)
            tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign up"])
            with tab_login:
                with st.form("login_form"):
                    u = st.text_input("Username", placeholder="Enter your username")
                    p = st.text_input("Password", type="password", placeholder="Enter your password")
                    if st.form_submit_button("Login", use_container_width=True, type="primary"):
                        user = db.login(u.strip(), p)
                        if user:
                            st.session_state.user = user
                            st.session_state.page = "Home"
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
            with tab_signup:
                with st.form("signup_form"):
                    full = st.text_input("Full name", placeholder="Enter your full name")
                    email = st.text_input("Email", placeholder="Enter your email address")
                    u = st.text_input("Choose a username", placeholder="Create a username")
                    p = st.text_input("Choose a password", type="password",
                                      placeholder="Create a password",
                                      help="Minimum 6 characters")
                    st.caption("The first registered account is given admin access.")
                    if st.form_submit_button("Create account", use_container_width=True, type="primary"):
                        ok, msg = db.signup(u.strip(), p, full.strip(), email.strip())
                        (st.success if ok else st.error)(msg)
            st.markdown("""
            <div class="auth-secure">
              Your account helps keep your assessment history separate and private.
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────── Sidebar ─────────────────────────
def sidebar():
    with st.sidebar:
        u = st.session_state.user
        st.markdown(f"### 👤 {u['full_name'] or u['username']}")
        st.caption(f"@{u['username']}")
        st.markdown("---")

        nav = [("Home","🏠"), ("Predict","🔍"), ("History","📜"),
               ("Dashboard","📊"), ("Health Assistant","🤖"),
               ("Data Explorer","🔬"), ("About","ℹ️")]
        if u["is_admin"]:
            nav.insert(4, ("Admin Panel", "🛡️"))

        for p, icon in nav:
            if st.button(f"{icon}  {p}", key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()

        st.markdown("---")
        if st.button("🚪  Logout"):
            for k in ["user","last_prediction","ai_explanation","diet_plan"]:
                st.session_state[k] = None
            st.session_state.page = "Home"
            st.rerun()
        st.markdown(
            f"<small>Model: <b>{META['best_model']}</b><br>"
            f"Accuracy: <b>{META['metrics_best']['accuracy']*100:.2f}%</b><br>"
            f"AUC: <b>{META['metrics_best']['roc_auc']:.3f}</b></small>",
            unsafe_allow_html=True)

# ────────────────────────── Pages ───────────────────────────
def page_home():
    u = st.session_state.user
    st.markdown(f"""<div class='hero'>
        <h1>Welcome, {u['full_name'] or u['username']} 👋</h1>
        <p>Your personal glucose health dashboard.</p>
    </div>""", unsafe_allow_html=True)

    m = META["metrics_best"]
    cards = [("Accuracy", f"{m['accuracy']*100:.1f}%", "Test set"),
             ("Precision", f"{m['precision']*100:.1f}%", "Positive predictive"),
             ("Recall", f"{m['recall']*100:.1f}%", "Sensitivity"),
             ("ROC-AUC", f"{m['roc_auc']:.3f}", "Discrimination")]
    cols = st.columns(4)
    for col, (l, v, s) in zip(cols, cards):
        col.markdown(f"<div class='stat-card'><h3>{l}</h3>"
                     f"<div class='v'>{v}</div><div class='sub'>{s}</div></div>",
                     unsafe_allow_html=True)

    st.markdown(" ")
    preds = db.get_user_predictions(u["id"])
    cA, cB = st.columns([2, 1])
    with cA:
        st.subheader("📈 Your risk trend")
        if preds:
            df = pd.DataFrame(preds).sort_values("created_at")
            df["created_at"] = pd.to_datetime(df["created_at"])
            fig = px.line(df, x="created_at", y="risk_score", markers=True,
                          labels={"risk_score":"Risk %","created_at":"Date"})
            fig.update_traces(line_color="#0d47a1", line_width=3,
                              marker=dict(size=11, color="#1976d2",
                                          line=dict(color="white", width=2)))
            fig.add_hline(y=30, line_dash="dot", line_color="#2e7d32",
                          annotation_text="Low")
            fig.add_hline(y=60, line_dash="dot", line_color="#c62828",
                          annotation_text="High")
            fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10),
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet — head to **Predict** to make your first one.")
    with cB:
        st.subheader("⚡ Quick actions")
        if st.button("🔍 New Prediction", use_container_width=True, type="primary"):
            st.session_state.page = "Predict"; st.rerun()
        if st.button("📜 View History", use_container_width=True):
            st.session_state.page = "History"; st.rerun()
        if st.button("🤖 Ask Health Assistant", use_container_width=True):
            st.session_state.page = "Health Assistant"; st.rerun()
        if u["is_admin"]:
            if st.button("🛡️ Admin Panel", use_container_width=True):
                st.session_state.page = "Admin Panel"; st.rerun()


def page_predict():
    st.title("🔍 Diabetes Risk Prediction")
    st.caption("Enter the patient's clinical measurements below.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 1)
            bp   = st.number_input("Blood Pressure (mm Hg)", 0, 200, 72)
            ins  = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
            dpf  = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        with c2:
            glu  = st.number_input("Glucose (mg/dL)", 0, 250, 120)
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            bmi  = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
            age  = st.number_input("Age (years)", 1, 120, 30)
        submitted = st.form_submit_button("🔍 Predict", type="primary",
                                          use_container_width=True)

    if submitted:
        inputs = {"Pregnancies":preg,"Glucose":glu,"BloodPressure":bp,
                  "SkinThickness":skin,"Insulin":ins,"BMI":bmi,
                  "DiabetesPedigreeFunction":dpf,"Age":age}
        x = pd.DataFrame([[inputs[c] for c in FEATURES]], columns=FEATURES)
        pred = int(MODEL.predict(x)[0])
        risk = float(MODEL.predict_proba(x)[0][1] * 100)
        db.save_prediction(st.session_state.user["id"], inputs, pred, risk)
        st.session_state.last_prediction = {"inputs":inputs,"pred":pred,"risk":risk}
        st.session_state.ai_explanation = None
        st.session_state.diet_plan = None

    lp = st.session_state.last_prediction
    if not lp: return
    inputs, pred, risk = lp["inputs"], lp["pred"], lp["risk"]

    # Banner
    if pred == 1:
        st.markdown(f"<div class='result-banner result-pos pulse-anim'>"
                    f"⚠️ DIABETIC — Risk Probability {risk:.2f}%</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-banner result-neg'>"
                    f"✅ NON-DIABETIC — Risk Probability {risk:.2f}%</div>",
                    unsafe_allow_html=True)
    st.markdown("")

    # Gauge + contributions
    c1, c2 = st.columns([1, 1])
    with c1:
        band_color = "#2e7d32" if risk<30 else ("#f9a825" if risk<60 else "#c62828")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=risk,
            number={"suffix":"%","font":{"size":46,"color":band_color}},
            delta={"reference":50,"increasing":{"color":"#c62828"},
                   "decreasing":{"color":"#2e7d32"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#666"},
                "bar":{"color":band_color,"thickness":0.32},
                "bgcolor":"white","borderwidth":2,"bordercolor":"#eee",
                "steps":[
                    {"range":[0,30], "color":"#e8f5e9"},
                    {"range":[30,60],"color":"#fff8e1"},
                    {"range":[60,100],"color":"#ffebee"}],
                "threshold":{"line":{"color":"red","width":3},
                             "thickness":0.85,"value":risk},
            },
            title={"text":"<b>Diabetes Risk Score</b>", "font":{"size":18}},
        ))
        gauge.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
    with c2:
        st.markdown("##### 🧠 Feature contributions")
        imp = META["feature_importance"]
        means = META["dataset_stats"]["feature_means"]
        stds  = META["dataset_stats"]["feature_stds"]
        contribs = []
        for f in FEATURES:
            z = (inputs[f] - means[f]) / (stds[f] or 1)
            contribs.append({"Feature":f,"Importance":imp[f],
                             "z":z,"Contribution":imp[f]*z})
        cdf = pd.DataFrame(contribs).sort_values("Contribution", key=abs, ascending=False)
        cdf["Effect"] = cdf["Contribution"].apply(lambda v: "↑ Raises risk" if v>0 else "↓ Lowers risk")
        fig = px.bar(cdf, x="Contribution", y="Feature", color="Effect",
                     orientation="h",
                     color_discrete_map={"↑ Raises risk":"#c62828","↓ Lowers risk":"#2e7d32"})
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
                          showlegend=True, legend_title="",
                          plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # AI explanation
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Explanation")
    cA, cB = st.columns([3, 1])
    with cB:
        if st.button("✨ Explain with AI", use_container_width=True, type="primary"):
            with st.spinner("Creating a clear explanation..."):
                st.session_state.ai_explanation = ai_helper.explain_prediction(inputs, pred, risk)
    with cA:
        if st.session_state.ai_explanation:
            st.markdown(f"<div class='glass'>{st.session_state.ai_explanation}</div>",
                        unsafe_allow_html=True)
        else:
            st.info("Click **Explain with AI** to get a plain-English breakdown of your risk result.")

    # Diet & lifestyle
    st.markdown("### 🥗 Personalised Diet & Lifestyle Plan")
    cA, cB = st.columns([3, 1])
    with cB:
        if st.button("🍽️ Generate Plan", use_container_width=True, type="primary"):
            with st.spinner("Creating your personalised 7-day plan..."):
                st.session_state.diet_plan = ai_helper.diet_and_lifestyle_plan(inputs, pred, risk)
    with cA:
        if st.session_state.diet_plan:
            st.markdown(f"<div class='glass'>{st.session_state.diet_plan}</div>",
                        unsafe_allow_html=True)
        else:
            st.info("Click **Generate Plan** for a tailored 7-day diet, exercise, "
                    "and lifestyle plan based on your inputs.")

    # Static health recs
    st.markdown("### 💡 Quick Health Recommendations")
    recs = []
    if inputs["BMI"] > 30: recs.append("**BMI in obese range** — consider a structured weight-management plan.")
    elif inputs["BMI"] > 25: recs.append("**BMI overweight** — aim for 5–10% weight reduction.")
    if inputs["Glucose"] > 140: recs.append("**Glucose elevated** — repeat fasting glucose test recommended.")
    elif inputs["Glucose"] > 100: recs.append("**Glucose borderline** — monitor regularly and reduce refined carbs.")
    if inputs["BloodPressure"] > 90: recs.append("**Blood pressure high** — monitor daily and reduce sodium intake.")
    if inputs["Age"] > 45: recs.append("**Age 45+** — annual diabetes screening is advisable.")
    if not recs: recs.append("✅ All key indicators look reasonable. Maintain a balanced diet and regular exercise.")
    for r in recs: st.markdown(f"- {r}")

    # PDF
    st.markdown("---")
    pdf_bytes = build_pdf(st.session_state.user, inputs, pred, risk,
                          META["best_model"], META["metrics_best"]["accuracy"])
    st.download_button("📄 Download PDF Report", pdf_bytes,
                       file_name=f"diabetes_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                       mime="application/pdf", use_container_width=True, type="primary")


def page_history():
    st.title("📜 Prediction History")
    rows = db.get_user_predictions(st.session_state.user["id"])
    if not rows:
        st.info("No predictions yet."); return
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total predictions", len(df))
    c2.metric("Avg risk", f"{df.risk_score.mean():.1f}%")
    c3.metric("Diabetic predictions", int((df.prediction==1).sum()))
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["Result"] = df["prediction"].map({0:"Non-Diabetic", 1:"Diabetic"})
    df["Risk"] = df["risk_score"].apply(lambda v: f"{v:.1f}%")
    show = df[["created_at","Result","Risk","glucose","bmi","age",
               "blood_pressure","insulin"]].rename(columns={
        "created_at":"Date","glucose":"Glucose","bmi":"BMI","age":"Age",
        "blood_pressure":"BP","insulin":"Insulin"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download as CSV", df.to_csv(index=False).encode(),
                       file_name="my_prediction_history.csv", mime="text/csv")


def page_dashboard():
    st.title("📊 Model Performance Dashboard")
    m = META["metrics_best"]
    st.markdown(f"**Best model:** `{META['best_model']}`")
    cols = st.columns(5)
    for col, k, label in [(cols[0],"accuracy","Accuracy"),(cols[1],"precision","Precision"),
                          (cols[2],"recall","Recall"),(cols[3],"f1","F1"),
                          (cols[4],"roc_auc","ROC-AUC")]:
        col.metric(label, f"{m[k]:.3f}")

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Model comparison")
        comp = pd.DataFrame({
            "Model":list(META["metrics_all"].keys()),
            "Accuracy":[v["accuracy"] for v in META["metrics_all"].values()],
            "F1":[v["f1"] for v in META["metrics_all"].values()],
            "AUC":[v["roc_auc"] for v in META["metrics_all"].values()],
        })
        comp_long = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(comp_long, x="Model", y="Score", color="Metric",
                     barmode="group",
                     color_discrete_sequence=["#0d47a1","#1976d2","#42a5f5"])
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.subheader("ROC Curve (best model)")
        roc = META["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"AUC={m['roc_auc']:.3f}",
                                 line=dict(color="#0d47a1", width=3),
                                 fill="tozeroy", fillcolor="rgba(13,71,161,.1)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="Random",
                                 line=dict(dash="dash", color="grey")))
        fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          height=380, margin=dict(l=10,r=10,t=20,b=10),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        st.subheader("Confusion matrix")
        cm = m["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred Non-Diabetic","Pred Diabetic"],
                        y=["Actual Non-Diabetic","Actual Diabetic"])
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cD:
        st.subheader("Feature importance")
        imp = META["feature_importance"]
        idf = pd.DataFrame({"Feature":list(imp), "Importance":list(imp.values())}
                          ).sort_values("Importance")
        fig = px.bar(idf, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=20,b=10),
                          coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def page_admin():
    if not st.session_state.user.get("is_admin"):
        st.error("⛔ Access denied — admins only."); return
    st.title("🛡️ Admin Analytics Dashboard")
    s = db.admin_stats()
    cards = [("Total Users", s["users"], "Registered accounts"),
             ("Total Predictions", s["predictions"], "All-time"),
             ("Diabetic Cases", s["diabetic"],
              f"{(s['diabetic']/max(s['predictions'],1))*100:.1f}% of all"),
             ("Avg Risk", f"{s['avg_risk']:.1f}%", "Across all predictions")]
    cols = st.columns(4)
    for col, (l, v, sub) in zip(cols, cards):
        col.markdown(f"<div class='stat-card'><h3>{l}</h3>"
                     f"<div class='v'>{v}</div><div class='sub'>{sub}</div></div>",
                     unsafe_allow_html=True)

    st.markdown(" ")
    all_preds = db.admin_all_predictions()
    cA, cB = st.columns(2)
    with cA:
        st.subheader("📈 Predictions over time")
        if all_preds:
            df = pd.DataFrame(all_preds)
            df["date"] = pd.to_datetime(df["created_at"]).dt.date
            daily = df.groupby(["date","prediction"]).size().reset_index(name="count")
            daily["Result"] = daily["prediction"].map({0:"Non-Diabetic",1:"Diabetic"})
            fig = px.bar(daily, x="date", y="count", color="Result",
                         color_discrete_map={"Non-Diabetic":"#2e7d32","Diabetic":"#c62828"})
            fig.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions in the system yet.")
    with cB:
        st.subheader("🎯 Risk band distribution")
        if all_preds:
            df = pd.DataFrame(all_preds)
            df["band"] = pd.cut(df["risk_score"], bins=[-0.1,30,60,101],
                                labels=["Low (0–30%)","Moderate (30–60%)","High (60–100%)"])
            band_counts = df["band"].value_counts().reset_index()
            band_counts.columns = ["Band","Count"]
            fig = px.pie(band_counts, names="Band", values="Count", hole=0.55,
                         color="Band",
                         color_discrete_map={"Low (0–30%)":"#2e7d32",
                                             "Moderate (30–60%)":"#f9a825",
                                             "High (60–100%)":"#c62828"})
            fig.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    cC, cD = st.columns(2)
    with cC:
        st.subheader("🧮 Avg patient profile")
        if all_preds:
            df = pd.DataFrame(all_preds)
            avg = df[["glucose","bmi","blood_pressure","age","insulin"]].mean()
            metric_labels = {
                "glucose": "Glucose",
                "bmi": "BMI",
                "blood_pressure": "Blood Pressure",
                "age": "Age",
                "insulin": "Insulin",
            }
            adf = pd.DataFrame({
                "Metric": [metric_labels[m] for m in avg.index],
                "Average": avg.values,
            }).sort_values("Average")
            fig = px.bar(
                adf,
                x="Average",
                y="Metric",
                orientation="h",
                text=adf["Average"].round(1),
                color_discrete_sequence=["#1976d2"],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=360,
                margin=dict(l=130, r=35, t=10, b=45),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Average",
                yaxis_title=None,
            )
            fig.update_xaxes(range=[0, max(adf["Average"]) * 1.2], gridcolor="rgba(128,128,128,.25)")
            fig.update_yaxes(automargin=True)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data yet.")
    with cD:
        st.subheader("👥 Most active users")
        users = db.admin_all_users()
        if users:
            udf = pd.DataFrame(users).sort_values("prediction_count", ascending=False).head(10)
            udf["Avg Risk"] = udf["avg_risk"].fillna(0).map(lambda v: f"{v:.1f}% avg risk")
            fig = px.bar(
                udf,
                x="prediction_count",
                y="username",
                orientation="h",
                text="prediction_count",
                hover_data={"Avg Risk": True, "prediction_count": True, "username": False},
                labels={"prediction_count":"Predictions", "username": ""},
                color_discrete_sequence=["#ff7043"],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            max_predictions = max(int(udf["prediction_count"].max()), 1)
            fig.update_layout(
                height=360,
                margin=dict(l=135, r=45, t=10, b=45),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predictions",
                yaxis_title=None,
            )
            fig.update_xaxes(range=[0, max_predictions + 1], dtick=1, gridcolor="rgba(128,128,128,.25)")
            fig.update_yaxes(automargin=True)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No users yet.")

    st.subheader("👥 All users")
    users = db.admin_all_users()
    if users:
        udf = pd.DataFrame(users)
        udf["Role"] = udf["is_admin"].map({1:"Admin",0:"User"})
        udf["Avg Risk"] = udf["avg_risk"].apply(lambda v: f"{v:.1f}%")
        udf["Joined"] = pd.to_datetime(udf["created_at"]).dt.strftime("%Y-%m-%d")
        st.dataframe(udf[["username","full_name","email","Role","Joined",
                          "prediction_count","Avg Risk"]].rename(columns={
            "username":"Username","full_name":"Full Name","email":"Email",
            "prediction_count":"# Predictions"}),
            use_container_width=True, hide_index=True)


def page_chatbot():
    st.title("🤖 Health Assistant")
    st.caption("Ask general questions about diabetes awareness, diet, prevention, and your latest risk result.")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role":"assistant","content":
            "Hi! I can help with general diabetes awareness, symptoms, diet, "
            "prevention, and questions about your latest risk result."}]
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    q = st.chat_input("Ask a question...")
    if q:
        st.session_state.chat.append({"role":"user","content":q})
        with st.chat_message("user"): st.markdown(q)
        api_key = ai_helper.get_api_key()
        if not api_key:
            reply = ("⚠️ The health assistant is not configured on this computer yet. "
                     "Add your Gemini API key to the `.env` file to enable replies.")
        else:
            with st.spinner("Thinking..."):
                reply = ai_helper.health_assistant_reply(
                    st.session_state.chat,
                    st.session_state.last_prediction,
                )
        st.session_state.chat.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"): st.markdown(reply)


def page_explorer():
    st.title("🔬 Health Data Explorer")
    df = load_dataset()
    df["Risk Group"] = np.where(df["Outcome"] == 1, "Higher risk", "Lower risk")
    features = [c for c in df.columns if c not in ["Outcome", "Risk Group"]]
    feature_names = {
        "Pregnancies": "Pregnancies",
        "Glucose": "Glucose",
        "BloodPressure": "Blood pressure",
        "SkinThickness": "Skin thickness",
        "Insulin": "Insulin",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "Family history score",
        "Age": "Age",
    }
    label_to_feature = {v: k for k, v in feature_names.items()}

    st.caption("Explore health records, compare risk groups, and understand how different health measurements relate to one another.")

    st.markdown("### Filters")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        risk_filter = st.radio(
            "Risk group",
            ["All records", "Lower risk", "Higher risk"],
            horizontal=True,
        )
    with f2:
        age_range = st.slider(
            "Age range",
            int(df["Age"].min()),
            int(df["Age"].max()),
            (int(df["Age"].min()), int(df["Age"].max())),
        )
    with f3:
        glucose_range = st.slider(
            "Glucose range",
            int(df["Glucose"].min()),
            int(df["Glucose"].max()),
            (int(df["Glucose"].min()), int(df["Glucose"].max())),
        )

    filtered = df[
        df["Age"].between(age_range[0], age_range[1])
        & df["Glucose"].between(glucose_range[0], glucose_range[1])
    ].copy()
    if risk_filter != "All records":
        filtered = filtered[filtered["Risk Group"] == risk_filter]

    if filtered.empty:
        st.warning("No records match the selected filters. Adjust the filters to see results.")
        return

    higher_risk_count = int((filtered["Outcome"] == 1).sum())
    higher_risk_rate = higher_risk_count / len(filtered) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(filtered):,}")
    m2.metric("High Risk", f"{higher_risk_count:,}")
    m3.metric("High Risk %", f"{higher_risk_rate:.1f}%")
    m4.metric("Avg Glucose", f"{filtered['Glucose'].mean():.1f}")

    overview, compare, relationships, table = st.tabs(
        ["Overview", "Compare Groups", "Relationships", "Data Table"]
    )

    with overview:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            selected_label = st.selectbox("Choose a health measurement", list(label_to_feature.keys()))
            selected_feature = label_to_feature[selected_label]
            fig = px.histogram(
                filtered,
                x=selected_feature,
                color="Risk Group",
                nbins=30,
                barmode="overlay",
                color_discrete_map={"Lower risk":"#2e7d32", "Higher risk":"#c62828"},
                labels={selected_feature:selected_label},
            )
            fig.update_layout(
                title=f"{selected_label} distribution",
                height=410,
                margin=dict(l=10, r=10, t=55, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Risk group",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            group_counts = filtered["Risk Group"].value_counts().reset_index()
            group_counts.columns = ["Risk Group", "Records"]
            fig = px.pie(
                group_counts,
                names="Risk Group",
                values="Records",
                hole=.55,
                color="Risk Group",
                color_discrete_map={"Lower risk":"#2e7d32", "Higher risk":"#c62828"},
            )
            fig.update_layout(
                title="Risk group mix",
                height=410,
                margin=dict(l=10, r=10, t=55, b=10),
                legend_title_text="Risk group",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Measurement summary")
        summary = filtered[features].agg(["mean", "median", "min", "max"]).T.round(2)
        summary = summary.rename(index=feature_names, columns={
            "mean": "Average",
            "median": "Median",
            "min": "Lowest",
            "max": "Highest",
        })
        st.dataframe(summary, use_container_width=True)

    with compare:
        group_means = filtered.groupby("Risk Group")[features].mean().T.reset_index()
        group_means["Measurement"] = group_means["index"].map(feature_names)
        group_means = group_means.drop(columns=["index"]).melt(
            id_vars="Measurement",
            var_name="Risk Group",
            value_name="Average value",
        )
        fig = px.bar(
            group_means,
            x="Measurement",
            y="Average value",
            color="Risk Group",
            barmode="group",
            color_discrete_map={"Lower risk":"#2e7d32", "Higher risk":"#c62828"},
        )
        fig.update_layout(
            title="Average measurements by risk group",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30,
            legend_title_text="Risk group",
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_compare_label = st.selectbox(
            "Compare one measurement in detail",
            list(label_to_feature.keys()),
            key="compare_feature",
        )
        selected_compare = label_to_feature[selected_compare_label]
        fig = px.box(
            filtered,
            x="Risk Group",
            y=selected_compare,
            color="Risk Group",
            points="all",
            color_discrete_map={"Lower risk":"#2e7d32", "Higher risk":"#c62828"},
            labels={selected_compare:selected_compare_label},
        )
        fig.update_layout(
            title=f"{selected_compare_label} spread by risk group",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with relationships:
        r1, r2 = st.columns(2)
        with r1:
            x_label = st.selectbox("X-axis", list(label_to_feature.keys()), index=1)
        with r2:
            y_label = st.selectbox("Y-axis", list(label_to_feature.keys()), index=5)
        x_feature = label_to_feature[x_label]
        y_feature = label_to_feature[y_label]
        fig = px.scatter(
            filtered,
            x=x_feature,
            y=y_feature,
            color="Risk Group",
            size="Age",
            hover_data=["Age", "Glucose", "BMI", "BloodPressure"],
            color_discrete_map={"Lower risk":"#2e7d32", "Higher risk":"#c62828"},
            labels={x_feature:x_label, y_feature:y_label},
        )
        fig.update_layout(
            title=f"{x_label} vs {y_label}",
            height=460,
            margin=dict(l=10, r=10, t=55, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Risk group",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation between measurements")
        corr = filtered[features].corr().rename(index=feature_names, columns=feature_names).round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(height=540, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with table:
        st.subheader("Filtered records")
        display = filtered.drop(columns=["Outcome"]).rename(columns=feature_names)
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv = display.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download filtered data",
            csv,
            "health_data_filtered.csv",
            "text/csv",
            use_container_width=True,
        )


def page_about():
    st.title("ℹ️ About This Site")
    st.markdown("""
### GlucoCare Health Portal

This website helps users check their possible diabetes risk by entering basic
health measurements such as glucose level, blood pressure, BMI, age, and other
related details.

#### What You Can Do Here
1. Create an account and sign in securely.
2. Enter health details to receive a diabetes risk result.
3. View your previous risk assessments in one place.
4. Track changes in your results over time.
5. Download a simple report for your records.
6. Read general health guidance related to diabetes awareness and prevention.

#### Who This Site Is For
This site is useful for students, patients, and general users who want a simple
way to understand diabetes-related risk factors and keep their assessment
history organized.

#### Important Note
The result shown on this website is only an awareness-based estimate. It should
not be treated as a medical diagnosis. For proper testing, treatment, or medical
advice, please consult a qualified doctor or healthcare professional.
""")


# ────────────────────────── Router ──────────────────────────
if not st.session_state.user:
    auth_screen()
else:
    sidebar()
    pages = {
        "Home":         page_home,
        "Predict":      page_predict,
        "History":      page_history,
        "Dashboard":    page_dashboard,
        "Admin Panel":  page_admin,
        "Health Assistant": page_chatbot,
        "AI Chatbot":   page_chatbot,
        "Data Explorer":page_explorer,
        "About":        page_about,
    }
    pages.get(st.session_state.page, page_home)()

if st.session_state.get("page") != "About":
    st.markdown(
        "<div class='footer'>This website provides diabetes risk awareness only. "
        "It is not a medical diagnosis. Please consult a qualified doctor for "
        "proper testing, treatment, and health advice.</div>",
        unsafe_allow_html=True,
    )
