"""
Streamlit UI for AI Agriculture Field Assistant (Punjab, Pakistan)
Bilingual (Roman Urdu / Urdu / English) crop disease diagnosis with RAG, weather, vision, web search.
"""
import sys
import os
import uuid
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Agriculture Field Assistant - Punjab",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.agripunjab.gov.pk",
        "About": "AI Agriculture Field Assistant - Punjab Agriculture Department, Pakistan",
    },
)

# ─── Imports (after sys.path) ─────────────────────────────────────────────────
from app.config import PUNJAB_DISTRICTS, PUNJAB_AGRI_HELPLINE, GROQ_API_KEY
from app.graph.decision_graph import field_agent_graph
from app.search_agent import web_search_agent


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Overall theme ---- */
    .stApp { background: #f5f7f0; }
    
    /* ---- Header ---- */
    .main-header {
        background: linear-gradient(135deg, #1a5c2a 0%, #2d8a4e 60%, #4caf50 100%);
        color: white; padding: 20px 28px; border-radius: 12px; margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 4px 0 0; opacity: 0.88; font-size: 0.95rem; }

    /* ---- Diagnosis banner ---- */
    .diag-banner-high   { background:#e8f5e9; border-left:5px solid #2e7d32; padding:14px 18px; border-radius:8px; margin-bottom:12px; }
    .diag-banner-medium { background:#fff8e1; border-left:5px solid #f9a825; padding:14px 18px; border-radius:8px; margin-bottom:12px; }
    .diag-banner-low    { background:#fce4ec; border-left:5px solid #c62828; padding:14px 18px; border-radius:8px; margin-bottom:12px; }

    /* ---- Field test box ---- */
    .field-test-box {
        background:#fff3cd; border:1px solid #ffc107; border-radius:8px;
        padding:14px 18px; margin:12px 0;
    }

    /* ---- Section cards ---- */
    .section-card {
        background:white; border-radius:10px; padding:18px 20px;
        box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:14px;
    }

    /* ---- Helpline button ---- */
    .helpline-btn {
        display:inline-block; background:#c62828; color:white!important;
        padding:12px 28px; border-radius:30px; font-size:1.1rem;
        font-weight:700; text-decoration:none; text-align:center;
        box-shadow:0 4px 12px rgba(198,40,40,0.35); letter-spacing:0.5px;
    }
    .helpline-btn:hover { background:#b71c1c; }

    /* ---- Quick prompt pills ---- */
    .quick-pill {
        display:inline-block; background:#e8f5e9; border:1px solid #a5d6a7;
        border-radius:20px; padding:5px 14px; margin:4px; font-size:0.85rem;
        cursor:pointer; color:#2e7d32; font-weight:500;
    }

    /* ---- Confidence badge ---- */
    .conf-badge-high   { background:#2e7d32; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
    .conf-badge-medium { background:#f9a825; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; }
    .conf-badge-low    { background:#c62828; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] { background:#f0f4f0; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; }

    /* hide streamlit default header */ 
    #MainMenu, header, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helper: health check ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_system_health():
    """Check core system components (cached 60s)."""
    health = {"groq": False, "qdrant": False, "weather": False, "search": False}
    try:
        from app.knowledge_base.qdrant_rag import qdrant_rag
        health["qdrant"] = qdrant_rag.client is not None
    except Exception:
        pass
    try:
        from app.weather_service import weather_service
        health["weather"] = weather_service is not None
    except Exception:
        pass
    health["groq"] = bool(GROQ_API_KEY)
    health["search"] = True  # ddgs always available
    return health


def status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Kisan Sahulat Panel")
    st.markdown("**Punjab Agriculture Field Assistant**")
    st.divider()

    # --- System health -------------------------------------------------------
    with st.expander("⚙️ System Health", expanded=False):
        health = get_system_health()
        st.markdown(
            f"""
| Component | Status |
|-----------|--------|
| Groq LLM  | {status_dot(health['groq'])} |
| Qdrant KB | {status_dot(health['qdrant'])} |
| Weather   | {status_dot(health['weather'])} |
| Web Search| {status_dot(health['search'])} |
"""
        )
        if st.button("Refresh Health"):
            get_system_health.clear()
            st.rerun()

    st.divider()

    # --- Crop selector -------------------------------------------------------
    crop_options = {
        "Wheat (Gandum) 🌾":    "Wheat (Gandum)",
        "Cotton (Kapas) 🌿":    "Cotton (Kapas)",
        "Rice (Dhan) 🌾":       "Rice (Dhan)",
        "Maize (Makai) 🌽":     "Maize (Makai)",
    }
    selected_crop_label = st.selectbox(
        "🌱 Fasal (Crop) Chunein",
        list(crop_options.keys()),
        index=0,
        help="Apni fasal select karein",
    )
    selected_crop = crop_options[selected_crop_label]

    # --- District selector ---------------------------------------------------
    district_names = {v["name"]: k for k, v in PUNJAB_DISTRICTS.items()}
    district_display = [v["name"] for v in PUNJAB_DISTRICTS.values()]
    selected_district_name = st.selectbox(
        "📍 Zila (District) Chunein",
        district_display,
        index=0,
        help="Apna zila chunein — mausam is se milega",
    )
    selected_district_key = district_names[selected_district_name]
    dist_info = PUNJAB_DISTRICTS[selected_district_key]

    st.caption(f"Zone: {dist_info['zone']}")
    st.caption(f"GPS: {dist_info['lat']}°N, {dist_info['lon']}°E")

    # --- Crop stage ----------------------------------------------------------
    stage_options = [
        "Vegetative / Tillering",
        "Tillering (Shakhain nikalna)",
        "Booting (Gabha stage)",
        "Heading / Sitta nikalna",
        "Grain filling (Dana banna)",
        "Maturity / Harvest",
    ]
    selected_stage = st.selectbox(
        "📅 Fasal ka Marhala (Growth Stage)",
        stage_options,
        index=0,
    )

    # --- Image upload --------------------------------------------------------
    st.divider()
    uploaded_image = st.file_uploader(
        "📷 Pattay ki Tasveer Upload Karein (Optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Beemar pattay ya fasal ki tasveer lagain — AI analysis karega",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded leaf image", use_container_width=True)

    # --- Quick prompts -------------------------------------------------------
    st.divider()
    st.markdown("**⚡ Quick Queries:**")

    quick_prompts = {
        "🟡 Pattay Peelay":          "Gandum ke pattay peelay ho rahe hain aur haldi jaisa powder bhi hai",
        "🐛 Keera / Tela":           "Kapas mein safaid makhi aur pattay merey ho kar gir rahe hain",
        "💧 Pani Kharabi":           "Baarish ke baad khet mein pani khara hai aur fasal murjha rahi hai",
        "🌿 Nitrogen Kami":          "Puranay neechay waalay pattay peeley hain lekin koi powder nahi hai",
    }

    quick_choice = st.radio(
        "Ek query chunein ya khud likhein:",
        ["(Khud likhein...)"] + list(quick_prompts.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    # Helpline
    st.markdown(
        f'<a class="helpline-btn" href="tel:0800-15000">📞 Helpline: {PUNJAB_AGRI_HELPLINE}</a>',
        unsafe_allow_html=True,
    )
    st.caption("Punjab Agriculture Department — Muft Mashwara")


# ─── Main Area ────────────────────────────────────────────────────────────────

# Header banner
st.markdown(
    """
    <div class="main-header">
      <h1>🌾 AI Agriculture Field Assistant</h1>
      <p>Punjab Agriculture Department Pakistan &nbsp;|&nbsp; Evidence-Based Crop Disease Diagnosis &nbsp;|&nbsp; Bilingual Advisory</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Determine query text
if quick_choice != "(Khud likhein...)":
    default_query = quick_prompts[quick_choice]
else:
    default_query = ""

# --- Query input -------------------------------------------------------------
col_q, col_btn = st.columns([5, 1])
with col_q:
    user_query = st.text_area(
        "✍️ Apna masla ya sawal likhein (Roman Urdu / Urdu / English):",
        value=default_query,
        height=110,
        placeholder="Misal: Gandum ke pattay peelay ho rahe hain aur unpar haldi jaisa powder hai...",
        key="user_query_input",
    )
with col_btn:
    st.markdown("<br><br>", unsafe_allow_html=True)
    submit = st.button("🔍 Tashkhees Karein", type="primary", use_container_width=True)
    clear  = st.button("🗑️ Saaf Karein", use_container_width=True)

if clear:
    st.session_state.pop("last_result", None)
    st.rerun()

# --- Audio note --------------------------------------------------------------
st.info(
    "🎙️ **Voice Note:** Streamlit mein microphone seedha kaam nahi karta. "
    "Apna sawal text mein likhein ya audio file upload karein (future feature)."
)

# ─── Diagnosis Execution ─────────────────────────────────────────────────────
if submit and user_query.strip():
    # Save image to temp file if uploaded
    image_path = None
    if uploaded_image:
        suffix = os.path.splitext(uploaded_image.name)[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_image.read())
        tmp.close()
        image_path = tmp.name

    # Build initial state
    initial_state = {
        "session_id": str(uuid.uuid4()),
        "user_message": user_query,
        "audio_path": None,
        "image_path": image_path,
        "crop": selected_crop,
        "district": selected_district_name,
        "crop_stage": selected_stage,
        "symptoms": [],
        "weather_context": None,
        "vision_findings": None,
        "retrieved_evidence": [],
        "web_search_results": None,
        "web_search_summary": None,
        "web_search_performed": False,
        "differential_diagnosis": [],
        "top_diagnosis": None,
        "confidence_score": 0.0,
        "clarification_needed": False,
        "clarification_questions": [],
        "escalation_needed": False,
        "escalation_reason": None,
        "expert_helpline": PUNJAB_AGRI_HELPLINE,
        "final_advisory_urdu": "",
        "final_advisory_roman_urdu": "",
        "distinguishing_test": "",
        "official_sources": [],
        "recommended_actions": [],
        "safety_guidelines": [],
    }

    with st.spinner("🔄 AI Tashkhees ho rahi hai... Zara intezaar farmaein..."):
        start_t = time.time()
        try:
            result = field_agent_graph.invoke(initial_state)
            elapsed = round(time.time() - start_t, 1)
            st.session_state["last_result"] = result
            st.session_state["elapsed"] = elapsed
        except Exception as e:
            st.error(f"Graph invoke error: {e}")
            result = None

elif submit and not user_query.strip():
    st.warning("⚠️ Kripya pehle apna masla likhein ya quick query chunein.")


# ─── Results Display ─────────────────────────────────────────────────────────
result = st.session_state.get("last_result")

if result:
    elapsed   = st.session_state.get("elapsed", "?")
    top_diag  = result.get("top_diagnosis") or "Undetermined"
    conf      = result.get("confidence_score") or 0.0
    conf_pct  = int(conf * 100)
    escalate  = result.get("escalation_needed", False)
    diff      = result.get("differential_diagnosis") or []
    weather   = result.get("weather_context") or {}
    vision    = result.get("vision_findings") or {}
    dist_test = result.get("distinguishing_test") or ""
    recs      = result.get("recommended_actions") or []
    safety    = result.get("safety_guidelines") or []
    sources   = result.get("official_sources") or []
    web_res   = result.get("web_search_results") or []
    web_sum   = result.get("web_search_summary") or ""
    adv_roman = result.get("final_advisory_roman_urdu") or ""
    adv_urdu  = result.get("final_advisory_urdu") or ""
    crop_used = result.get("crop") or selected_crop
    dist_used = result.get("district") or selected_district_name
    symptoms  = result.get("symptoms") or []
    clarif    = result.get("clarification_needed", False)
    clarif_qs = result.get("clarification_questions") or []

    st.markdown(f"<small style='color:#888'>Analysis complete in {elapsed}s</small>", unsafe_allow_html=True)
    st.divider()

    # ── Clarification needed? ─────────────────────────────────────────────
    if clarif and clarif_qs:
        st.warning("❓ **Thodi aur maloomat chahiye (Clarification Needed):**")
        for q in clarif_qs:
            st.markdown(f"- {q}")
        st.info("Upar apni query mein yeh details shamil karein aur dobara submit karein.")

    # ── 1. DIAGNOSIS BANNER ───────────────────────────────────────────────
    if conf >= 0.70:
        banner_cls  = "diag-banner-high"
        badge_cls   = "conf-badge-high"
        status_icon = "✅"
    elif conf >= 0.50:
        banner_cls  = "diag-banner-medium"
        badge_cls   = "conf-badge-medium"
        status_icon = "⚠️"
    else:
        banner_cls  = "diag-banner-low"
        badge_cls   = "conf-badge-low"
        status_icon = "🔴"

    st.markdown(
        f"""
        <div class="{banner_cls}">
          <h2 style="margin:0 0 6px">{status_icon} {top_diag}</h2>
          <span>Fasal: <strong>{crop_used}</strong> &nbsp;|&nbsp; Zila: <strong>{dist_used}</strong></span>
          &nbsp;&nbsp;
          <span class="{badge_cls}">Yaqeen: {conf_pct}%</span>
          {"&nbsp;&nbsp;<span style='color:#c62828;font-weight:700'>⚠️ EXPERT ESCALATION RECOMMENDED</span>" if escalate else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Escalation alert
    if escalate:
        reason = result.get("escalation_reason") or ""
        st.error(
            f"🚨 **Expert se Rabta Zaroori hai!** {reason}\n\n"
            f"**Punjab Agri Helpline: {PUNJAB_AGRI_HELPLINE}** — Muft call karein!"
        )

    # Detected symptoms
    if symptoms:
        st.markdown(
            "**Detected symptoms:** " + " &nbsp;|&nbsp; ".join(f"`{s}`" for s in symptoms),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 2. FIELD DISTINGUISHING TEST ──────────────────────────────────────
    if dist_test:
        st.markdown(
            f"""
            <div class="field-test-box">
              <strong>🔬 Khet mein Tasdeeq ka Tareeqa (Field Test):</strong><br>
              {dist_test}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 3. DIFFERENTIAL DIAGNOSIS CHART ───────────────────────────────────
    if diff:
        st.markdown("### 📊 Differential Diagnosis (Mumkin Asbaab)")
        for item in diff[:5]:
            cname  = item.get("condition_name", "?")
            score  = item.get("likelihood_score", 0.0)
            cat    = item.get("category", "")
            marks  = item.get("diagnostic_markers") or []
            col_a, col_b = st.columns([3, 7])
            with col_a:
                color = "#2e7d32" if score >= 0.60 else ("#f9a825" if score >= 0.35 else "#9e9e9e")
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>{cname}</span><br>"
                    f"<small style='color:#888'>{cat}</small>",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.progress(min(score, 1.0), text=f"{int(score*100)}%")
            if marks:
                with st.expander(f"Evidence markers ({len(marks)})", expanded=False):
                    for m in marks:
                        st.markdown(f"- {m}")

    st.divider()

    # ── 4. WEATHER CONTEXT ────────────────────────────────────────────────
    if weather:
        st.markdown("### 🌦️ Mausam ka Haal (Weather Context)")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("🌡️ Temperature", f"{weather.get('temperature', 'N/A')}°C")
        wc2.metric("💧 Humidity",    f"{weather.get('humidity', 'N/A')}%")
        wc3.metric("🌧️ Rain (mm)",   f"{weather.get('precipitation_mm', 0.0)}")
        wc4.metric("💨 Wind (km/h)", f"{weather.get('wind_speed', 'N/A')}")

        risk_list = weather.get("risk_assessment") or []
        if risk_list:
            risk_colors = {
                "YELLOW_RUST": "#f9a825",
                "APHID": "#ef6c00",
                "WATERLOGGING": "#1565c0",
                "HIGH": "#c62828",
            }
            for r in risk_list:
                color = next((v for k, v in risk_colors.items() if k in r), "#555")
                st.markdown(
                    f"<span style='background:{color};color:white;border-radius:12px;"
                    f"padding:3px 10px;font-size:0.82rem;margin:2px'>{r}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No critical weather risks detected for this district.")

        if weather.get("summary"):
            st.caption(weather["summary"])

    # ── 5. VISION FINDINGS ────────────────────────────────────────────────
    if vision.get("has_image"):
        st.divider()
        st.markdown("### 📷 Tasveer ka Jaiza (Vision Analysis)")
        v1, v2, v3 = st.columns(3)
        v1.metric("Yellow %", f"{round(vision.get('chlorosis_yellow_pct', 0), 1)}%")
        v2.metric("Brown %",  f"{round(vision.get('necrosis_brown_pct', 0), 1)}%")
        v3.metric("Stripes?", "Han ✅" if vision.get("linear_stripes_detected") else "Nahi ❌")
        if vision.get("visual_summary"):
            st.info(vision["visual_summary"])

    st.divider()

    # ── 6. ADVISORY (Tabbed: Roman Urdu / Urdu) ───────────────────────────
    st.markdown("### 📋 Mehakma Ziraat Punjab ki Sifarishaat (Advisory)")
    tab_roman, tab_urdu, tab_recs = st.tabs(["Roman Urdu", "اردو", "Quick Actions"])

    with tab_roman:
        st.markdown(adv_roman if adv_roman else "_Advisory generating..._")

    with tab_urdu:
        st.markdown(
            f"<div style='direction:rtl;text-align:right;font-size:1.05rem;line-height:1.8'>"
            f"{adv_urdu or '_اردو ایڈوائزری دستیاب نہیں_'}</div>",
            unsafe_allow_html=True,
        )

    with tab_recs:
        if recs:
            st.markdown("**Official Recommended Actions:**")
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}.** {r}")
        else:
            st.info("Recommendations included in the advisory above.")

    # ── 7. SAFETY GUIDELINES ──────────────────────────────────────────────
    if safety:
        st.warning(
            "⚠️ **Hifazati Tadabeer (Safety & PHI):**\n\n"
            + "\n\n".join(f"- {s}" for s in safety)
        )

    # ── 8. WEB SEARCH RESULTS ─────────────────────────────────────────────
    if web_res or web_sum:
        st.divider()
        st.markdown("### 🌐 Live Internet Search Findings")
        if web_sum:
            st.info(web_sum)
        if web_res:
            with st.expander(f"Search Results ({len(web_res)} results)", expanded=False):
                for item in web_res:
                    title  = item.get("title") or item.get("query", "Result")
                    url    = item.get("url") or item.get("link", "")
                    body   = item.get("body") or item.get("summary") or item.get("text") or ""
                    source = item.get("source") or ""
                    st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
                    if source:
                        st.caption(f"Source: {source}")
                    if body:
                        st.markdown(f"> {body[:300]}{'...' if len(body) > 300 else ''}")
                    st.markdown("---")

    # ── 9. OFFICIAL SOURCES ───────────────────────────────────────────────
    if sources:
        with st.expander("📚 Sarkari Hawala (Official Sources)", expanded=False):
            for s in sources:
                st.markdown(f"- {s}")

    # ── 10. HELPLINE FOOTER ───────────────────────────────────────────────
    st.divider()
    fc1, fc2, fc3 = st.columns([2, 3, 2])
    with fc2:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;background:#fff3cd;border-radius:12px;border:1px solid #ffc107;">
              <div style="font-size:1.1rem;font-weight:700;color:#c62828;margin-bottom:8px">
                📞 Punjab Agriculture Helpline
              </div>
              <div style="font-size:1.6rem;font-weight:800;color:#1a5c2a;letter-spacing:1px">
                {PUNJAB_AGRI_HELPLINE}
              </div>
              <div style="margin-top:8px;color:#555;font-size:0.88rem">
                Muft Mashwara · Free Advisory · روزانہ دستیاب
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─── Empty state ─────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#888">
          <div style="font-size:4rem;margin-bottom:16px">🌾</div>
          <h3 style="color:#2d8a4e">Kisan Bhai, AI Field Assistant tayyar hai!</h3>
          <p>Bayin taraf se <strong>fasal</strong> aur <strong>zila</strong> chunein,<br>
          phir apna masla upar likhein aur <strong>Tashkhees Karein</strong> button dabain.</p>
          <hr style="max-width:300px;margin:24px auto">
          <p style="font-size:0.9rem">
            <strong>Suported crops:</strong> Wheat · Cotton · Rice · Maize<br>
            <strong>Districts:</strong> 15 Punjab districts<br>
            <strong>Languages:</strong> Roman Urdu · Urdu · English<br>
            <strong>AI:</strong> LangGraph · Qdrant RAG · Groq LLM · Web Search
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Persistent sidebar footer ───────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown(
        """
        <small style='color:#888'>
        <strong>AI Agriculture Field Assistant</strong><br>
        Punjab Agriculture Department, Pakistan<br>
        Sources: AARI · EVPMRC · Pest Warning Dept<br>
        v2.0 | LangGraph + Qdrant RAG + Web Search
        </small>
        """,
        unsafe_allow_html=True,
    )

