"""
Automated Policy Summarization & Scenario Simulation System
"""

import os
from dotenv import load_dotenv
import streamlit as st
import PyPDF2
import io
import time
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ─── LOAD ENV ─────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("API key not found. Please set GROQ_API_KEY in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Summarization & Scenario Simulation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ffffff;
        color: #1a1a1a;
    }
    .main { background-color: #ffffff; }

    /* Gold accent variables */
    :root {
        --gold: #C9A84C;
        --gold-light: #F5E6C0;
        --gold-dark: #9C7A28;
        --white: #ffffff;
        --gray-soft: #f8f6f1;
        --border: #e8dfc8;
    }

    /* Header */
    .header-block {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-bottom: 3px solid #C9A84C;
        padding: 28px 36px;
        border-radius: 10px;
        margin-bottom: 28px;
        text-align: center;
    }
    .header-block h1 {
        color: #C9A84C;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0 0 6px 0;
    }
    .header-block p {
        color: #cccccc;
        font-size: 0.88rem;
        margin: 0;
    }

    /* Panel headers */
    .panel-header {
        background: linear-gradient(90deg, #C9A84C, #e8c86e);
        color: #1a1a1a;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.4px;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #C9A84C, #9C7A28);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Reset button */
    .reset-btn > button {
        background: #f0f0f0 !important;
        color: #555 !important;
        border: 1px solid #ddd !important;
    }

    /* Output cards */
    .output-section {
        background: #f8f6f1;
        border: 1px solid #e8dfc8;
        border-left: 4px solid #C9A84C;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .output-section h4 {
        color: #9C7A28;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 0 0 10px 0;
        border-bottom: 1px solid #e8dfc8;
        padding-bottom: 6px;
    }
    .output-section p {
        color: #2a2a2a;
        font-size: 0.9rem;
        line-height: 1.7;
        margin: 0;
    }

    /* Word count */
    .word-count {
        text-align: right;
        font-size: 0.76rem;
        color: #9C7A28;
        margin-top: 4px;
    }

    /* Divider */
    .gold-divider {
        border: none;
        border-top: 2px solid #C9A84C;
        margin: 20px 0;
        opacity: 0.4;
    }

    /* File uploader */
    .stFileUploader { border: 2px dashed #C9A84C !important; border-radius: 8px; }

    /* Selectbox + textarea */
    .stSelectbox, .stTextArea { border-radius: 8px !important; }
    .stTextArea textarea { border: 1px solid #e8dfc8 !important; }

    /* Download button styling */
    .stDownloadButton > button {
        background: #1a1a1a !important;
        color: #C9A84C !important;
        border: 1px solid #C9A84C !important;
        border-radius: 8px;
        font-size: 0.85rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def count_words(text):
    """Return word count of a string."""
    return len(text.split()) if text else 0


def call_groq(system_prompt, user_prompt, model="llama-3.3-70b-versatile"):
    """Call Groq API and return content string."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def generate_summary(policy_text):
    """Generate structured policy summary."""
    system = (
        "You are an expert policy analyst. Analyse policy documents and return structured summaries. "
        "Be concise, professional, and academic in tone. Do not use bullet symbols; use plain sentences."
    )
    user = f"""Analyse the following policy text and produce a structured summary with exactly these three sections.
Return plain text only, labelled exactly as below:

MAIN GOALS:
[Write 3-4 sentences summarising the primary objectives of the policy.]

KEY MEASURES:
[Write 3-4 sentences describing the main mechanisms, tools, or interventions the policy introduces.]

OVERALL DIRECTION:
[Write 2-3 sentences describing the strategic direction and long-term vision of the policy.]

Policy Text:
{policy_text[:6000]}
"""
    return call_groq(system, user)


def generate_scenario(policy_text, scenario):
    """Generate scenario simulation output."""
    system = (
        "You are a strategic policy advisor and risk analyst. "
        "Evaluate how a given policy would perform under specific real-world scenarios. "
        "Be analytical, structured, and professional."
    )
    user = f"""Given the following policy, simulate how it would respond to the scenario described.
Return plain text only, labelled exactly as below:

SCENARIO IMPACT:
[Write 3-4 sentences describing how the policy directly addresses or is affected by this scenario.]

STRATEGIC ANALYSIS:
[Write 3-4 sentences on the strategic implications — how should the policy be applied or adapted?]

RISKS AND OPPORTUNITIES:
[Write 3-4 sentences identifying key risks the scenario introduces and opportunities the policy can leverage.]

Policy:
{policy_text[:4000]}

Scenario:
{scenario}
"""
    return call_groq(system, user)


def parse_sections(text, keys):
    """
    Parse LLM output into a dict keyed by section names.
    keys: list of tuples (label_in_output, dict_key)
    """
    result = {}
    upper = text.upper()
    positions = []
    for label, key in keys:
        idx = upper.find(label.upper())
        if idx != -1:
            positions.append((idx, label, key))
    positions.sort()
    for i, (idx, label, key) in enumerate(positions):
        start = idx + len(label) + 1  # skip the colon
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        result[key] = text[start:end].strip().strip(":")
    return result


def build_pdf(title, sections, filename_prefix):
    """Build a styled PDF and return bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.8*inch, leftMargin=0.8*inch,
        topMargin=0.8*inch, bottomMargin=0.8*inch
    )

    gold = colors.HexColor("#C9A84C")
    dark = colors.HexColor("#1a1a1a")
    soft = colors.HexColor("#f8f6f1")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Normal"],
        fontSize=16, textColor=gold, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle("SubTitle", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER, spaceAfter=16)
    section_head = ParagraphStyle("SectionHead", parent=styles["Normal"],
        fontSize=10, textColor=dark, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6, textTransform="uppercase")
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, textColor=dark, leading=16, spaceAfter=6)

    story = []
    story.append(Paragraph("Automated Policy Summarization &amp; Scenario Simulation System", title_style))
    story.append(Paragraph(f"{title} — Generated Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=gold, spaceAfter=14))

    for heading, content in sections.items():
        story.append(Paragraph(heading, section_head))
        story.append(HRFlowable(width="100%", thickness=0.5, color=gold, spaceAfter=6))
        for line in content.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), body_style))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=gold))
    story.append(Paragraph("Policy Analysis System", subtitle_style))

    doc.build(story)
    return buffer.getvalue()


# ─── SESSION STATE ───────────────────────────────────────────────────────────────
for key in ["summary_raw", "scenario_raw", "policy_text"]:
    if key not in st.session_state:
        st.session_state[key] = ""


# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>Automated Policy Summarization &amp; Scenario Simulation System</h1>
    
</div>
""", unsafe_allow_html=True)


# ─── TWO-PANEL LAYOUT ────────────────────────────────────────────────────────────
left_col, divider_col, right_col = st.columns([5, 0.1, 5])


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Policy Input & Summarization
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="panel-header">Policy Input & Summarization</div>', unsafe_allow_html=True)

    # Upload PDF
    uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"], label_visibility="visible")

    st.markdown("<div style='text-align:center; color:#9C7A28; font-size:0.82rem; margin:6px 0;'>OR</div>",
                unsafe_allow_html=True)

    # Paste Text
    pasted_text = st.text_area("Paste Policy Text", height=160,
                               placeholder="Paste the full policy text here...")

    word_input = count_words(pasted_text) if pasted_text else 0
    st.markdown(f'<div class="word-count">Word count: {word_input}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    generate_col, reset_col = st.columns([3, 1])

    with generate_col:
        gen_summary_btn = st.button("Generate Summary", key="gen_sum")

    with reset_col:
        if st.button("Reset", key="reset_left"):
            st.session_state["summary_raw"] = ""
            st.session_state["policy_text"] = ""
            st.rerun()

    # Generate summary
    if gen_summary_btn:
        # Determine input source
        policy_text = ""
        if uploaded_file:
            with st.spinner("Extracting PDF text..."):
                policy_text = extract_pdf_text(uploaded_file)
        elif pasted_text.strip():
            policy_text = pasted_text.strip()
        else:
            st.warning("Please upload a PDF or paste policy text first.")

        if policy_text:
            st.session_state["policy_text"] = policy_text
            progress_bar = st.progress(0, text="Initialising analysis...")
            for pct, msg in [(20, "Reading policy document..."),
                             (45, "Identifying key objectives..."),
                             (70, "Extracting key measures..."),
                             (90, "Finalising summary...")]:
                time.sleep(0.35)
                progress_bar.progress(pct, text=msg)
            try:
                raw = generate_summary(policy_text)
                st.session_state["summary_raw"] = raw
                progress_bar.progress(100, text="Summary complete.")
                time.sleep(0.4)
                progress_bar.empty()
            except Exception as e:
                progress_bar.empty()
                st.error(f"Error calling Groq API: {e}")

    # ── Summary Output ──
    if st.session_state["summary_raw"]:
        st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
        st.markdown("**Summary Output**")

        raw = st.session_state["summary_raw"]
        keys = [("MAIN GOALS:", "Main Goals"),
                ("KEY MEASURES:", "Key Measures"),
                ("OVERALL DIRECTION:", "Overall Direction")]
        sections = parse_sections(raw, [(k[0].replace(":", ""), k[1]) for k in keys])

        # Fallback: show raw if parsing fails
        if not sections:
            sections = {"Full Summary": raw}

        for heading, content in sections.items():
            st.markdown(f"""
            <div class="output-section">
                <h4>{heading}</h4>
                <p>{content}</p>
            </div>""", unsafe_allow_html=True)

        word_out = count_words(" ".join(sections.values()))
        st.markdown(f'<div class="word-count">Output word count: {word_out}</div>', unsafe_allow_html=True)

        # Download Summary PDF
        pdf_bytes = build_pdf("Policy Summary", sections, "summary")
        st.download_button(
            label="Download Summary as PDF",
            data=pdf_bytes,
            file_name="policy_summary.pdf",
            mime="application/pdf",
            key="dl_summary"
        )


# Thin gold divider column
with divider_col:
    st.markdown("""
    <div style="width:2px; background: linear-gradient(180deg, transparent, #C9A84C, transparent);
    height:100%; min-height:600px; margin:0 auto;"></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Scenario Simulation
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    st.markdown('<div class="panel-header">Scenario Simulation</div>', unsafe_allow_html=True)

    SCENARIOS = [
        "Select a scenario...",
        "Stricter Government Regulation",
        "Expansion into Low-Income Developing Markets",
        "Economic recession with rising unemployment",
        "Rapid technological disruption (AI & automation)",
        "Public health emergency / pandemic outbreak",
        "Climate change & environmental crisis",
        "Political instability and governance changes",
        "Global supply chain disruption",
        "Demographic shift (ageing population)",
        "Cyber security threat to infrastructure",
        "Social unrest and inequality surge",
        "Custom (type below)"
    ]

    selected_scenario = st.selectbox("Select Scenario", SCENARIOS)
    custom_scenario = st.text_area("Or Type Custom Scenario", height=100,
                                   placeholder="Describe a specific scenario to simulate...")

    word_scenario = count_words(custom_scenario) if custom_scenario else 0
    st.markdown(f'<div class="word-count">Word count: {word_scenario}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    gen_col, res_col = st.columns([3, 1])

    with gen_col:
        gen_scenario_btn = st.button("Generate Scenario", key="gen_scen")

    with res_col:
        if st.button("Reset", key="reset_right"):
            st.session_state["scenario_raw"] = ""
            st.rerun()

    if gen_scenario_btn:
        # Determine scenario text
        if custom_scenario.strip():
            scenario_text = custom_scenario.strip()
        elif selected_scenario != "Select a scenario..." and selected_scenario != "Custom (type below)":
            scenario_text = selected_scenario
        else:
            scenario_text = ""

        # Determine policy source
        policy_for_scenario = st.session_state.get("policy_text", "")
        if not policy_for_scenario and pasted_text.strip():
            policy_for_scenario = pasted_text.strip()

        if not scenario_text:
            st.warning("Please select or type a scenario.")
        elif not policy_for_scenario:
            st.warning("Please generate a summary first, or paste policy text in the left panel.")
        else:
            progress_bar2 = st.progress(0, text="Initialising simulation...")
            for pct, msg in [(20, "Loading policy context..."),
                             (45, "Simulating scenario impact..."),
                             (70, "Conducting strategic analysis..."),
                             (90, "Identifying risks and opportunities...")]:
                time.sleep(0.35)
                progress_bar2.progress(pct, text=msg)
            try:
                raw_s = generate_scenario(policy_for_scenario, scenario_text)
                st.session_state["scenario_raw"] = raw_s
                progress_bar2.progress(100, text="Simulation complete.")
                time.sleep(0.4)
                progress_bar2.empty()
            except Exception as e:
                progress_bar2.empty()
                st.error(f"Error calling Groq API: {e}")

    # ── Scenario Output ──
    if st.session_state["scenario_raw"]:
        st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
        st.markdown("**Scenario Output**")

        raw_s = st.session_state["scenario_raw"]
        scen_keys = [
            ("SCENARIO IMPACT", "Scenario Impact"),
            ("STRATEGIC ANALYSIS", "Strategic Analysis"),
            ("RISKS AND OPPORTUNITIES", "Risks and Opportunities")
        ]
        scen_sections = parse_sections(raw_s, scen_keys)

        if not scen_sections:
            scen_sections = {"Simulation Result": raw_s}

        for heading, content in scen_sections.items():
            st.markdown(f"""
            <div class="output-section">
                <h4>{heading}</h4>
                <p>{content}</p>
            </div>""", unsafe_allow_html=True)

        word_scen_out = count_words(" ".join(scen_sections.values()))
        st.markdown(f'<div class="word-count">Output word count: {word_scen_out}</div>', unsafe_allow_html=True)

        # Download Scenario PDF
        pdf_scen_bytes = build_pdf("Scenario Simulation", scen_sections, "scenario")
        st.download_button(
            label="Export Scenario as PDF",
            data=pdf_scen_bytes,
            file_name="scenario_simulation.pdf",
            mime="application/pdf",
            key="dl_scenario"
        )


# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:14px; border-top: 2px solid #C9A84C;
color:#9C7A28; font-size:0.78rem; margin-top:20px;">
    Automated Policy Summarization &amp; Scenario Simulation System
</div>
""", unsafe_allow_html=True)

