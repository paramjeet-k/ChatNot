# app.py
import math
import re
from typing import List, Dict, Tuple
import streamlit as st

# ---------------------------
# Utilities
# ---------------------------

def mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0

def inches_to_m(x_in: float) -> float:
    return x_in * 0.0254

def ft_to_m(x_ft: float) -> float:
    return x_ft * 0.3048

def kg_per_m3_to_g_per_cc(x: float) -> float:
    return x / 1000.0

def area_of_hole(diam_m: float) -> float:
    r = diam_m / 2.0
    return math.pi * r * r

def parse_float(s, default=None):
    try:
        return float(s)
    except:
        return default

# ---------------------------
# Minimal knowledge base (curated facts)
# ---------------------------
KB = [
    {
        "q": "What is powder factor?",
        "a": "Powder Factor (PF) is the mass of explosive used divided by the rock volume broken. Units often kg/m³ (metric) or lb/yd³ (imperial). Typical surface bench blasting values range ~0.3–1.0 kg/m³ depending on rock strength, fragmentation target, and energy of the explosive."
    },
    {
        "q": "How do I estimate burden and spacing?",
        "a": "A common starting point for bench blasting (ANFO/ANFO blends) is: Burden B ≈ (25–35) × hole diameter (in mm) / 1000 (m), or B ≈ 25–40×D (in) in inches. Spacing S ≈ 1.15–1.4 × B. Adjust for rock mass quality, stiffness, and energy."
    },
    {
        "q": "What is stemming and how much should I use?",
        "a": "Stemming is inert material at the top of the hole to confine gases. A quick rule: stemming length T ≈ 0.7–1.0 × burden (bench blasting), or T ≈ 20–30 × hole diameter (in mm), whichever suits fragmentation and flyrock control."
    },
    {
        "q": "How do I compute charge per hole?",
        "a": "Charge per hole (kg) = explosive density (kg/m³) × hole cross-section area (m²) × charged length (m). Charged length is typically (bench height + subdrill – stemming)."
    },
    {
        "q": "What is scaled distance and why is it used?",
        "a": "Scaled Distance (SD) = distance (m) / sqrt(charge per delay, kg). It correlates with ground vibration. Lower SD implies higher vibration. Site-specific constants are required for accurate PPV prediction."
    },
    {
        "q": "How do initiation and delays affect results?",
        "a": "Using short-delay intervals between holes/rows reduces instantaneous charge per delay, improves muckpile throw and fragmentation, and helps control vibration. Keep actual per-delay charge consistent with design assumptions."
    },
    {
        "q": "How to reduce flyrock?",
        "a": "Avoid overcharging, increase stemming or burden (within limits), improve hole collar quality, check for decking voids, and ensure accurate drilling to design angles and positions."
    },
    {
        "q": "What to do in case of a misfire?",
        "a": "Follow site SOPs: secure the area, notify supervisor/blaster-in-charge, mark and record the hole, forbid drilling or digging near the misfire, and only re-initiate or make safe under approved procedures with proper clearance."
    },
    {
        "q": "How do water conditions affect explosive choice?",
        "a": "In dry holes, ANFO is economical. In wet or dynamic water, use water-resistant emulsions or heavy ANFO blends. Consider gas generation, density, and energy with supplier tech sheets."
    },
    {
        "q": "What inputs do I need to design a bench blast?",
        "a": "Rock properties (UCS/RQD/JSA), bench height, hole diameter, explosive density & energy, desired fragmentation, face conditions, equipment dig/haul constraints, environmental limits (vibration, airblast), and safety/legal standards."
    },
]

# Simple bag-of-words retrieval (tiny BM25-like)
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

VOCAB = {}
DOCS = []
for item in KB:
    text = (item["q"] + " " + item["a"]).lower()
    tokens = tokenize(text)
    DOCS.append(tokens)
    for t in set(tokens):
        VOCAB[t] = VOCAB.get(t, 0) + 1

def bm25_like(query: str, k1=1.5, b=0.75) -> List[Tuple[int, float]]:
    N = len(DOCS)
    avgdl = sum(len(d) for d in DOCS) / N
    q_terms = tokenize(query)
    scores = [0.0] * N
    for qi in set(q_terms):
        n_qi = VOCAB.get(qi, 0)
        if n_qi == 0:
            continue
        idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        for idx, doc in enumerate(DOCS):
            f = doc.count(qi)
            dl = len(doc)
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score = idf * (f * (k1 + 1)) / (denom if denom != 0 else 1)
            scores[idx] += score
    return sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

# ---------------------------
# Chatbot brain (router)
# ---------------------------

def handle_calculations(message: str, defaults: Dict) -> Tuple[bool, str]:
    msg = message.lower()

    # Extract common numbers if the user writes like "H=10m, D=165mm, rho=1000"
    nums = dict(re.findall(r"(b|s|h|j|t|d|rho|pf)\s*=\s*([0-9]*\.?[0-9]+)", msg))

    # Calculator: Powder Factor (also reports charge per hole)
    if "powder factor" in msg or re.search(r"\bpf\b", msg):
        H = parse_float(nums.get("h"), defaults["H"])
        B = parse_float(nums.get("b"), defaults["B"])
        S = parse_float(nums.get("s"), defaults["S"])
        J = parse_float(nums.get("j"), defaults["J"])
        T = parse_float(nums.get("t"), defaults["T"])
        D = parse_float(nums.get("d"), defaults["D"])
        rho = parse_float(nums.get("rho"), defaults["rho"])

        diam_m = D / 1000.0  # mm -> m
        area = area_of_hole(diam_m)
        charged_len = max(H + J - T, 0.0)
        charge_per_hole = rho * area * charged_len  # kg
        rock_volume = B * S * H  # m3
        pf = (charge_per_hole / rock_volume) if rock_volume > 0 else 0.0

        return True, (
            f"🔢 **Powder Factor Calculator**\n\n"
            f"- Hole diameter D: **{D:.1f} mm**  \n"
            f"- Bench height H: **{H:.2f} m**, Subdrill J: **{J:.2f} m**, Stemming T: **{T:.2f} m**  \n"
            f"- Burden B: **{B:.2f} m**, Spacing S: **{S:.2f} m**  \n"
            f"- Explosive density ρ: **{rho:.0f} kg/m³**  \n\n"
            f"**Charged length** = H + J − T = **{charged_len:.2f} m**  \n"
            f"**Charge per hole** = ρ × area × charged_len = **{charge_per_hole:.1f} kg**  \n"
            f"**Rock volume per hole** = B × S × H = **{rock_volume:.2f} m³**  \n"
            f"**Powder Factor (PF)** = charge/volume = **{pf:.3f} kg/m³**"
        )

    # Calculator: Scaled Distance (vibration proxy)
    if "scaled distance" in msg or re.search(r"\bsd\b", msg) or "vibration" in msg:
        m_dist = re.search(r"(\d+\.?\d*)\s*(m|meter|metre|ft)\b", msg)
        m_charge = re.search(r"(\d+\.?\d*)\s*(kg|lb)\b", msg)
        if m_dist and m_charge:
            dist_val = float(m_dist.group(1))
            if m_dist.group(2) == "ft":
                dist_m = ft_to_m(dist_val)
            else:
                dist_m = dist_val
            q = float(m_charge.group(1))
            if m_charge.group(2) == "lb":
                q *= 0.453592
            if q <= 0:
                return True, "Please provide a positive charge mass per delay."
            sd = dist_m / math.sqrt(q)
            return True, f"📉 **Scaled Distance (metric)** = distance / √charge_per_delay = **{sd:.2f} m/√kg**"
        else:
            return True, "To compute Scaled Distance, include a distance (m or ft) and a charge per delay (kg or lb) in your message."

    # Quick rules for B, S, T
    if "burden" in msg and "spacing" in msg and any(k in msg for k in ["estimate", "rule", "start"]):
        D = parse_float(nums.get("d"), defaults["D"])
        B = 30 * (D / 1000.0)  # ≈ 30 × D(mm) → m
        S = 1.25 * B
        T = 25 * (D / 1000.0)  # ≈ 25 × D(mm) → m
        return True, (
            f"📐 **Starter rules (bench blasting)**  \n"
            f"- Burden B ≈ **{B:.2f} m**  \n"
            f"- Spacing S ≈ **{S:.2f} m**  \n"
            f"- Stemming T ≈ **{T:.2f} m**  \n"
            f"(Assumed D={D:.0f} mm; tweak for your rock/energy constraints.)"
        )

    return False, ""

def retrieve_answer(message: str) -> str:
    ranks = bm25_like(message)
    top_idx, _ = ranks[0]
    item = KB[top_idx]
    suggestion = ""
    if any(k in message.lower() for k in ["calculate", "how much", "compute", "powder factor", " pf", "pf "]):
        suggestion = "\n\nTry: 'PF h=10 b=3 s=3 j=0.5 t=2 d=165 rho=1000'"
    return f"**{item['q']}**\n{item['a']}{suggestion}"

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Mining Drilling & Blasting Chatbot", page_icon="💥", layout="wide")

st.title("💥 Drilling & Blasting Chatbot (Streamlit)")
st.caption("For educational and planning support only — follow site SOPs, regulations, and a licensed blaster's direction.")

with st.sidebar:
    st.header("Settings")
    units = st.selectbox("Units", ["Metric"], index=0, help="This prototype uses metric internally.")
    st.markdown("---")
    st.subheader("Default Design Inputs")

    # Defaults (stored in session)
    if "defaults" not in st.session_state:
        st.session_state.defaults = {
            "H": 10.0,    # bench height (m)
            "J": 0.5,     # subdrill (m)
            "T": 2.0,     # stemming (m)
            "B": 3.0,     # burden (m)
            "S": 3.5,     # spacing (m)
            "D": 165.0,   # diameter (mm)
            "rho": 1000.0 # explosive density (kg/m3)
        }

    d = st.session_state.defaults
    d["H"]   = st.number_input("Bench height H (m)", 1.0, 100.0, d["H"], 0.1)
    d["J"]   = st.number_input("Subdrill J (m)", 0.0, 5.0, d["J"], 0.1)
    d["T"]   = st.number_input("Stemming T (m)", 0.0, 10.0, d["T"], 0.1)
    d["B"]   = st.number_input("Burden B (m)", 0.5, 10.0, d["B"], 0.1)
    d["S"]   = st.number_input("Spacing S (m)", 0.5, 12.0, d["S"], 0.1)
    d["D"]   = st.number_input("Hole diameter D (mm)", 50.0, 311.0, d["D"], 1.0)
    d["rho"] = st.number_input("Explosive density ρ (kg/m³)", 700.0, 1400.0, d["rho"], 10.0)

    st.info("Tip: Ask me to 'estimate burden & spacing rules' or 'calculate powder factor'.", icon="💡")

st.markdown("""
**What I can do**
- Answer drilling & blasting questions (burden/spacing, stemming, misfires, water, initiation).
- Do quick calcs: Powder Factor (PF), charge per hole, scaled distance (basic).
- Give starter rules for bench blast design.
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about drilling & blasting. For PF, try: 'PF h=10 b=3 s=3 j=0.5 t=2 d=165 rho=1000'."}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
prompt = st.chat_input("Type your drilling & blasting question…")

def reply(user_text: str):
    handled, answer = handle_calculations(user_text, st.session_state.defaults)
    if handled:
        return answer
    return retrieve_answer(user_text)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = reply(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

st.markdown("---")
st.caption("⚠️ This tool is for learning and preliminary planning. Always comply with applicable laws, standards, and your site's blast management plan under a licensed blaster.")
