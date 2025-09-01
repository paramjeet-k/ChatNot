# app.py
import math
import re
from typing import Dict, List, Tuple
import streamlit as st

# ---------------------------
# Utility helpers
# ---------------------------
def parse_kv_numbers(text: str, keys: List[str]) -> Dict[str, float]:
    """
    Parses k=v pairs (numbers) from free text, case-insensitive.
    Example: 'h=10 b=3 s=3 j=0.5 t=2 d=165 rho=1000'
    """
    out = {}
    for k in keys:
        m = re.search(rf"\b{k}\s*=\s*([-+]?\d*\.?\d+)", text, flags=re.I)
        if m:
            out[k] = float(m.group(1))
    return out

def mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0

def ft_to_m(x_ft: float) -> float:
    return x_ft * 0.3048

def lb_to_kg(x_lb: float) -> float:
    return x_lb * 0.45359237

def circle_area(diam_m: float) -> float:
    r = diam_m / 2.0
    return math.pi * r * r

# ---------------------------
# Calculation engines
# ---------------------------
def calc_powder_factor(H, B, S, J, T, D_mm, rho):
    """Returns (pf_kg_per_m3, charge_per_hole_kg, charged_length_m, rock_volume_m3)"""
    d_m = mm_to_m(D_mm)
    area = circle_area(d_m)
    charged_len = max(H + J - T, 0.0)
    charge_per_hole = rho * area * charged_len
    rock_volume = B * S * H
    pf = charge_per_hole / rock_volume if rock_volume > 0 else 0.0
    return pf, charge_per_hole, charged_len, rock_volume

def calc_scaled_distance(distance, distance_units, charge_kg):
    """Scaled Distance (metric): SD = distance(m) / sqrt(charge_kg_per_delay)"""
    if distance_units.lower() in ["ft", "feet"]:
        distance_m = ft_to_m(distance)
    else:
        distance_m = distance
    if charge_kg <= 0:
        return None
    return distance_m / math.sqrt(charge_kg)

def lk_burden_spacing(D_mm, k, alpha, F):
    """
    A configurable Langefors–Kihlström-style rule of thumb.
    We keep it parametric to avoid hard-coding site-specific constants.

    B = k * (D_mm/1000) * sqrt(F)
    S = alpha * B

    Where:
    - k: burden coefficient (typ. 22–35 for bench work depending on rock/energy)
    - alpha: spacing-to-burden ratio (typ. 1.1–1.4)
    - F: strength/energy factor (dimensionless). Keep ~1.0 unless you calibrate.
    """
    B = k * mm_to_m(D_mm) * math.sqrt(max(F, 0))
    S = alpha * B
    return B, S

def nobel_cartridge_method(charged_len_m, cart_len_m, cart_diam_mm, rho_cart):
    """
    Simple 'Nobel cartridge' style estimate:
    - Approximate number of cartridges = charged_len / cart_len (rounded up)
    - Each cartridge mass = rho_cart * (pi*(d/2)^2)*cart_len
    Returns: (num_carts, mass_per_cart_kg, total_mass_kg)
    """
    d_m = mm_to_m(cart_diam_mm)
    vol_per_cart = circle_area(d_m) * cart_len_m
    mass_per_cart = rho_cart * vol_per_cart
    num_carts = max(int(math.ceil(charged_len_m / cart_len_m)), 0)
    total_mass = num_carts * mass_per_cart
    return num_carts, mass_per_cart, total_mass

# ---------------------------
# Knowledge base (expandable)
# ---------------------------
KB = [
    {
        "q": "Powder factor",
        "a": "Powder Factor (PF) = charge mass / rock volume (kg/m³ metric). For a bench hole: charge ≈ ρ × (π(D/2)²) × charged_length, and volume ≈ B × S × H."
    },
    {
        "q": "Langefors–Kihlström method",
        "a": "Langefors–Kihlström provides empirical relationships linking burden/spacing to hole diameter, explosive/rock factors, and desired results. A practical parametric form: B = k·(D/1000)·√F and S = α·B, with k (≈22–35), α (≈1.1–1.4), and F as a site-tuned factor."
    },
    {
        "q": "Nobel cartridge method",
        "a": "A cartridge-based charging approach: estimate number of cartridges from charged length and cartridge length, then compute mass using cartridge diameter and density. Useful where packaged explosives are used."
    },
    {
        "q": "Burden and spacing",
        "a": "Common starting point for surface benches: B ≈ 25–35 × D(mm)/1000 (in meters), S ≈ 1.15–1.4 × B. Tune for geology, stiffness, face conditions, and explosive energy."
    },
    {
        "q": "Stemming",
        "a": "Stemming confines gases. Quick rules: T ≈ 0.7–1.0 × B or ≈20–30 × D(mm). Increase to reduce flyrock/airblast; ensure collar quality."
    },
    {
        "q": "Charge per hole",
        "a": "Charge per hole (kg) = ρ (kg/m³) × π(D/2)² (m²) × charged length (m). Charged length ≈ H + J − T."
    },
    {
        "q": "Scaled distance and vibration",
        "a": "Scaled Distance (metric) = distance (m) / √(charge per delay in kg). Lower SD → higher vibration (PPV). Always calibrate with site data."
    },
    {
        "q": "Flyrock reduction",
        "a": "Avoid overcharge, increase stemming/burden within limits, ensure accurate drilling (collar/angle/position), avoid voids, and keep per-delay charge consistent."
    },
    {
        "q": "Water in holes",
        "a": "Use water-resistant emulsions or heavy-ANFO in wet/dynamic-water holes. Consider density, energy, gas generation, and manufacturer guidance."
    },
    {
        "q": "Safety and misfires",
        "a": "Follow SOPs and regulatory code: secure area, mark misfire, do not drill/dig, notify blaster-in-charge, and neutralize under approved procedures only."
    },
]

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

# Light BM25-like retrieval (no heavy deps)
VOCAB = {}
DOCS = []
for item in KB:
    tokens = tokenize(item["q"] + " " + item["a"])
    DOCS.append(tokens)
    for t in set(tokens):
        VOCAB[t] = VOCAB.get(t, 0) + 1

def bm25_like(query: str, k1=1.5, b=0.75) -> Tuple[int, float]:
    N = len(DOCS)
    if N == 0:
        return 0, 0.0
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
            scores[idx] += idf * (f * (k1 + 1)) / (denom if denom else 1.0)
    top_idx = max(range(N), key=lambda i: scores[i])
    return top_idx, scores[top_idx]

def kb_answer(query: str) -> str:
    idx, _ = bm25_like(query)
    item = KB[idx]
    return f"**{item['q']}**\n{item['a']}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Drilling & Blasting Chatbot", page_icon="💥", layout="wide")
st.title("💥 Drilling & Blasting Chatbot")
st.caption("Educational planner. Always follow site SOPs, regulations, and a licensed blaster's direction.")

with st.sidebar:
    st.header("Global defaults")
    if "defaults" not in st.session_state:
        st.session_state.defaults = {
            "H": 10.0,     # bench height (m)
            "J": 0.5,      # subdrill (m)
            "T": 2.0,      # stemming (m)
            "B": 3.0,      # burden (m) default
            "S": 3.5,      # spacing (m) default
            "D": 165.0,    # hole diameter (mm)
            "rho": 1000.0, # explosive density (kg/m³)
            # L-K method defaults (all adjustable)
            "lk_k": 30.0,      # burden coefficient
            "lk_alpha": 1.25,  # spacing/burden ratio
            "lk_F": 1.00,      # energy/rock factor
            # Nobel cartridge defaults
            "cart_len": 0.40,  # m
            "cart_diam": 83.0, # mm
            "cart_rho": 1100.0 # kg/m³
        }
    d = st.session_state.defaults
    d["H"]   = st.number_input("Bench height H (m)", 1.0, 100.0, d["H"], 0.1)
    d["J"]   = st.number_input("Subdrill J (m)", 0.0, 5.0, d["J"], 0.1)
    d["T"]   = st.number_input("Stemming T (m)", 0.0, 10.0, d["T"], 0.1)
    d["B"]   = st.number_input("Burden B (m) [default]", 0.5, 12.0, d["B"], 0.1)
    d["S"]   = st.number_input("Spacing S (m) [default]", 0.5, 15.0, d["S"], 0.1)
    d["D"]   = st.number_input("Hole diameter D (mm)", 50.0, 310.0, d["D"], 1.0)
    d["rho"] = st.number_input("Explosive density ρ (kg/m³)", 700.0, 1400.0, d["rho"], 10.0)

    st.markdown("---")
    st.subheader("Langefors–Kihlström (tunable)")
    d["lk_k"]     = st.number_input("k (burden coefficient)", 10.0, 60.0, d["lk_k"], 0.5)
    d["lk_alpha"] = st.number_input("α (spacing/burden)", 1.0, 2.0, d["lk_alpha"], 0.05)
    d["lk_F"]     = st.number_input("F (factor √F used)", 0.1, 3.0, d["lk_F"], 0.05)

    st.markdown("---")
    st.subheader("Nobel cartridge (tunable)")
    d["cart_len"]  = st.number_input("Cartridge length L (m)", 0.05, 1.00, d["cart_len"], 0.01)
    d["cart_diam"] = st.number_input("Cartridge diameter (mm)", 20.0, 150.0, d["cart_diam"], 1.0)
    d["cart_rho"]  = st.number_input("Cartridge density (kg/m³)", 800.0, 1600.0, d["cart_rho"], 10.0)

st.markdown("""
**What I can do**
- Answer D&B questions (burden/spacing, stemming, water, misfires, vibration).
- Calculators: **Powder Factor**, **charge per hole**, **Scaled Distance**, **L-K burden/spacing**, **Nobel cartridge** estimate.
- All coefficients are adjustable in the sidebar (so this fits your site’s calibration).
""")

# Session chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything about drilling & blasting. Examples:\n"
                       "- `pf h=10 b=3 s=3 j=0.5 t=2 d=165 rho=1000`\n"
                       "- `lk d=165 k=30 alpha=1.25 f=1.0`\n"
                       "- `nobel d=165 L=0.40 cart_d=83 rho=1100 h=10 j=0.5 t=2`\n"
                       "- `sd 300 m, 35 kg`"
        }
    ]

# Render transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def respond(user_text: str) -> str:
    txt = user_text.strip()

    # ---- Powder Factor ------------------------------------------------------
    if re.search(r"\bpf\b|\bpowder\s*factor\b", txt, flags=re.I):
        vals = parse_kv_numbers(txt, ["h","b","s","j","t","d","rho"])
        H   = vals.get("h", st.session_state.defaults["H"])
        B   = vals.get("b", st.session_state.defaults["B"])
        S   = vals.get("s", st.session_state.defaults["S"])
        J   = vals.get("j", st.session_state.defaults["J"])
        T   = vals.get("t", st.session_state.defaults["T"])
        Dmm = vals.get("d", st.session_state.defaults["D"])
        rho = vals.get("rho", st.session_state.defaults["rho"])
        pf, qhole, Lc, V = calc_powder_factor(H,B,S,J,T,Dmm,rho)
        return (f"🔢 **Powder Factor**\n"
                f"- Inputs: H={H:.2f} m, B={B:.2f} m, S={S:.2f} m, J={J:.2f} m, T={T:.2f} m, D={Dmm:.0f} mm, ρ={rho:.0f} kg/m³\n"
                f"- Charged length Lc = H+J−T = **{Lc:.2f} m**\n"
                f"- Charge per hole = **{qhole:.1f} kg**\n"
                f"- Rock volume per hole = **{V:.2f} m³**\n"
                f"- **PF = {pf:.3f} kg/m³**")

    # ---- Scaled Distance -----------------------------------------------------
    if re.search(r"\bsd\b|scaled\s*distance|vibration", txt, flags=re.I):
        # accept patterns like "sd 300 m, 35 kg"
        md = re.search(r"(\d+(\.\d+)?)\s*(m|meter|metre|ft)", txt, flags=re.I)
        mq = re.search(r"(\d+(\.\d+)?)\s*(kg|lb)", txt, flags=re.I)
        if md and mq:
            dist = float(md.group(1)); du = md.group(3)
            q = float(mq.group(1)); qu = mq.group(3)
            if qu.lower() == "lb":
                q = lb_to_kg(q)
            sd = calc_scaled_distance(dist, du, q)
            if sd is None:
                return "Provide a positive charge per delay."
            return f"📉 **Scaled Distance** = distance/√charge = **{sd:.2f} m/√kg** (distance={dist} {du}, charge={q:.2f} kg)"
        else:
            return "To compute SD, include a distance (m/ft) and charge per delay (kg/lb). Example: `sd 300 m, 35 kg`."

    # ---- Langefors–Kihlström (parametric) -----------------------------------
    if re.search(r"\blk\b|kihl|langefors", txt, flags=re.I):
        vals = parse_kv_numbers(txt, ["d","k","alpha","f"])
        Dmm   = vals.get("d", st.session_state.defaults["D"])
        k     = vals.get("k", st.session_state.defaults["lk_k"])
        alpha = vals.get("alpha", st.session_state.defaults["lk_alpha"])
        F     = vals.get("f", st.session_state.defaults["lk_F"])
        B, S = lk_burden_spacing(Dmm, k, alpha, F)
        return (f"📐 **Langefors–Kihlström (parametric)**\n"
                f"- D={Dmm:.0f} mm, k={k:.2f}, α={alpha:.2f}, F={F:.2f}\n"
                f"- **Burden B ≈ {B:.2f} m**, **Spacing S ≈ {S:.2f} m**\n"
                f"_Tune k/α/F to your site calibration; this keeps the method flexible._")

    # ---- Nobel cartridge method ---------------------------------------------
    if re.search(r"\bnobel\b|cartridge", txt, flags=re.I):
        vals = parse_kv_numbers(txt, ["h","j","t","L","cart_d","rho","d"])
        H   = vals.get("h", st.session_state.defaults["H"])
        J   = vals.get("j", st.session_state.defaults["J"])
        T   = vals.get("t", st.session_state.defaults["T"])
        L   = vals.get("L", st.session_state.defaults["cart_len"])
        cart_d = vals.get("cart_d", st.session_state.defaults["cart_diam"])
        rho_c = vals.get("rho", st.session_state.defaults["cart_rho"])
        # charged length from bench
        charged_len = max(H + J - T, 0.0)
        n, m_cart, m_total = nobel_cartridge_method(charged_len, L, cart_d, rho_c)
        return (f"🧯 **Nobel / cartridge-based estimate**\n"
                f"- Charged length Lc ≈ **{charged_len:.2f} m**; cartridge L={L:.2f} m, Ø={cart_d:.0f} mm, ρ={rho_c:.0f} kg/m³\n"
                f"- Mass per cartridge ≈ **{m_cart:.2f} kg**\n"
                f"- Estimated number of cartridges ≈ **{n}**\n"
                f"- **Total charge ≈ {m_total:.1f} kg**\n"
                f"_Adjust cartridge size/density to match product datasheet._")

    # ---- Quick burden/spacing rule-of-thumb ---------------------------------
    if re.search(r"burden|spacing", txt, flags=re.I) and re.search(r"rule|start|estimate", txt, flags=re.I):
        vals = parse_kv_numbers(txt, ["d"])
        Dmm = vals.get("d", st.session_state.defaults["D"])
        B = 30 * mm_to_m(Dmm)
        S = 1.25 * B
        T = 25 * mm_to_m(Dmm)
        return (f"📏 **Starter rules**\n"
                f"- Burden B ≈ **{B:.2f} m**\n- Spacing S ≈ **{S:.2f} m**\n- Stemming T ≈ **{T:.2f} m**\n"
                f"(Assumed D={Dmm:.0f} mm; tune for rock/energy/SOP.)")

    # ---- Otherwise: knowledge base answer -----------------------------------
    return kb_answer(txt)

# Chat input & output
user_msg = st.chat_input("Type your question or a calc (pf / lk / nobel / sd)…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    bot = respond(user_msg)
    st.session_state.messages.append({"role": "assistant", "content": bot})
    with st.chat_message("assistant"):
        st.markdown(bot)

st.markdown("---")
st.caption("⚠️ For planning education only. Validate with site trials and comply with code/SOP under a licensed blaster.")
