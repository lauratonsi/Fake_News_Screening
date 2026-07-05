"""Streamlit demo of the fake-news screening system.

Run locally:
    streamlit run app.py

The heavy lifting lives in ``src/predict.py`` — this file is UI only.
"""
import streamlit as st

st.set_page_config(page_title="Fake News Screening", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_system():
    from src.predict import ScreeningSystem

    return ScreeningSystem()


st.title("🛡️ Fake News Screening")
st.markdown(
    "**Hybrid Disinformation Screening System (HDSS)** — a calibrated SVM, a "
    "Bi-GRU and a Bi-LSTM vote on the input; a retrieval layer against the "
    "training corpora adds evidence."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. **Ensemble** — three models trained on ISOT + WELFake + COVID-19
   score the text; the average is the fake probability.
2. **Reference corpus** — the input is compared with snippets of known
    real/fake articles. A strong match overrides the ensemble, a weak
    one only shifts the score, and the closest retrieved snippets are shown.
3. **Agreement check** — if the models disagree strongly, the verdict
   is flagged for human review.
        """
    )
    st.header("Limitations")
    st.markdown(
        """
- English-language news only.
- Training data covers 2015–2021: recent events are out of domain.
- The reference lookup is **similarity matching, not fact-checking** —
    it cannot verify claims it has never seen, but it now returns the retrieved
    evidence.
- Out-of-domain accuracy is substantially lower than in-domain
  (see the README benchmark) — treat verdicts as a screening aid,
  not a truth oracle.
        """
    )

st.warning("The system is trained on English news datasets — please paste English text.")

examples = {
    "— pick an example or paste your own text —": "",
    "Hoax (sensational)": "WOW! Hillary Clinton caught on secret video admitting the election was a total fraud. MUST WATCH!",
    "Plausible real news": "The Federal Reserve announced on Wednesday a new set of regulations to monitor inflation and support the labor market.",
    "Health misinformation": "A recent peer-reviewed study from Oxford University confirms that daily consumption of vitamin C completely prevents all viral infections.",
}
choice = st.selectbox("Examples", list(examples.keys()))
text = st.text_area("News text", value=examples[choice], height=150)

if st.button("Analyze", type="primary") and text.strip():
    system = load_system()
    with st.spinner("Scoring..."):
        result = system.predict(text)

    st.divider()
    color = "red" if result["verdict"] == "FAKE" else "green"
    st.markdown(
        f"<h2 style='text-align:center;color:{color};'>Verdict: {result['verdict']}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h4 style='text-align:center;'>Fake probability: "
        f"{result['fake_probability']:.1%} &nbsp;|&nbsp; {result['reason']}</h4>",
        unsafe_allow_html=True,
    )
    if result["needs_review"]:
        st.error(
            "⚠️ The models disagree strongly on this text — the verdict is "
            "low-confidence and would be routed to a human reviewer."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model scores")
        for name, score in result["model_scores"].items():
            st.progress(min(max(score, 0.0), 1.0), text=f"**{name.upper()}** — {score:.1%} fake probability")
    with col2:
        st.subheader("Reference corpus (heuristic)")
        message = result["reference"]["message"]
        if result["reference"]["verdict"] == "FAKE":
            st.error(f"⛔ {message}")
        elif result["reference"]["verdict"] == "REAL":
            st.success(f"✅ {message}")
        else:
            st.info(f"ℹ️ {message}")
        st.caption(
            "Retrieval against articles already known to be real or fake. "
            "This is a support signal, not fact-checking."
        )

    evidence = result["reference"].get("evidence", [])
    if evidence:
        with st.expander("Retrieved evidence", expanded=False):
            for hit in evidence[:4]:
                label = "REAL" if hit["label"] == "REAL" else "FAKE"
                st.markdown(f"**{label}** — similarity {hit['score']:.1%}")
                st.write(hit["text"])
                st.divider()
