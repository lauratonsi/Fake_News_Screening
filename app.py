"""Streamlit demo of the fake-news screening system.

Run locally:
    streamlit run app.py

The heavy lifting lives in ``src/predict.py`` — this file is UI only.
"""
import os

import streamlit as st

st.set_page_config(page_title="Fake News Screening", page_icon="🛡️", layout="wide")


def _bridge_secrets_to_env():
    """Make ``st.secrets`` values visible to ``os.getenv``.

    Streamlit Community Cloud already exposes root-level secrets as environment
    variables, but a local ``.streamlit/secrets.toml`` only populates
    ``st.secrets``. The retrieval code reads ``os.getenv(...)``, so mirror the
    keys we use into the environment to behave identically in both places.
    """
    for key in ("GOOGLE_FACTCHECK_API_KEY",):
        if os.getenv(key):
            continue
        try:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass  # no secrets file locally is fine


@st.cache_resource
def load_system():
    _bridge_secrets_to_env()
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
2. **Most similar known articles** — the input is compared, by semantic
    embedding similarity, with snippets of known real/fake articles (so
    reworded claims still match). Matching a known *fake* claim is evidence
    of fakeness and boosts the score; matching real reporting is shown as
    evidence only, since sharing a topic with true news is not proof — it
    sways the verdict solely on a near-verbatim match.
3. **Agreement check** — if the models disagree strongly, the verdict
   is flagged for human review.
        """
    )
    st.header("Limitations")
    st.markdown(
        """
- English-language news only.
- Training data covers 2015–2021: recent events are out of domain.
- The reference lookup is **semantic similarity matching, not fact-checking** —
    it cannot verify claims it has never seen (in any wording), but it now
    returns the retrieved evidence.
- Out-of-domain accuracy is substantially lower than in-domain
  (see the README benchmark) — treat verdicts as a screening aid,
  not a truth oracle.
        """
    )

st.warning("The system is trained on English news datasets — please paste English text.")

examples = {
    "— pick an example or paste your own text —": "",
    "Hoax (sensational style)": "WOW! Hillary Clinton caught on secret video admitting the election was a total fraud. MUST WATCH!",
    "Hoax (calm, no sensational style)": "A leaked memo reveals that all major banks secretly agreed to delete private debts on a specific date.",
    "Health misinformation": "A recent peer-reviewed study from Oxford University confirms that daily consumption of vitamin C completely prevents all viral infections.",
    "COVID conspiracy claim": "The COVID-19 vaccine changes your DNA and will be inherited by your children.",
    "Same COVID claim, reworded": "Getting the COVID shot permanently alters your genetic code.",
    "Plausible real news": "The Federal Reserve announced on Wednesday a new set of regulations to monitor inflation and support the labor market.",
    "True political fact (flagged for review)": "Barack Obama served two terms as President from 2009 to 2017.",
    "True political fact (a known hard case)": "Donald Trump won the 2016 presidential election defeating Hillary Clinton.",
}
choice = st.selectbox("Examples", list(examples.keys()))
st.caption(
    "The examples span hoaxes in different writing styles, a claim reworded "
    "to test semantic retrieval, and two true statements the system "
    "handles differently — including one it still gets wrong. That's "
    "intentional: see the README's out-of-domain benchmark for why."
)
text = st.text_area("News text", value=examples[choice], height=150)

if st.button("Analyze", type="primary") and text.strip():
    system = load_system()
    try:
        with st.spinner("Scoring..."):
            result = system.predict(text)
    except Exception:
        import traceback

        print(traceback.format_exc())  # visible in Streamlit Cloud logs
        st.error(
            "⚠️ Something went wrong while scoring this text — a live "
            "retrieval source may be temporarily unavailable. Please try "
            "again in a few seconds."
        )
        st.stop()

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
        st.subheader("Most similar known articles")
        message = result["reference"]["message"]
        ref_verdict = result["reference"]["verdict"]
        if ref_verdict == "FAKE":
            st.error(f"⛔ {message}")
        elif ref_verdict == "REAL":
            st.success(f"✅ {message}")
        else:
            st.info(f"ℹ️ {message}")
        for hit in result["reference"]["evidence"][:3]:
            tag = "🟥 FAKE" if hit["label"] == "FAKE" else "🟩 REAL"
            snippet = hit["text"][:120].strip() or "(empty snippet)"
            st.markdown(f"- **{tag}** ({hit['score']:.0%}) — {snippet}")
        st.caption(
            "The closest snippets the system has already seen, with the label "
            "they carried. This is semantic *similarity*, not fact-checking — "
            "sharing a topic with real reporting is not proof a claim is true."
        )

    claim_analysis = result.get("claim_analysis", {})
    claims = claim_analysis.get("claims", [])
    if claims:
        st.subheader("Claim-level retrieval")
        st.caption(
            "Per claim, the closest known snippets and whether they were "
            "labelled real or fake. These are **evidence** labels, not a truth "
            "check — only a live fact-check (below) is an actual verdict."
        )
        summary = claim_analysis.get("summary", {})
        cols = st.columns(4)
        cols[0].metric("Claims", summary.get("claims_total", 0))
        cols[1].metric("Match known false", summary.get("matches_fake", 0))
        cols[2].metric("Match known real", summary.get("matches_real", 0))
        cols[3].metric("No close match", summary.get("no_match", 0))

        with st.expander("Claim-by-claim evidence", expanded=False):
            for item in claims:
                status = item["status"]
                if status == "MATCHES_KNOWN_FALSE":
                    st.error(f"⛔ Matches a known FALSE claim — {item['claim']}")
                elif status == "MATCHES_KNOWN_REAL":
                    st.success(f"✅ Matches known REAL reporting — {item['claim']}")
                else:
                    st.info(f"ℹ️ No close match — {item['claim']}")
                st.caption(f"{item['message']} | score {item['score']:.1%}")
                for hit in item.get("evidence", [])[:2]:
                    st.markdown(f"- **{hit['label']}** ({hit['score']:.1%}): {hit['text']}")
                st.divider()

        st.subheader("Live retrieval")
        st.caption(
            f"Source: {claim_analysis.get('source') or 'local-only'} — "
            "Google Fact Check (verdict, if a key is set), else Wikipedia "
            "(context, no key). A live fact-check verdict takes precedence; "
            "Wikipedia is context, not verification."
        )
        with st.expander("Live evidence by claim", expanded=False):
            for item in claims:
                live = item.get("live") or {}
                st.markdown(f"**{item['claim']}**")
                st.caption(f"{live.get('message', item['message'])}")
                for hit in live.get("evidence", [])[:3]:
                    title = hit.get("title") or "Untitled source"
                    publisher = hit.get("publisher") or hit.get("source") or "live"
                    url = hit.get("url") or ""
                    st.markdown(f"- **{publisher}**: {title}")
                    if url:
                        st.caption(url)
                st.divider()
