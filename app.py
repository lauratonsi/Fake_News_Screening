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
    "Bi-GRU and a Bi-LSTM vote on the input; a live fact-check verdict (when "
    "available) takes precedence, and reference retrieval, a manipulation-"
    "technique detector and a confidence tier all add evidence around it."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. **Ensemble** — three models trained on ISOT + WELFake + COVID-19
   score the text; their average is the fake probability.
2. **Live fact-check overrides the ensemble** — if a professional
   fact-checker (Google Fact Check Tools) has already rated the claim, that
   verdict wins: it's the only signal that is both authoritative and
   *current*, so it can correct the models on events the 2015–2020 training
   data never saw.
3. **Most similar known articles** — the input is compared, by semantic
    embedding similarity, with snippets of known real/fake articles (so
    reworded claims still match). Matching a known *fake* claim is evidence
    of fakeness and boosts the score; matching real reporting is shown as
    evidence only, since sharing a topic with true news is not proof — it
    sways the verdict solely on a near-verbatim match.
4. **Manipulation-technique detection** — flags *how* the text tries to
   persuade (appeal to hidden knowledge, unverifiable sources, fake
   authority, urgency, us-vs-them), independent of whether the claim is
   true. Naming the technique is *inoculation*, not a verdict.
5. **Confidence tier** — every result is High/Medium/Low confidence. A
   model-only call on a short, out-of-domain claim — where the models are
   most often confidently wrong — is shown as a low-confidence *screening
   signal to verify*, never a settled verdict. This only lowers confidence;
   it never flips a FAKE label to REAL.
6. **Explainability** — the SVM is linear, so its score breaks down into the
   exact words pushing it toward FAKE or REAL.
7. **Agreement check + feedback** — if the models disagree strongly, the
   verdict is flagged for human review; a 👍/👎 form below lets you correct
   the system, logged for future improvement.
        """
    )
    st.header("Limitations")
    st.markdown(
        """
- English-language news only.
- The classifiers are trained on text from 2015–2020; live fact-check
  (when a rating exists) is the only signal that reaches current events —
  everything else is out of domain past that window.
- Reference-corpus matching is **semantic similarity, not verification** —
  it cannot confirm a claim it has never seen in any wording.
- Out-of-domain accuracy is **76.2%** (vs. 94.6% in-domain) on a 101-scenario
  adversarial benchmark — see the README for the full breakdown.
- **Fluent, AI-generated disinformation is measurably harder to catch than
  classic-style hoaxes**: 100% vs. ~74% recall on 43 real, independently-
  sourced test cases (see the README's *"AI-generated disinformation is
  harder to detect"*). Treat every result as a screening aid, not a truth oracle —
  especially on calm, well-sourced-sounding claims.
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

    # Persist across reruns: st.button() is only True in the exact run it was
    # clicked in, but the feedback form below triggers its own reruns (typing
    # a comment, picking a label, submitting). Without session_state, any of
    # those would make `st.button("Analyze")` re-evaluate to False and the
    # entire result panel — including the feedback form itself — would
    # vanish before the submission could be recorded.
    st.session_state["result"] = result
    st.session_state["analyzed_text"] = text

if "result" in st.session_state:
    result = st.session_state["result"]
    analyzed_text = st.session_state["analyzed_text"]

    st.divider()
    confidence = result.get("confidence", "medium")
    evidence_backed = result.get("evidence_backed", False)
    live = result.get("live")

    color = "red" if result["verdict"] == "FAKE" else "green"
    # A model-only verdict on a short/out-of-domain claim is a *screening flag*,
    # not a settled truth judgement — say so plainly instead of a confident
    # colour. Only an evidence-backed verdict gets the strong red/green.
    if confidence == "low" and not evidence_backed:
        color = "#b8860b"  # amber: uncertain, needs verification
    headline = "Verdict" if evidence_backed else "Screening signal"
    st.markdown(
        f"<h2 style='text-align:center;color:{color};'>{headline}: {result['verdict']}</h2>",
        unsafe_allow_html=True,
    )

    badge = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence",
             "low": "🟠 Low confidence — verify"}.get(confidence, "")
    st.markdown(
        f"<h4 style='text-align:center;'>Fake probability: "
        f"{result['fake_probability']:.1%} &nbsp;|&nbsp; {badge}</h4>"
        f"<p style='text-align:center;color:gray;'>{result['reason']}</p>",
        unsafe_allow_html=True,
    )

    # A1: the live fact-check verdict is the highest-signal evidence — surface
    # it prominently when it drove or conflicts with the headline.
    if live and live.get("verdict"):
        src = live.get("source") or "fact-check"
        if live["verdict"] == "FAKE":
            st.error(f"⛔ **Live fact-check ({src}): rated FALSE.** This is a "
                     "professional fact-checker's verdict and takes precedence.")
        else:
            st.success(f"✅ **Live fact-check ({src}): rated TRUE.** A professional "
                       "fact-checker corroborates this claim.")

    if confidence == "low" and not evidence_backed:
        st.warning(
            "⚠️ **Low confidence.** No external evidence (live fact-check or a "
            "near-verbatim known article) corroborates this, and the input is a "
            "short, out-of-domain claim — the range the models get wrong most "
            "often, always by over-flagging true statements. Treat this as a "
            "prompt for human verification, not a verdict."
        )
    if result["needs_review"]:
        st.error(
            "⚠️ The models disagree strongly (or conflict with a fact-checker) "
            "on this text — it would be routed to a human reviewer."
        )

    # --- Manipulation techniques (the prebunking / inoculation layer) --------
    manipulation = result.get("manipulation", {})
    if manipulation.get("count", 0) > 0:
        st.subheader("🎯 Manipulation techniques detected")
        st.caption(
            "How the text tries to persuade you — independent of whether the "
            "underlying claim is true. Naming the technique is *inoculation*: it "
            "builds resistance better than a bare true/false label. This is "
            "evidence, not a verdict — honest reporting can use forceful "
            "language too."
        )
        for tech in manipulation["techniques"]:
            quoted = ", ".join(f"“{m}”" for m in tech["matches"][:6])
            st.markdown(f"**{tech['label']}** — {tech['explanation']}")
            st.caption(f"Flagged phrasing: {quoted}")
    else:
        st.caption(
            "🎯 No manipulation techniques detected in the phrasing (this checks "
            "*how* the text argues, not whether the claim is true)."
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

    # --- Why the SVM scored this (exact linear contributions) ----------------
    explanation = result.get("explanation", {})
    if explanation.get("available"):
        st.subheader("Why the SVM scored this")
        st.caption(
            "The SVM is a linear model, so its score decomposes exactly into "
            "per-word contributions (TF-IDF weight × model coefficient) — this "
            "is the model's own arithmetic, not a post-hoc approximation. The "
            "RNNs stay black boxes by nature."
        )
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("**Pushed toward FAKE**")
            for c in explanation["fake_pushing"][:6]:
                st.markdown(f"- 🟥 `{c['token']}` (+{c['weight']:.3f})")
            if not explanation["fake_pushing"]:
                st.caption("—")
        with ecol2:
            st.markdown("**Pushed toward REAL**")
            for c in explanation["real_pushing"][:6]:
                st.markdown(f"- 🟩 `{c['token']}` ({c['weight']:.3f})")
            if not explanation["real_pushing"]:
                st.caption("—")

    # --- Feedback loop: capture corrections to improve the tool over time ----
    st.divider()
    st.subheader("Was this helpful?")
    st.caption(
        "Your feedback is logged locally to build a real-world evaluation set "
        "and to collect the hard cases (especially true statements wrongly "
        "flagged) for a future retraining round."
    )
    with st.form("feedback_form", clear_on_submit=True):
        agree = st.radio(
            "Is the screening result correct?",
            ["👍 Yes, correct", "👎 No, wrong"],
            horizontal=True,
        )
        correct_label = st.selectbox(
            "If wrong, what is the correct label?",
            ["(leave blank)", "REAL (true statement)", "FAKE (false claim)"],
        )
        comment = st.text_input("Optional comment")
        if st.form_submit_button("Submit feedback"):
            from src.feedback import record_feedback

            label = None
            if correct_label.startswith("REAL"):
                label = "REAL"
            elif correct_label.startswith("FAKE"):
                label = "FAKE"
            record_feedback(
                analyzed_text,
                result,
                agrees=agree.startswith("👍"),
                correct_label=label,
                comment=comment,
            )
            st.success("Thanks — your feedback was recorded.")
