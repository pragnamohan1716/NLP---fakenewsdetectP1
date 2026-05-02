"""
Fake News Detection - Streamlit Web Application.
Run: streamlit run app.py
"""
import time
from datetime import datetime
import html as html_lib

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc

import config
from evaluate import ensemble_weighted_probability
from preprocess import clean_for_display
from data_loader import load_kaggle_data, combine_title_text, get_train_val_test

# Page config: wider layout with sidebar
st.set_page_config(
    page_title="Veracity Lab | Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — modern dashboard / editorial feel
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: linear-gradient(165deg, #f0f4ff 0%, #f8fafc 35%, #eef2ff 100%);
    }
    .main .block-container {
        padding-top: 1.25rem;
        max-width: 1200px;
    }
    h1 {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 2.35rem;
        font-weight: 400;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }
    h2, h3 { color: #0f172a; }
    .stTextArea label { font-weight: 600; color: #334155; font-size: 0.9rem; }
    div[data-testid="stTabs"] button {
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .hero-wrap {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        border-radius: 16px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 20px 50px rgba(49, 46, 129, 0.35);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .hero-wrap .hero-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 0.35rem;
    }
    .hero-wrap .hero-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.65rem;
        color: #fff;
        line-height: 1.2;
        margin: 0 0 0.5rem 0;
    }
    .hero-wrap .hero-sub {
        color: #c7d2fe;
        font-size: 0.95rem;
        line-height: 1.45;
        margin: 0;
        max-width: 52rem;
    }
    .input-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.15rem 0.25rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    }
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
        padding: 1rem 1.15rem;
        margin: 0.75rem 0 1.25rem 0;
    }
    .badge-fake {
        display: inline-block;
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        color: #fff;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .badge-real {
        display: inline-block;
        background: linear-gradient(90deg, #059669, #047857);
        color: #fff;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .badge-ready {
        display: inline-block;
        background: linear-gradient(90deg, #16a34a, #047857);
        color: #fff;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        vertical-align: middle;
    }
    .badge-missing {
        display: inline-block;
        background: linear-gradient(90deg, #64748b, #334155);
        color: #fff;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        vertical-align: middle;
    }
    .result-box {
        padding: 1.1rem 1.35rem;
        border-radius: 14px;
        margin: 0.75rem 0 1.25rem 0;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #64748b;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
    }
    .result-fake {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, #fff5f5 0%, #fef2f2 50%, #ffffff 100%);
        border-color: #fecaca;
    }
    .result-real {
        border-left-color: #059669;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 50%, #ffffff 100%);
        border-color: #a7f3d0;
    }
    .word-highlight {
        background: #fef08a;
        padding: 0 3px;
        border-radius: 4px;
    }
    .word-highlight-fake {
        background: #b91c1c;
        color: #fff;
        padding: 0 3px;
        border-radius: 4px;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #fafafa 0%, #f1f5f9 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load all three models once and cache."""
    nb = lstm_model = lstm_tok = maxlen = bert_model = bert_tok = None
    try:
        from models.naive_bayes_model import load_nb
        nb = load_nb()
    except Exception as e:
        st.warning("Naive Bayes model not loaded. Run train_all.py first.")
    try:
        from models.lstm_model import load_lstm
        lstm_model, lstm_tok, maxlen = load_lstm()
    except Exception as e:
        st.error(f"LSTM loading error: {e}")
    try:
        from models.bert_model import load_bert
        bert_model, bert_tok = load_bert()
    except Exception as e:
        st.warning("BERT model not loaded. Run train_all.py first.")
    return {
        "nb": nb,
        "lstm_model": lstm_model,
        "lstm_tok": lstm_tok,
        "lstm_maxlen": maxlen,
        "bert_model": bert_model,
        "bert_tok": bert_tok,
    }


def run_inference(text, models):
    """Run all loaded models and return probas + optional SHAP/NB word importance."""
    from models.naive_bayes_model import predict_proba_nb, get_nb_feature_importance
    from models.lstm_model import predict_proba_lstm
    from models.bert_model import predict_proba_bert

    content = clean_for_display(text) if text else ""
    if not content.strip():
        return None

    probas = []
    # NB
    if models["nb"] is not None:
        p = predict_proba_nb(models["nb"], [content])
        probas.append(("Naive Bayes", p[0]))
    if models["lstm_model"] is not None:
        p = predict_proba_lstm(
            models["lstm_model"], models["lstm_tok"], models["lstm_maxlen"], [content]
        )
        probas.append(("LSTM", p[0]))
    if models["bert_model"] is not None:
        p = predict_proba_bert(models["bert_model"], models["bert_tok"], [content])
        probas.append(("BERT", p[0]))

    if not probas:
        return None

    # Ensemble weighted probability
    weights = config.ENSEMBLE_WEIGHTS[: len(probas)]
    weights = np.array(weights) / sum(weights)
    proba_list = [np.array([p]) for _, p in probas]
    avg_proba, _ = ensemble_weighted_probability(proba_list, weights)
    ensemble_proba = avg_proba[0]

    # Word importance (NB only) – try SHAP-style first, then fall back
    important_words = []
    if models["nb"] is not None:
        shap_words = compute_shap_like_importance(models["nb"], text, top_k=15)
        if shap_words:
            important_words = shap_words
        else:
            important_words = get_nb_feature_importance(models["nb"], text, top_k=15)

    return {
        "probas": probas,
        "ensemble_proba": ensemble_proba,
        "important_words": important_words,
    }


def render_confidence_bar(prob_real, prob_fake, label=""):
    """Draw a horizontal bar for Real vs Fake confidence."""
    fig, ax = plt.subplots(figsize=(6.2, 1.0))
    ax.barh([0], [prob_real], height=0.55, color="#059669", label="Real", edgecolor="white", linewidth=0.5)
    ax.barh([0], [prob_fake], height=0.55, left=prob_real, color="#ef4444", label="Fake", edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Probability share")
    if label:
        ax.set_title(label, fontsize=10, fontweight="600", color="#334155")
    ax.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


def render_per_model_bars(probas, threshold=0.5):
    """Horizontal bar chart of P(Fake) per model."""
    names = [n for n, _ in probas]
    p_fake = [float(p[1]) for _, p in probas]
    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.5 * len(names))))
    palette = ["#4f46e5", "#7c3aed", "#db2777", "#0d9488", "#ca8a04"]
    colors = [palette[i % len(palette)] for i in range(len(names))]
    y = np.arange(len(names))
    ax.barh(y, p_fake, color=colors, height=0.52, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(Fake)")
    ax.axvline(
        threshold,
        color="#94a3b8",
        linestyle="--",
        linewidth=1,
        label=f"{threshold:.2f} threshold",
    )
    ax.set_title("Per-model agreement", fontsize=11, fontweight="600", color="#0f172a", pad=10)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()
    return fig


def build_analysis_report(text_snippet, res, pred, confidence, prob_real, prob_fake, elapsed):
    """Plain-text summary for download."""
    lines = [
        "=== Veracity Lab — Analysis Report ===",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Verdict: {pred} (confidence {confidence:.1%})",
        f"P(Real): {prob_real:.2%} | P(Fake): {prob_fake:.2%}",
    ]
    if elapsed is not None:
        lines.append(f"Inference time: {elapsed:.2f} s")
    lines.append("")
    lines.append("Per-model P(Real) / P(Fake):")
    for name, p in res["probas"]:
        lines.append(f"  {name}: real {float(p[0]):.2%}, fake {float(p[1]):.2%}")
    lines.append("")
    lines.append("Text (preview):")
    preview = text_snippet[:2000] + ("..." if len(text_snippet) > 2000 else "")
    lines.append(preview)
    return "\n".join(lines)


def highlight_words(text, word_scores, fake_direction=True):
    """Highlight words in text by wrapping with span. word_scores: list of (word, score)."""
    from preprocess import tokenize_for_importance
    cleaned = clean_for_display(text)
    tokens = tokenize_for_importance(text)
    word_to_score = {w: s for w, s in word_scores}
    out = []
    for w in tokens:
        if w in word_to_score:
            s = word_to_score[w]
            cls = "word-highlight-fake" if (fake_direction and s > 0) or (not fake_direction and s < 0) else "word-highlight"
            out.append(f'<span class="{cls}">{w}</span>')
        else:
            out.append(w)
    return " ".join(out)


def compute_shap_like_importance(nb_pipeline, text, top_k=15):
    """
    Try to compute SHAP-style word importances for the Naive Bayes model.
    If SHAP is not available or anything fails, returns None and the app
    will fall back to the built-in NB importance.
    """
    try:
        import shap
    except Exception:
        return None

    try:
        # Use SHAP's generic Explainer on the pipeline's predict_proba.
        explainer = shap.Explainer(nb_pipeline.predict_proba)
        shap_values = explainer([text])
        # For binary classification take contributions to the configured Fake class index
        if shap_values.values.ndim == 3:
            # shape: (1, n_features, n_classes)
            vals = shap_values.values[0, :, config.LABEL_INDEX_FAKE]
        else:
            vals = shap_values.values[0]
        feature_names = shap_values.feature_names
        pairs = [(feature_names[i], float(vals[i])) for i in range(len(feature_names))]
        # Sort by absolute impact
        pairs.sort(key=lambda x: -abs(x[1]))
        return pairs[:top_k]
    except Exception:
        return None


# --- UI ---
st.markdown(
    """
<div class="hero-wrap">
    <div class="hero-kicker">Veracity Lab</div>
    <p class="hero-title">Multi-model fake news analysis</p>
    <p class="hero-sub">
        Ensemble <strong>Naive Bayes</strong>, <strong>LSTM</strong> &amp; <strong>BERT</strong>
        · explainability · word patterns · ROC on your dataset
    </p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("### Paste a headline or article")
st.caption("Run **Analyze** to fuse model scores, explore word patterns, and inspect ROC curves.")

# Sidebar: info + controls
with st.sidebar:
    st.subheader("About")
    st.markdown(
        "- **Fusion**: weighted blend of model probabilities\n"
        "- **Explainability**: SHAP-style + NB word cues\n"
        "- **Analysis**: word clouds, frequencies, ROC"
    )
    st.markdown("---")
    st.markdown("**Fusion weights** (NB · LSTM · BERT)")
    _w = config.ENSEMBLE_WEIGHTS
    st.caption(
        f"{_w[0]:.0%} · {_w[1]:.0%} · {_w[2]:.0%} — adjust in `config.ENSEMBLE_WEIGHTS`"
    )
    with st.expander("How fusion works"):
        st.markdown(
            "Each loaded model outputs **P(Real)** and **P(Fake)**. "
            "We take a **weighted average** of those vectors (see sidebar weights), "
            "then label **Fake** if averaged P(Fake) ≥ 0.5. "
            "Missing models are dropped and remaining weights are renormalized."
        )
    st.markdown("---")
    st.markdown(
        "**Tip**: Paste a headline or full article below, then click **Analyze Text**."
    )
    st.markdown("---")
    st.markdown("**Display options**")
    decision_threshold = st.slider(
        "Fake decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="If P(Fake) >= threshold, the app labels the article as Fake.",
    )
    display_mode = st.radio(
        "Choose what to show",
        ["Ensemble (weighted)", "Per-model prediction", "Explainable AI"],
        index=0,
    )
    sidebar_show_highlights = st.checkbox(
        "Highlight important words",
        value=st.session_state.get("show_highlights", False),
        key="sidebar_highlights",
    )

# Session state for input and results
if "news_text" not in st.session_state:
    st.session_state.news_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None
if "show_highlights" not in st.session_state:
    st.session_state.show_highlights = False

news_input = st.text_area(
    "News title or full article text",
    value=st.session_state.news_text,
    height=160,
    placeholder="Paste the headline or full article here...",
    key="input_area",
)

action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
with action_col1:
    analyze_clicked = st.button("Analyze Text", type="primary", use_container_width=True)
with action_col2:
    clear_clicked = st.button("Clear / Reset", use_container_width=True)
with action_col3:
    show_highlights = st.checkbox(
        "Highlight important words",
        value=sidebar_show_highlights,
        key="cb_highlights",
    )

if clear_clicked:
    st.session_state.news_text = ""
    st.session_state["input_area"] = ""
    st.session_state.result = None
    st.session_state.processing_time = None
    st.session_state.show_highlights = show_highlights
    st.rerun()

models = load_models()

with st.sidebar:
    st.markdown("---")
    st.subheader("Model status")
    st.caption("Ready models are loaded from `models/` and used for inference.")
    ready_span = '<span class="badge-ready">READY</span>'
    missing_span = '<span class="badge-missing">MISSING</span>'
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"**Naive Bayes** {ready_span if models.get('nb') is not None else missing_span}",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"**LSTM** {ready_span if models.get('lstm_model') is not None else missing_span}",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"**BERT** {ready_span if models.get('bert_model') is not None else missing_span}",
            unsafe_allow_html=True,
        )

input_text_for_model = None

if analyze_clicked and news_input.strip():
    input_text_for_model = news_input.strip()

if input_text_for_model:
    with st.spinner("Analyzing..."):
        t0 = time.perf_counter()
        result = run_inference(input_text_for_model, models)
        t1 = time.perf_counter()
    st.session_state.news_text = input_text_for_model
    st.session_state.result = result
    st.session_state.processing_time = t1 - t0
    st.session_state.show_highlights = show_highlights
    if result is None:
        st.error("No models loaded. Run `python train_all.py` and place trained models in the `models/` folder.")

# Display result + analysis tabs
if st.session_state.result is not None:
    res = st.session_state.result
    ensemble_proba = res["ensemble_proba"]

    # Model outputs are always ordered as [P(real), P(fake)]
    prob_real = float(ensemble_proba[0])
    prob_fake = float(ensemble_proba[1])

    pred = "Fake" if prob_fake >= decision_threshold else "Real"
    confidence = prob_fake if pred == "Fake" else prob_real

    # Input summary card (kept small to avoid UI clutter)
    preview_raw = " ".join(news_input.strip().split())
    preview = preview_raw[:220] + ("..." if len(preview_raw) > 220 else "")
    preview_safe = html_lib.escape(preview)
    st.markdown(
        f'<div class="section-card">'
        f'<strong>Input summary</strong><br/>'
        f'<span style="color:#334155">Words: {len(preview_raw.split())} · Characters: {len(news_input)}</span><br/>'
        f'<span style="display:block; margin-top:0.5rem; color:#0f172a;">{preview_safe}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    detector_tab, words_tab, perf_tab = st.tabs(
        ["Fusion Decision", "Word Patterns", "Model Performance"]
    )

    # --- Fusion Decision tab (real-time detector) ---
    with detector_tab:
        main_left, main_right = st.columns([2, 1])

        # Ensemble (weighted) view
        if display_mode == "Ensemble (weighted)":
            with main_left:
                st.subheader("Fusion decision (ensemble)")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Verdict", pred, delta=None)
                with m2:
                    st.metric("P (Fake)", f"{prob_fake:.1%}")
                with m3:
                    st.metric("P (Real)", f"{prob_real:.1%}")

                css_class = "result-fake" if pred == "Fake" else "result-real"
                badge = (
                    '<span class="badge-fake">FAKE</span>'
                    if pred == "Fake"
                    else '<span class="badge-real">REAL</span>'
                )
                st.markdown(
                    f'<div class="result-box {css_class}">'
                    f'<strong>Prediction:</strong> {pred} {badge}<br/>'
                    f'<strong>Confidence:</strong> {confidence:.1%} &nbsp;·&nbsp; '
                    f'<strong>Blend:</strong> P(Real) {prob_real:.1%}, P(Fake) {prob_fake:.1%}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("#### Ensemble confidence")
                fig = render_confidence_bar(prob_real, prob_fake, "Ensemble (weighted)")
                st.pyplot(fig)
                plt.close()

                st.markdown("#### Model comparison")
                fig_cmp = render_per_model_bars(res["probas"], threshold=decision_threshold)
                st.pyplot(fig_cmp)
                plt.close()

                _report = build_analysis_report(
                    news_input,
                    res,
                    pred,
                    confidence,
                    prob_real,
                    prob_fake,
                    st.session_state.processing_time,
                )
                st.download_button(
                    label="Download report (.txt)",
                    data=_report,
                    file_name="veracity_lab_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_report_ensemble",
                )

            with main_right:
                if st.session_state.processing_time is not None:
                    st.metric("Inference time", f"{st.session_state.processing_time:.2f}s")
                st.markdown("**Tips**")
                st.caption(
                    "Compare the bar chart to the fusion bar: models that disagree "
                    "strongly often drive borderline ensemble scores."
                )

        # Per-model prediction view
        elif display_mode == "Per-model prediction":
            with main_left:
                st.subheader("Per-model prediction")
                rows = []
                for name, p in res["probas"]:
                    pr, pf = float(p[0]), float(p[1])
                    pred_i = "Fake" if pf >= decision_threshold else "Real"
                    rows.append(
                        {
                            "Model": name,
                            "Prediction": pred_i,
                            "P(Real)": f"{pr:.2%}",
                            "P(Fake)": f"{pf:.2%}",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown("#### P(Fake) by model")
                fig_pm = render_per_model_bars(res["probas"], threshold=decision_threshold)
                st.pyplot(fig_pm)
                plt.close()
                _report2 = build_analysis_report(
                    news_input,
                    res,
                    pred,
                    confidence,
                    prob_real,
                    prob_fake,
                    st.session_state.processing_time,
                )
                st.download_button(
                    label="Download report (.txt)",
                    data=_report2,
                    file_name="veracity_lab_report.txt",
                    mime="text/plain",
                    key="download_report_per_model",
                )

            with main_right:
                st.markdown("#### Fusion decision")
                css_class = "result-fake" if pred == "Fake" else "result-real"
                badge = (
                    '<span class="badge-fake">FAKE</span>'
                    if pred == "Fake"
                    else '<span class="badge-real">REAL</span>'
                )
                st.markdown(
                    f'<div class="result-box {css_class}">'
                    f'<strong>Ensemble:</strong> {pred} {badge}<br/>'
                    f'<strong>Confidence:</strong> {confidence:.1%}'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.session_state.processing_time is not None:
                    st.caption(f"⏱ {st.session_state.processing_time:.2f} s")

        # Explainable AI view
        elif display_mode == "Explainable AI":
            with main_left:
                st.subheader("Explainable AI (SHAP-style)")
                ex1, ex2, ex3 = st.columns(3)
                with ex1:
                    st.metric("Verdict", pred)
                with ex2:
                    st.metric("P (Fake)", f"{prob_fake:.1%}")
                with ex3:
                    st.metric("P (Real)", f"{prob_real:.1%}")
                st.markdown(
                    "For this article, each model outputs its own prediction, and the highlighted "
                    "words show which tokens most strongly push the ensemble towards **fake**."
                )

                # Per-model textual explanation
                for name, p in res["probas"]:
                    pr, pf = float(p[0]), float(p[1])
                    pred_i = "Fake" if pf >= decision_threshold else "Real"
                    st.markdown(
                        f"- **{name}** predicts **{pred_i}** "
                        f"(P(fake) = {pf:.1%}, P(real) = {pr:.1%})."
                    )

                if show_highlights and res.get("important_words"):
                    st.markdown("")
                    st.markdown(
                        "The highlight uses SHAP-style importance (falling back to Naive Bayes "
                        "word importance if SHAP is unavailable). Stronger color means a bigger "
                        "push towards predicting *fake*."
                    )
                    highlighted = highlight_words(
                        news_input, res["important_words"], fake_direction=True
                    )
                    st.markdown(highlighted, unsafe_allow_html=True)

                _report3 = build_analysis_report(
                    news_input,
                    res,
                    pred,
                    confidence,
                    prob_real,
                    prob_fake,
                    st.session_state.processing_time,
                )
                st.download_button(
                    label="Download report (.txt)",
                    data=_report3,
                    file_name="veracity_lab_report.txt",
                    mime="text/plain",
                    key="download_report_explain",
                )

            with main_right:
                if st.session_state.processing_time is not None:
                    st.metric("Inference time", f"{st.session_state.processing_time:.2f}s")
                st.caption("Toggle **Highlight important words** to see token-level cues.")

    # --- Word Patterns tab: word clouds & frequencies ---
    with words_tab:
        st.subheader("WordClouds and most frequent words")
        try:
            df = load_kaggle_data()
            df = combine_title_text(df)
            _, _, test_df = get_train_val_test(df)
            fake_texts = test_df[test_df[config.LABEL_COL] == config.LABEL_INDEX_FAKE]["content"]
            real_texts = test_df[test_df[config.LABEL_COL] == config.LABEL_INDEX_REAL]["content"]

            col_wc_fake, col_wc_real = st.columns(2)
            # WordClouds
            with col_wc_fake:
                st.markdown("**Fake news WordCloud**")
                text = " ".join(fake_texts.astype(str))
                if text.strip():
                    wc_fake = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="Reds",
                    ).generate(text)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.imshow(wc_fake, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No fake news samples found for WordCloud.")

            with col_wc_real:
                st.markdown("**Real news WordCloud**")
                text = " ".join(real_texts.astype(str))
                if text.strip():
                    wc_real = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="Greens",
                    ).generate(text)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.imshow(wc_real, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No real news samples found for WordCloud.")

            st.markdown("---")
            # Most frequent words
            from collections import Counter

            def top_words(series, top_k=20):
                tokens = " ".join(series.astype(str)).split()
                counter = Counter(tokens)
                return pd.DataFrame(counter.most_common(top_k), columns=["Word", "Count"])

            col_freq_fake, col_freq_real = st.columns(2)
            with col_freq_fake:
                st.markdown("**Most frequent words in fake news**")
                st.dataframe(top_words(fake_texts), hide_index=True, use_container_width=True)
            with col_freq_real:
                st.markdown("**Most frequent words in real news**")
                st.dataframe(top_words(real_texts), hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(
                "Unable to load dataset for word-level analysis. "
                "Make sure the Kaggle CSV files are present in the `data/` folder."
            )

    # --- Model Performance tab: ROC curves ---
    with perf_tab:
        st.subheader("ROC curves for model comparison")
        st.caption(
            "Evaluated on the held-out test split of the Kaggle Fake/Real News dataset."
        )

        try:
            df = load_kaggle_data()
            df = combine_title_text(df)
            _, _, test_df = get_train_val_test(df)
            X_test = test_df["content"].tolist()
            y_test = test_df[config.LABEL_COL].values
            # ROC-AUC should treat "fake" as the positive class.
            y_fake = (y_test == config.LABEL_INDEX_FAKE).astype(int)

            from models.naive_bayes_model import predict_proba_nb
            from models.lstm_model import predict_proba_lstm
            from models.bert_model import predict_proba_bert

            curves = {}
            proba_list = []
            labels = []

            # Naive Bayes
            if models["nb"] is not None:
                p_nb = predict_proba_nb(models["nb"], X_test)
                fpr, tpr, _ = roc_curve(y_fake, p_nb[:, 1])
                curves["Naive Bayes"] = (fpr, tpr, auc(fpr, tpr))
                proba_list.append(p_nb)
                labels.append("Naive Bayes")

            # LSTM
            if models["lstm_model"] is not None:
                p_lstm = predict_proba_lstm(
                    models["lstm_model"], models["lstm_tok"], models["lstm_maxlen"], X_test
                )
                fpr, tpr, _ = roc_curve(y_fake, p_lstm[:, 1])
                curves["LSTM"] = (fpr, tpr, auc(fpr, tpr))
                proba_list.append(p_lstm)
                labels.append("LSTM")

            # BERT
            if models["bert_model"] is not None:
                p_bert = predict_proba_bert(models["bert_model"], models["bert_tok"], X_test)
                fpr, tpr, _ = roc_curve(y_fake, p_bert[:, 1])
                curves["BERT"] = (fpr, tpr, auc(fpr, tpr))
                proba_list.append(p_bert)
                labels.append("BERT")

            # Ensemble fusion ROC
            if proba_list:
                weights = config.ENSEMBLE_WEIGHTS[: len(proba_list)]
                weights = np.array(weights) / np.sum(weights)
                stacked = np.stack(proba_list, axis=0)
                ensemble_proba = np.tensordot(weights, stacked, axes=(0, 0))
                fpr, tpr, _ = roc_curve(y_fake, ensemble_proba[:, 1])
                curves["Ensemble (fusion decision)"] = (fpr, tpr, auc(fpr, tpr))

            if not curves:
                st.warning(
                    "No models are loaded, so ROC curves cannot be computed. "
                    "Run `python train_all.py` first."
                )
            else:
                roc_colors = ["#4f46e5", "#7c3aed", "#db2777", "#0d9488", "#ca8a04"]
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.set_facecolor("#f8fafc")
                fig.patch.set_facecolor("#ffffff")
                for i, (name, (fpr, tpr, roc_auc)) in enumerate(curves.items()):
                    c = roc_colors[i % len(roc_colors)]
                    ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
                ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2, label="Random")
                ax.set_xlabel("False Positive Rate", fontsize=10, color="#334155")
                ax.set_ylabel("True Positive Rate", fontsize=10, color="#334155")
                ax.set_title("ROC curves (Fake = positive class)", fontsize=12, fontweight="600", color="#0f172a", pad=12)
                ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
                ax.grid(alpha=0.35, linestyle=":", color="#cbd5e1")
                st.pyplot(fig)
                plt.close()
        except Exception:
            st.warning(
                "Unable to compute ROC curves. Ensure the dataset and trained models are available."
            )

st.markdown("---")
st.caption(
    "**Veracity Lab** — ensemble NB + LSTM + BERT. Train or refresh artifacts with `python train_all.py`."
)