"""AI Recommender page — rule-based + optional LLM-enhanced methodology advice."""

from __future__ import annotations

import logging

import streamlit as st

from aquascope.ai_engine.knowledge_base import METHODOLOGIES
from aquascope.dashboard import _state

logger = logging.getLogger(__name__)

_HOSTED_KEY = "hosted"

_PROVIDER_LABELS = {
    "rule_based": "Rule-based (free, no key needed)",
    "huggingface": "HuggingFace Inference API (free tier — bring your own token)",
    "groq": "Groq (free tier — fast open-source models)",
    "openai": "OpenAI",
    "ollama": "Ollama (local)",
}

_PROVIDER_KEY_LINKS = {
    "huggingface": ("Get free HF token", "https://huggingface.co/settings/tokens"),
    "groq": ("Get free Groq key", "https://console.groq.com/keys"),
    "openai": ("OpenAI API keys", "https://platform.openai.com/api-keys"),
    "ollama": None,
}


def render() -> None:
    st.title("🤖 AI Methodology Recommender")
    st.markdown(
        "Get research-methodology recommendations matched to your dataset's profile — "
        f"{len(METHODOLOGIES)} methodologies scored by parameter mix, record count, spatial and temporal coverage."
    )

    llm_config = _render_llm_config()

    tab_auto, tab_manual = st.tabs(["✨ Auto-detect from workspace", "✍️ Manual profile"])

    with tab_auto:
        df = _state.get_data()
        if df is None:
            st.info("No dataset in the workspace — load one, or use the Manual profile tab.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✨ Load demo dataset", width="stretch", key="ai_demo"):
                    _state.load_demo("water_quality")
                    st.rerun()
            with c2:
                if st.button("🌐 Collect real data →", width="stretch", key="ai_collect"):
                    _state.goto("collect")
        else:
            st.success(f"Workspace dataset: **{len(df):,} records** ({_state.source_label()})")
            goal_auto = st.text_area(
                "Research goal (optional)",
                key="goal_auto",
                placeholder="e.g. Trend analysis of dissolved oxygen in the Tamsui River",
            )
            top_k_auto = st.slider("Number of recommendations", 1, 20, 5, key="topk_auto")

            if st.button("🔍 Recommend from data", key="btn_rec_auto", type="primary"):
                with st.spinner("Profiling dataset and generating recommendations…"):
                    try:
                        from aquascope.analysis.eda import profile_dataset

                        profile = profile_dataset(df)
                        if goal_auto:
                            profile.research_goal = goal_auto

                        result = _run_recommender(profile, top_k_auto, llm_config)
                        _display_mode_banner(result, llm_config)
                        _display_recommendations(result.recommendations)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Recommendation failed: {exc}")
                        logger.exception("AI recommender error")

    with tab_manual:
        goal = st.text_area("Research goal", placeholder="e.g. Trend analysis of dissolved oxygen in Tamsui River")
        parameters = st.text_input("Parameters (comma-separated)", placeholder="DO, BOD5, COD, pH, NH3-N")
        c1, c2 = st.columns(2)
        scope = c1.text_input("Geographic scope", value="Taiwan")
        keywords = c2.text_input("Keywords (comma-separated)", placeholder="trend, seasonal, water quality")

        c3, c4, c5 = st.columns(3)
        n_records = c3.number_input("Number of records", 0, 1_000_000, 0)
        n_stations = c4.number_input("Number of stations", 0, 10_000, 0)
        years = c5.number_input("Time span (years)", 0.0, 100.0, 0.0)

        top_k = st.slider("Number of recommendations", 1, 20, 5)

        if st.button("🔍 Get recommendations", key="btn_rec_manual", type="primary"):
            with st.spinner("Generating recommendations…"):
                try:
                    from aquascope.ai_engine.recommender import DatasetProfile

                    profile = DatasetProfile(
                        parameters=[p.strip() for p in parameters.split(",") if p.strip()],
                        n_records=int(n_records),
                        n_stations=int(n_stations),
                        time_span_years=float(years),
                        geographic_scope=scope,
                        research_goal=goal,
                        keywords=[k.strip() for k in keywords.split(",") if k.strip()],
                    )

                    result = _run_recommender(profile, top_k, llm_config)
                    _display_mode_banner(result, llm_config)
                    _display_recommendations(result.recommendations)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Recommendation failed: {exc}")
                    logger.exception("AI recommender error")


def _run_recommender(profile, top_k: int, llm_config: dict | None):
    """Run the recommender in whichever mode the user selected."""
    from aquascope.ai_engine.recommender import (
        RecommendationResult,
        recommend,
        recommend_with_llm_detailed,
    )

    if llm_config:
        return recommend_with_llm_detailed(
            profile,
            top_k=top_k,
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )
    return RecommendationResult(
        recommendations=recommend(profile, top_k=top_k), mode="rule_based"
    )


def _display_mode_banner(result, llm_config: dict | None) -> None:
    """Say which engine produced the list, and why the LLM was skipped."""
    is_hosted = bool(llm_config and llm_config.get("hosted"))

    if result.mode == "llm":
        if is_hosted:
            st.success(f"✨ Enhanced by the free hosted demo model · `{result.model}`")
        else:
            st.success(f"✨ Enhanced by {result.provider} · `{result.model}`")
        return

    if is_hosted:
        # The shared free quota is finite; say so and point at the way out.
        st.warning(
            f"**Free hosted AI unavailable — showing rule-based results.** {result.error} "
            "You can add your own free API key under **⚙️ LLM enhancement** above.",
            icon="⚠️",
        )
        return

    if llm_config:
        # The user asked for an LLM and did not get one — never fail silently.
        st.warning(
            f"**LLM unavailable — showing rule-based results.** {result.error}",
            icon="⚠️",
        )
        return

    st.caption("Scored by the built-in rule-based engine (no LLM requested).")


_SECRET_KEYS = (
    "AQUASCOPE_LLM_API_KEY",
    "AQUASCOPE_LLM_BASE_URL",
    "AQUASCOPE_LLM_MODEL",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _hydrate_secrets_into_env() -> None:
    """
    Mirror Streamlit secrets into the environment.

    HuggingFace Spaces expose secrets as environment variables, but Streamlit
    Community Cloud exposes them through ``st.secrets``.  Copying them across
    lets one deployment-supplied token work on either host, and keeps the
    library side (``hosted_llm_config``) free of any Streamlit dependency.
    """
    import os

    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — no secrets file configured is normal
        return

    for key in _SECRET_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = secrets[key]
        except Exception:  # noqa: BLE001 — key simply absent
            continue
        if value:
            os.environ[key] = str(value)


def _render_llm_config() -> dict | None:
    from aquascope.ai_engine.recommender import (
        PROVIDER_BASE_URLS,
        PROVIDER_MODELS,
        hosted_llm_config,
    )

    # Set by the deployment (e.g. HF_TOKEN as a Space secret), never bundled.
    _hydrate_secrets_into_env()
    hosted = hosted_llm_config()

    options = list(_PROVIDER_LABELS.keys())
    labels = dict(_PROVIDER_LABELS)
    if hosted:
        options.insert(0, _HOSTED_KEY)
        labels[_HOSTED_KEY] = "✨ Free AI — hosted for this demo, no key needed"

    with st.expander("⚙️ LLM enhancement (optional)", expanded=False):
        if hosted:
            st.caption(
                "This deployment provides a free hosted model, so you can try the "
                "AI recommender without an account. It runs on a shared free quota — "
                "add your own key below if you hit a limit or want a different model."
            )
        else:
            st.caption(
                "Augment rule-based scoring with a language model for richer rationales. "
                "HuggingFace and Groq offer free tiers — no credit card required."
            )

        provider = st.selectbox(
            "Provider",
            options,
            format_func=lambda k: labels[k],
            key="llm_provider",
        )
        if provider == _HOSTED_KEY:
            st.caption(f"Model: `{hosted['model']}` via {hosted['provider']}")
            return hosted
        if provider == "rule_based":
            return None

        default_models = PROVIDER_MODELS.get(provider, [])
        model = st.selectbox("Model", default_models + ["(custom)"], key="llm_model_select")
        if model == "(custom)":
            model = st.text_input("Custom model name", key="llm_model_custom")

        link_info = _PROVIDER_KEY_LINKS.get(provider)
        if link_info:
            label, url = link_info
            st.caption(f"[{label}]({url})")

        if provider == "ollama":
            base_url = st.text_input("Ollama base URL", value=PROVIDER_BASE_URLS["ollama"], key="llm_base_url")
            api_key = None
        else:
            api_key = st.text_input("API key", type="password", key="llm_api_key")
            base_url = PROVIDER_BASE_URLS.get(provider)

        if not model:
            st.warning("Select or enter a model name.")
            return None

        return {"provider": provider, "model": model, "api_key": api_key or None, "base_url": base_url}


def _display_recommendations(recs: list) -> None:
    if not recs:
        st.warning("No recommendations found. Try broadening your parameters or goal.")
        return

    st.subheader(f"Top {len(recs)} recommendations")

    for i, rec in enumerate(recs, 1):
        # Status is shown with icon + text, never color alone.
        icon = "🟢" if rec.score >= 60 else "🟡" if rec.score >= 30 else "🔴"
        fit = "strong fit" if rec.score >= 60 else "possible fit" if rec.score >= 30 else "weak fit"
        with st.expander(f"{icon} #{i} — {rec.methodology.name} · score {rec.score:.0f} ({fit})", expanded=(i <= 3)):
            st.markdown(f"**Category:** {rec.methodology.category}")
            st.markdown(f"**Description:** {rec.methodology.description}")
            if rec.rationale:
                st.markdown(f"**Rationale:** {rec.rationale}")
            if rec.methodology.applicable_parameters:
                st.markdown(f"**Applicable parameters:** {', '.join(rec.methodology.applicable_parameters)}")
            if rec.methodology.tags:
                st.markdown(f"**Tags:** {', '.join(rec.methodology.tags)}")
            st.progress(min(1.0, rec.score / 100))
