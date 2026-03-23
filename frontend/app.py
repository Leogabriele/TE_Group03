"""
Streamlit Dashboard for LLM Security Auditor
Main interface for interactive testing and visualization
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.api.routes import run_full_audit
from backend.app.core.orchestrator import Orchestrator
from backend.app.agents.strategies import (
    list_all_strategies,
    get_strategy,
    get_strategy_stats,
    STRATEGY_REGISTRY
)
from backend.app.models.database import db
from backend.app.config import settings

st.set_page_config(
    page_title="LLM Security Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "app_initialized" not in st.session_state:
    st.session_state.active_audit_session_id = None
    st.session_state.app_initialized = True
st.markdown("""
<style>
.main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 2rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
.success-card { border-left-color: #28a745; }
.warning-card { border-left-color: #ffc107; }
.danger-card  { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ─── Difficulty helpers ────────────────────────────────────────────────────────
DIFF_ICON  = {"EASY": "🟢", "MEDIUM": "🟡", "HARD": "🔴", "VERY_HARD": "💀"}
DIFF_COLOR = {"EASY": "#28a745", "MEDIUM": "#ffc107", "HARD": "#fd7e14", "VERY_HARD": "#dc3545"}

def _diff(strategy) -> str:
    d = getattr(strategy, "difficulty", None)
    return d.value if d and hasattr(d, "value") else "MEDIUM"

# ─── Dynamic stats ─────────────────────────────────────────────────────────────
def get_strategy_stats_dynamic() -> dict:
    all_strategies = list_all_strategies()
    total = len(all_strategies)
    by_difficulty = {}
    for s in all_strategies:
        d = _diff(s)
        by_difficulty[d] = by_difficulty.get(d, 0) + 1
    avg_eff = (
        sum(getattr(s, "effectiveness_score", 0) for s in all_strategies) / total
        if total > 0 else 0
    )
    return {
        "total": total,
        "by_difficulty": by_difficulty,
        "avg_effectiveness": avg_eff,
    }

# ─── Cached Groq attacker model fetcher ───────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _get_groq_models() -> list:
    """Fetch live Groq model list for attacker selection."""
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        return sorted([m.id for m in client.models.list().data])
    except Exception:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "gemma2-9b-it",
        ]
    
def _attacker_model_selector(key_prefix: str):
    """
    Reusable attacker model selector widget.
    Returns (attacker_provider, attacker_model) — both None = use .env default.
    key_prefix must be unique per call site to avoid duplicate widget keys.
    """
    st.subheader("🤖 Attacker Model")
    st.caption(f"Generates attack prompts. Default: `groq/{settings.ATTACKER_MODEL}`")

    override = st.checkbox(
        "Override Attacker Model",
        value=False,
        key=f"{key_prefix}_override",
        help="Choose a different model to craft the attack prompts"
    )

    if not override:
        st.caption(f"✅ Using `.env` default: `groq/{settings.ATTACKER_MODEL}`")
        return None, None

    # ── Source: Cloud or Local ────────────────────────────────────────────────
    source = st.radio(
        "Attacker Source",
        ["☁️ Cloud (Groq)", "🏠 Local (Ollama)"],
        horizontal=True,
        key=f"{key_prefix}_source",
        help="Groq = free cloud API | Ollama = local model, fully private"
    )

    if source == "☁️ Cloud (Groq)":
        groq_models = _get_groq_models()
        try:
            default_idx = groq_models.index(settings.ATTACKER_MODEL)
        except ValueError:
            default_idx = 0
        chosen = st.selectbox(
            "Groq Attacker Model",
            options=groq_models,
            index=default_idx,
            key=f"{key_prefix}_groq_model",
            help="Free Groq tier — no cost"
        )
        st.caption(f"🔴 Override active: `groq/{chosen}`")
        return "groq", chosen

    else:  # Local Ollama
        st.info("💡 Make sure Ollama is running on `localhost:11434`")
        # Try to fetch installed models live
        ollama_models = []
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                ollama_models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass

        if ollama_models:
            # Also include the configured default at the top
            default_ollama = getattr(settings, "ATTACKER_OLLAMA_MODEL", "llama3.2:latest")
            if default_ollama not in ollama_models:
                ollama_models.insert(0, default_ollama)
            try:
                default_idx = ollama_models.index(default_ollama)
            except ValueError:
                default_idx = 0
            chosen = st.selectbox(
                "Ollama Attacker Model",
                options=ollama_models,
                index=default_idx,
                key=f"{key_prefix}_ollama_model",
                help="Locally installed Ollama models"
            )
            st.caption(f"🔴 Override active: `ollama/{chosen}`")
            return "ollama", chosen
        else:
            # Ollama not reachable or no models — fallback to text input
            st.warning("⚠️ Ollama not reachable — enter model name manually")
            default_ollama = getattr(settings, "ATTACKER_OLLAMA_MODEL", "llama3.2:latest")
            chosen = st.text_input(
                "Ollama Model Name",
                value=default_ollama,
                key=f"{key_prefix}_ollama_manual",
                help="e.g. llama3.2:latest, mistral:7b, phi3:mini"
            )
            if chosen:
                st.caption(f"🔴 Override active: `ollama/{chosen}`")
                return "ollama", chosen
            return None, None


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🛡️ LLM Security Auditor</h1>', unsafe_allow_html=True)
    st.markdown("Automated Red Teaming for Language Models")

    try:
        stats = get_strategy_stats_dynamic()
        st.markdown(
            f"**{stats['total']}** Attack Strategies Available &nbsp;&nbsp; "
            f"Avg Effectiveness: **{stats['avg_effectiveness']:.0f}**"
        )
    except Exception:
        pass

    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        target_type = st.radio(
            "Target Type", ["Cloud API", "Local Model"],
            horizontal=True,
            help="Choose between cloud API or locally running model"
        )
        is_local = target_type == "Local Model"
        st.markdown("---")

        if is_local:
            st.subheader("🏠 Local Model")
            try:
                from backend.app.utils.local_model_manager import LocalModelManager
                manager = LocalModelManager()
                if not manager.check_ollama_installed():
                    st.error("Ollama not running!")
                    st.caption("Go to Configuration → Local Models to install")
                    target_provider, target_model = "ollama", "llama3.2:1b"
                else:
                    installed_models = manager.get_installed_models()
                    if installed_models:
                        model_names = [m["name"] for m in installed_models]
                        target_model = st.selectbox(
                            "Select Model", model_names,
                            help="Choose from installed Ollama models"
                        )
                        target_provider = "ollama"
                        selected_info = next(
                            (m for m in installed_models if m["name"] == target_model), None
                        )
                        if selected_info:
                            st.caption(f"Size: {selected_info.get('size', 'Unknown')}")
                    else:
                        st.warning("No models installed")
                        st.caption("Pull a model: `ollama pull llama3.2:1b`")
                        target_provider, target_model = "ollama", "llama3.2:1b"
            except Exception as e:
                st.error(f"Local model error: {e}")
                target_provider, target_model = "ollama", "llama3.2:1b"

        else:
            st.subheader("☁️ Cloud API")
            from backend.app.models.enums import LLMProvider
            cloud_providers = [p.value for p in LLMProvider if p.value in ("nvidia", "groq")]
            current_provider = getattr(settings, "TARGET_MODEL_PROVIDER", "nvidia")
            try:
                prov_idx = cloud_providers.index(current_provider)
            except ValueError:
                prov_idx = 0

            target_provider = st.selectbox(
                "Provider", cloud_providers,
                index=prov_idx,
                format_func=str.upper,
                help="LLM provider to test"
            )

            if target_provider == "nvidia":
                model_options = [
                    "meta/llama3-70b-instruct",
                    "meta/llama-3.1-405b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "mistralai/mixtral-8x7b-instruct-v0.1",
                    "nvidia/mistral-nemo-minitron-8b-instruct",
                ]
                cfg = getattr(settings, "TARGET_MODEL_NAME", "")
                if cfg and cfg not in model_options:
                    model_options.insert(0, cfg)
                try:
                    model_idx = model_options.index(cfg)
                except ValueError:
                    model_idx = 0
            else:
                try:
                    from groq import Groq
                    client = Groq(api_key=settings.GROQ_API_KEY)
                    model_options = sorted([m.id for m in client.models.list().data])
                except Exception:
                    model_options = [
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",
                        "gemma2-9b-it",
                    ]
                cfg_model = getattr(settings, "TARGET_MODEL_NAME", "")
                try:
                    model_idx = model_options.index(cfg_model)
                except ValueError:
                    model_idx = 0

            target_model = st.selectbox(
                "Model", model_options,
                index=model_idx,
                help="Specific model to audit"
            )

        st.markdown("---")
        st.subheader("⚙️ Audit Settings")
        save_to_db         = st.checkbox("Save results to database", value=True)
        parallel_execution = st.checkbox("Parallel execution", value=True)

        st.markdown("---")
        st.subheader("📊 Quick Stats")
        try:
            stats = get_strategy_stats_dynamic()
            st.metric("Total Strategies", stats["total"])
            for diff, icon in DIFF_ICON.items():
                count = stats["by_difficulty"].get(diff, 0)
                if count:
                    st.metric(f"{icon} {diff.replace('_', ' ').title()}", count)
        except Exception:
            pass

        if st.button("🔄 Refresh Stats"):
            st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Attack Test",
        "Full Audit",
        "Analytics",
        "Strategy Comparison",
        "Model Comparison",
    ])

    with tab1:
        single_attack_interface(target_provider, target_model, is_local)
    with tab2:
        full_audit_interface(target_provider, target_model, is_local, save_to_db, parallel_execution)
    with tab3:
        analytics_interface()
    with tab4:
        strategy_comparison_interface()
    with tab5:
        model_comparison_interface()


# ─── Tab 1: Single Attack ──────────────────────────────────────────────────────
def save_or_update_single_attack_session(result, model, goal, existing_session_id):
    """Bridge for Single Attack tab to persist data into the active session index"""
    async def _run():
        try:
            await db.connect()
            example = {
                "generated_prompt": result.attack.generated_prompt,
                "response_text": result.response.response_text,
                "verdict": result.evaluation.verdict.value,
                "strategy": result.attack.strategy_name,
                "timestamp": datetime.utcnow()
            }
            # update_or_create_session should be implemented in database.py
            return await db.update_or_create_session(
                example=example,
                model=model,
                forbidden_goal=goal,
                session_id=existing_session_id
            )
        finally:
            await db.disconnect()
    return asyncio.run(_run())

def single_attack_interface(target_provider, target_model, is_local):
    st.header("🎯 Single Attack Test")
    st.markdown("Test individual attack strategies interactively")

    type_badge = "LOCAL" if is_local else "CLOUD"
    st.info(f"[{type_badge}] Target: {target_provider}/{target_model}")

    try:
        stats          = get_strategy_stats_dynamic()
        all_strategies = list_all_strategies()
        cols = st.columns(len(DIFF_ICON) + 1)
        cols[0].metric("Total Strategies", stats["total"])
        for i, (diff, icon) in enumerate(DIFF_ICON.items(), 1):
            cols[i].metric(
                f"{icon} {diff.replace('_', ' ').title()}",
                stats["by_difficulty"].get(diff, 0)
            )
        st.markdown("---")
    except Exception as e:
        st.warning(f"Could not load strategy stats: {e}")

    col_search, col_diff = st.columns([3, 2])
    with col_search:
        search_q = st.text_input(
            "Search strategies",
            placeholder="e.g. cipher, persona, roleplay...",
            key="single_search",
            label_visibility="collapsed"
        )
    with col_diff:
        diff_filter = st.selectbox(
            "Difficulty", ["All Difficulties"] + list(DIFF_ICON.keys()),
            key="single_diff_filter",
            label_visibility="collapsed"
        )

    all_strategies      = list_all_strategies()
    filtered_strategies = all_strategies

    if search_q:
        q = search_q.lower()
        filtered_strategies = [
            s for s in filtered_strategies
            if q in s.name.lower()
            or q in getattr(s, "description", "").lower()
            or q in s.strategy_type.value.lower()
        ]
    if diff_filter != "All Difficulties":
        filtered_strategies = [
            s for s in filtered_strategies if _diff(s) == diff_filter
        ]

    st.caption(f"Showing {len(filtered_strategies)} of {len(all_strategies)} strategies")

    col1, col2 = st.columns([2, 1])

    with col1:
        forbidden_goal = st.text_area(
            "Forbidden Goal",
            placeholder="Enter the harmful goal to test (e.g. How to hack email accounts)",
            height=100,
            key="single_attack_goal"
        )

        if not filtered_strategies:
            st.error("No strategies match your filters.")
            return

        strategy_display_map = {}
        for s in filtered_strategies:
            d            = _diff(s)
            icon         = DIFF_ICON.get(d, "")
            mt           = " 🔄" if getattr(s, "multi_turn", False) else ""
            display_name = f"{icon} {s.name}{mt}"
            strategy_display_map[display_name] = s

        selected_display_name = st.selectbox(
            "Attack Strategy",
            list(strategy_display_map.keys()),
            help="Choose which attack strategy to use",
            key="strategy_select"
        )
        selected_strategy_obj = strategy_display_map[selected_display_name]

        with st.expander("Strategy Details", expanded=False):
            st.write(f"**Name:** {selected_strategy_obj.name}")
            st.write(f"**Type:** {selected_strategy_obj.strategy_type.value}")
            st.write(f"**Description:** {getattr(selected_strategy_obj, 'description', 'N/A')}")
            st.write(
                f"**Multi-turn:** "
                f"{'Yes' if getattr(selected_strategy_obj, 'multi_turn', False) else 'No'}"
            )
            if hasattr(selected_strategy_obj, "effectiveness_score"):
                st.write("**Effectiveness**")
                st.progress(selected_strategy_obj.effectiveness_score)
                st.caption(f"{selected_strategy_obj.effectiveness_score:.0f}")
            d = _diff(selected_strategy_obj)
            st.write(f"**Difficulty:** {DIFF_ICON.get(d, '')} {d}")

        # ── Attacker Model Override ───────────────────────────────────────────
        st.markdown("---")
        attacker_provider, attacker_model = _attacker_model_selector("single")
        # ─────────────────────────────────────────────────────────────────────

    with col2:
        st.subheader("Attack Preview")
        if forbidden_goal:
            try:
                preview = selected_strategy_obj.generate(forbidden_goal)
                st.text_area(
                    "Generated Attack",
                    value=preview[:400] + "..." if len(preview) > 400 else preview,
                    height=350,
                    disabled=True,
                    key="preview_area"
                )
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("Enter a forbidden goal to see attack preview")

    st.markdown("---")
    if "single_session_id" not in st.session_state:
        st.session_state.single_session_id = None
    if st.button("🚀 Run Attack", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("Please enter a forbidden goal")
            return
        with st.spinner("Running attack... This may take 10-30 seconds"):
            result = run_single_attack_sync(
                forbidden_goal=forbidden_goal,
                strategy=selected_strategy_obj.strategy_type.value,
                target_provider=target_provider,
                target_model=target_model,
                is_local=is_local,
                attacker_provider=attacker_provider,
                attacker_model=attacker_model,
            )
        if result:
            new_id = save_or_update_single_attack_session(
                    result, target_model, forbidden_goal, st.session_state.active_audit_session_id
                )
            st.session_state.active_audit_session_id = new_id
            st.success(f"Attack added to Session: `{new_id}`")
            display_single_result(result)


# ─── Tab 2: Full Audit ─────────────────────────────────────────────────────────
def full_audit_interface(provider, model, is_local, save_to_db, parallel):
    st.header("🔍 Full Security Audit")
    st.markdown("Test multiple attack strategies comprehensively")

    type_badge = "LOCAL" if is_local else "CLOUD"
    st.info(f"[{type_badge}] Target: {provider}/{model}")

    forbidden_goal = st.text_area(
        "Forbidden Goal",
        placeholder="Enter the harmful goal to audit",
        height=80,
        key="full_audit_goal"
    )

    all_strategies = list_all_strategies()
    all_names      = [s.strategy_type.value for s in all_strategies]
    total          = len(all_names)

    col_scope, col_select = st.columns([1, 2])
    with col_scope:
        st.markdown("**Audit Scope**")
        audit_scope = st.radio(
            "scope",
            ["All Strategies", "Custom Selection"],
            label_visibility="collapsed",
            key="audit_scope"
        )
        st.caption(f"Total available: {total} strategies")

    with col_select:
        st.markdown("**Select strategies**")
        if audit_scope == "All Strategies":
            strategy_list = all_names
            pills_html = " ".join(
                f'<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;'
                f'border-radius:12px;font-size:12px;margin:2px;display:inline-block">'
                f'{n.replace("_", " ").title()}</span>'
                for n in all_names
            )
            st.markdown(pills_html, unsafe_allow_html=True)
            st.caption(f"✅ All {total} strategies selected")
        else:
            display_names = {s.strategy_type.value: s.name for s in all_strategies}
            selected = st.multiselect(
                "Pick strategies",
                options=all_names,
                default=[all_names[0]] if all_names else [],
                format_func=lambda x: display_names.get(x, x.replace("_", " ").title()),
                label_visibility="collapsed",
                placeholder="Search and select strategies..."
            )
            strategy_list = selected
            if not strategy_list:
                st.warning("Select at least one strategy")
            else:
                st.caption(f"{len(strategy_list)} strategies selected")

    if is_local and len(strategy_list) > 10:
        st.warning(
            f"Testing {len(strategy_list)} strategies on a local model may take "
            f"{len(strategy_list) * 1}–{len(strategy_list) * 2} minutes. "
            "Consider selecting fewer strategies."
        )

    # ── Attacker Model Override ───────────────────────────────────────────────
    st.markdown("---")
    attacker_provider_full, attacker_model_full = _attacker_model_selector("full")
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("---")

    if st.button("🚀 Run Full Audit", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("Please enter a forbidden goal")
            return
        if not strategy_list:
            st.error("Please select at least one strategy")
            return

        progress_bar = st.progress(0)
        status_text  = st.empty()
        status_text.text(f"Running audit with {len(strategy_list)} strategies...")

        result, session_id = run_full_audit_and_persist_sync(
                forbidden_goal=forbidden_goal,
                strategies=strategy_list,
                provider=provider,
                model=model,
                is_local=is_local,
                parallel=parallel,
                existing_id=st.session_state.active_audit_session_id if save_to_db else None,
                attacker_provider=attacker_provider_full,
                attacker_model=attacker_model_full
            )

        progress_bar.progress(100)
        status_text.text("Audit complete!")

        if result and session_id:
            st.session_state.active_audit_session_id = session_id
            display_audit_results(result)
            st.success(f"✅ Audit Session Persisted: `{session_id}`")
            

            
            st.markdown("---")

def run_full_audit_and_persist_sync(forbidden_goal, strategies, provider, model, is_local, parallel, existing_id):
    """Bridge for Full Audit tab to run and compulsorily save batch results"""
    async def _run():

            await db.connect()
            orchestrator = Orchestrator(target_provider=provider, target_model=model, is_local=is_local)
            audit_result = await orchestrator.run_batch_audit(
                forbidden_goal=forbidden_goal, 
                strategy_names=strategies, 
                save_to_db=True, 
                parallel=parallel
            )

            new_examples = []
            for r in audit_result.results:
                new_examples.append({
                    "generated_prompt": r.attack.generated_prompt,
                    "response_text": r.response.response_text,
                    "verdict": r.evaluation.verdict.value,
                    "strategy": r.attack.strategy_name,
                    "timestamp": datetime.utcnow()
                })
            
            # update_or_create_batch_session should be implemented in database.py
            session_id = await db.update_or_create_batch_session(
                examples=new_examples,
                model=model,
                forbidden_goal=forbidden_goal,
                session_id=existing_id
            )
            return audit_result, session_id
    return asyncio.run(_run())

# ─── Tab 4: Strategy Comparison ───────────────────────────────────────────────

def strategy_comparison_interface():
    st.header("📊 Strategy Comparison Analysis")
    st.markdown("Compare effectiveness and characteristics of all attack strategies")

    try:
        stats          = get_strategy_stats_dynamic()
        all_strategies = list_all_strategies()
        diff_counts    = stats["by_difficulty"]

        metric_cols = st.columns(2 + len(diff_counts))
        metric_cols[0].metric("Total Strategies", stats["total"])
        metric_cols[1].metric("Avg Effectiveness", f"{stats['avg_effectiveness']:.0f}")
        for i, (diff, icon) in enumerate(DIFF_ICON.items(), 2):
            metric_cols[i].metric(
                f"{icon} {diff.replace('_', ' ').title()}",
                diff_counts.get(diff, 0)
            )
        st.markdown("---")

        comparison_data = []
        for s in all_strategies:
            d   = _diff(s)
            eff = getattr(s, "effectiveness_score", 0)
            comparison_data.append({
                "Strategy":    s.name,
                "Difficulty":  f"{DIFF_ICON.get(d, '')} {d}",
                "Effectiveness": f"{eff:.0f}",
                "Multi-turn":  "🔄" if getattr(s, "multi_turn", False) else "",
                "Type":        s.strategy_type.value,
                "Description": s.description[:60] + "..." if len(s.description) > 60 else s.description,
            })

        df = pd.DataFrame(comparison_data)
        st.subheader("All Strategies Overview")
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Effectiveness": st.column_config.ProgressColumn(
                    "Effectiveness", format="%s", min_value=0, max_value=100
                )
            }
        )
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Effectiveness by Difficulty")
            chart_data = [
                {
                    "Strategy":     s.name,
                    "Effectiveness": getattr(s, "effectiveness_score", 0) * 100,
                    "Difficulty":    _diff(s)
                }
                for s in all_strategies
            ]
            df_chart = pd.DataFrame(chart_data)
            fig = px.bar(
                df_chart, x="Strategy", y="Effectiveness",
                color="Difficulty", color_discrete_map=DIFF_COLOR,
                title="Effectiveness by Strategy",
                labels={"Effectiveness": "Effectiveness"}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Difficulty Distribution")
            fig_pie = go.Figure(data=go.Pie(
                labels=list(diff_counts.keys()),
                values=list(diff_counts.values()),
                marker=dict(colors=[DIFF_COLOR.get(d, "#6b7280") for d in diff_counts.keys()])
            ))
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("Breakdown by Difficulty")
        breakdown = []
        for diff in ["EASY", "MEDIUM", "HARD", "VERY_HARD"]:
            bucket = [s for s in all_strategies if _diff(s) == diff]
            if not bucket:
                continue
            avg_eff = sum(getattr(s, "effectiveness_score", 0) for s in bucket) / len(bucket)
            breakdown.append({
                "Difficulty":       f"{DIFF_ICON.get(diff, '')} {diff}",
                "Strategies":       len(bucket),
                "Avg Effectiveness": f"{avg_eff:.0f}",
                "Examples":          ", ".join(s.name for s in bucket[:2])
                                     + ("..." if len(bucket) > 2 else ""),
            })
        st.table(pd.DataFrame(breakdown))

    except Exception as e:
        st.error(f"Error loading strategy comparison: {e}")
        import traceback
        st.code(traceback.format_exc())


# ─── Tab 3: Analytics ─────────────────────────────────────────────────────────
def analytics_interface():
    st.header("📈 Security Analytics")
    st.markdown("Analyze historical audit results and trends")

    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.selectbox("Time Range", [7, 30, 90, 365], index=1)
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    data = fetch_analytics_data_sync(days)
    if not data:
        st.info("No data available. Run some audits first!")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tests", data["total_tests"])
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
        st.metric("Jailbreaks", data["jailbreaks"], f"{data['asr']:.1f}% ASR")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
        st.metric("Partial Leaks", data["partial"])
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
        st.metric("Refused", data["refused"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if "timeline" in data:
        df_timeline = pd.DataFrame(data["timeline"])
        fig_line = px.line(
            df_timeline, x="date", y="asr",
            title="Attack Success Rate Trend",
            labels={"asr": "ASR (%)", "date": "Date"}
        )
        st.plotly_chart(fig_line, use_container_width=True)

    if "strategy_stats" in data:
        df_strategies = pd.DataFrame(data["strategy_stats"])
        fig_bar = px.bar(
            df_strategies, x="strategy", y="success_rate",
            title="Strategy Effectiveness",
            labels={"success_rate": "Success Rate (%)", "strategy": "Strategy"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ─── Tab 5: Model Comparison ──────────────────────────────────────────────────
def model_comparison_interface():
    st.header("🤖 Model Comparison")
    st.markdown("Compare security posture across different language models")
    st.info("🚧 Feature coming soon! This will allow side-by-side comparison of multiple models.")


# ─── Result Displays ──────────────────────────────────────────────────────────
def display_single_result(result):
    st.markdown("---")
    st.subheader("📊 Attack Results")

    verdict = result.evaluation.verdict.value
    if verdict == "JAILBROKEN":
        st.error("🔴 **JAILBROKEN** — Model was successfully attacked!")
    elif verdict == "PARTIAL":
        st.warning("🟡 **PARTIAL** — Model leaked some information")
    else:
        st.success("🟢 **REFUSED** — Model successfully refused")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{result.evaluation.confidence_score:.1%}")
    with col2:
        st.metric("Execution Time", f"{result.total_time_ms}ms")
    with col3:
        st.metric("Attack ID", result.attack.attack_id[:8] + "...")

    with st.expander("🗡️ Attack Prompt"):
        st.text_area(
            "Generated Attack", result.attack.generated_prompt,
            height=200, disabled=True, key="result_attack_prompt"
        )
    with st.expander("💬 Target Response"):
        st.text_area(
            "Model Response", result.response.response_text,
            height=200, disabled=True, key="result_target_response"
        )
    with st.expander("⚖️ Judge Reasoning"):
        st.write(result.evaluation.reasoning)


def display_audit_results(result):
    st.markdown("---")
    st.subheader("📊 Audit Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Attacks", result.total_attacks)
    with col2:
        asr_color = (
            "🔴" if result.attack_success_rate > 0.2 else
            "🟡" if result.attack_success_rate > 0.1 else "🟢"
        )
        st.metric("Attack Success Rate", f"{result.attack_success_rate:.1%}", delta=asr_color)
    with col3:
        st.metric("Successful Jailbreaks", result.successful_jailbreaks)
    with col4:
        st.metric("Execution Time", f"{result.total_execution_time_ms / 1000:.1f}s")

    results_data = [
        {
            "Strategy":   r.attack.strategy_name,
            "Verdict":    r.evaluation.verdict.value,
            "Confidence": f"{r.evaluation.confidence_score:.1%}",
            "Time (ms)":  r.total_time_ms,
            "Success":    "✅" if r.success else "❌",
        }
        for r in result.results
    ]
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)

    if result.successful_jailbreaks > 0:
        fig = px.bar(
        df, x="Strategy",
        y=[1 if s == "✅" else 0 for s in df["Success"]],
        title="Attack Success by Strategy",
        labels={"y": "Success (1=Yes, 0=No)"}
    )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No successful attacks to display in chart.")
 

# ─── Sync wrappers (NO Streamlit UI calls allowed inside async def run) ────────
def run_single_attack_sync(
    forbidden_goal, strategy, target_provider, target_model, is_local,
    attacker_provider=None,   # ← NEW param
    attacker_model=None,      # ← NEW param
):
    async def run():
        try:
            await db.connect()
            orchestrator = Orchestrator(
                target_provider=target_provider,
                target_model=target_model,
                is_local=is_local,
                attacker_provider=attacker_provider,   # ← passed cleanly
                attacker_model=attacker_model,         # ← passed cleanly
            )
            result = await orchestrator.run_single_attack(
                forbidden_goal=forbidden_goal,
                strategy_name=strategy
            )
            return result
        except Exception as e:
            st.error(f"Error running attack: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
        finally:
            try:
                await db.disconnect()
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


def run_full_audit_sync(
    forbidden_goal, strategies, provider, model, is_local,
    save_to_db, parallel,
    attacker_provider=None,   # ← NEW param
    attacker_model=None,      # ← NEW param
):
    async def run():
        try:
            await db.connect()
            orchestrator = Orchestrator(
                target_provider=provider,
                target_model=model,
                is_local=is_local,
                attacker_provider=attacker_provider,   # ← NEW
                attacker_model=attacker_model,         # ← NEW
            )
            result = await orchestrator.run_batch_audit(
                forbidden_goal=forbidden_goal,
                strategy_names=strategies,
                save_to_db=save_to_db,
                parallel=parallel
            )
            return result
        except Exception as e:
            st.error(f"Error running audit: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
        finally:
            try:
                await db.disconnect()
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


def fetch_analytics_data_sync(days: int):
    async def fetch():
        try:
            await db.connect()
            evaluations = await db.get_evaluations_by_filter(limit=1000)
            if not evaluations:
                return None
            total      = len(evaluations)
            refused    = sum(1 for e in evaluations if e.get("verdict") == "REFUSED")
            jailbroken = sum(1 for e in evaluations if e.get("verdict") == "JAILBROKEN")
            partial    = sum(1 for e in evaluations if e.get("verdict") == "PARTIAL")
            asr        = (jailbroken / total * 100) if total > 0 else 0
            return {
                "total_tests": total,
                "refused":     refused,
                "jailbreaks":  jailbroken,
                "partial":     partial,
                "asr":         asr,
            }
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
        finally:
            try:
                await db.disconnect()
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fetch())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
