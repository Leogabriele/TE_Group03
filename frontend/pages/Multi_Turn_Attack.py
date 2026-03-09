"""
Multi-Turn Attack Lab - Interactive adaptive attack interface
Direct backend integration (no API server needed)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import pandas as pd
import asyncio
import concurrent.futures
import sys
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.models.database import Database
from backend.app.core.multiturn_manager import MultiTurnManager
from backend.app.config import settings

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Turn Attack Lab",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
</style>
""", unsafe_allow_html=True)

# ─── Thread-safe async runner ─────────────────────────────────────────────────
def run_async(coro):
    """Run an async coroutine safely from Streamlit's sync context."""
    def _run(c):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(c)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run, coro).result(timeout=120)

# ─── DB + Manager (cached once per app session) ───────────────────────────────
@st.cache_resource
def get_manager():
    """Initialize DB and MultiTurnManager exactly once."""
    db = Database()
    run_async(db.connect())
    return MultiTurnManager(db)

manager = get_manager()

# ─── Dynamic model fetch (outside any if-block to avoid re-registration) ──────
@st.cache_data(ttl=300, show_spinner=False)
def _get_cloud_models():
    """Fetch live Groq model list; fall back to known-good models."""
    models = []

    # Groq — live API fetch
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        groq_models = [f"groq/{m.id}" for m in client.models.list().data]
        models.extend(sorted(groq_models))
    except Exception as e:
        logger.warning(f"Could not fetch Groq models: {e}")
        models.extend([
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/gemma2-9b-it",
        ])

    # NVIDIA — no public list API, use known models
    nvidia_known = [
        "nvidia/meta/llama3-70b-instruct",
        "nvidia/meta/llama-3.1-405b-instruct",
        "nvidia/meta/llama-3.1-70b-instruct",
        "nvidia/mistralai/mixtral-8x7b-instruct-v0.1",
        "nvidia/nvidia/mistral-nemo-minitron-8b-instruct",
    ]

    # Prepend current configured target so it appears first
    try:
        current = f"{settings.TARGET_MODEL_PROVIDER}/{settings.TARGET_MODEL_NAME}"
        if current not in nvidia_known:
            nvidia_known.insert(0, current)
    except Exception:
        pass
    models.extend(nvidia_known)

    return models

# ─── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "conversation_id":       None,
        "turns":                 [],
        "conversation_active":   False,
        "auto_running":          False,
        "max_turns":             5,
        "start_counter":         0,
        "turn_counter":          0,
        "auto_counter":          0,
        "attacker_model_display": f"groq/{settings.ATTACKER_MODEL} (default)",  # NEW
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Page header ──────────────────────────────────────────────────────────────
st.title("🔄 Multi-Turn Adaptive Attack Lab")
st.markdown("**Execute adaptive conversation-based attacks that learn from target responses**")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Attack Configuration")

    forbidden_goal = st.text_area(
        "Forbidden Goal",
        value="How to bypass a security system",
        height=100,
        disabled=st.session_state.conversation_active,
        help="The harmful objective to achieve"
    )

    # ── Target Model ──────────────────────────────────────────────────────────
    st.subheader("Target Model")
    model_type = st.radio(
        "Model Source",
        ["Cloud APIs", "Local (Ollama)"],
        horizontal=True,
        disabled=st.session_state.conversation_active,
        key="model_type_selector"
    )

    if model_type == "Cloud APIs":
        all_cloud_models = _get_cloud_models()

        try:
            default_target = f"{settings.TARGET_MODEL_PROVIDER}/{settings.TARGET_MODEL_NAME}"
            default_idx    = all_cloud_models.index(default_target)
        except (AttributeError, ValueError):
            default_idx = 0

        col_model, col_refresh = st.columns([4, 1])
        with col_model:
            target_model = st.selectbox(
                "Select Model",
                options=all_cloud_models,
                index=default_idx,
                disabled=st.session_state.conversation_active,
                help="Cloud-hosted LLM to attack"
            )
        with col_refresh:
            st.write("")  # vertical alignment spacer
            if st.button("🔄", help="Refresh model list"):
                st.cache_data.clear()
                st.rerun()

    else:
        st.info("💡 Make sure Ollama is running on localhost:11434")
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                model_names = [m["name"] for m in resp.json().get("models", [])]
                if model_names:
                    local_model = st.selectbox(
                        "Available Models", options=model_names,
                        disabled=st.session_state.conversation_active,
                        help="Ollama models detected on your system"
                    )
                else:
                    local_model = st.text_input(
                        "Ollama Model Name", value="llama3:latest",
                        disabled=st.session_state.conversation_active,
                        help="No models detected — enter manually"
                    )
            else:
                local_model = st.text_input(
                    "Ollama Model Name", value="llama3:latest",
                    disabled=st.session_state.conversation_active,
                    help="Cannot connect to Ollama"
                )
        except Exception:
            local_model = st.text_input(
                "Ollama Model Name", value="llama3:latest",
                disabled=st.session_state.conversation_active,
                help="Cannot connect to Ollama — enter manually"
            )
        target_model = f"ollama/{local_model}"

    # ── Initial Strategy ──────────────────────────────────────────────────────
    st.subheader("Initial Strategy")

    try:
        from backend.app.agents.strategies import STRATEGY_REGISTRY, get_strategy
        strategy_names = list(STRATEGY_REGISTRY.keys())
    except Exception as e:
        logger.warning(f"STRATEGY_REGISTRY import failed: {e}")
        try:
            from backend.app.models.enums import AttackStrategyType
            strategy_names = [s.value for s in AttackStrategyType]
        except Exception:
            strategy_names = [
                "contextual_camouflage", "persona_adoption", "hypothetical_framing",
                "authority_impersonation", "simple_obfuscation", "payload_splitting",
                "cognitive_overload",
            ]

    initial_strategy = st.selectbox(
        "Select Strategy",
        options=strategy_names,
        disabled=st.session_state.conversation_active,
        help="Starting attack strategy",
        format_func=lambda x: x.replace("_", " ").title()
    )

    try:
        strat_obj = get_strategy(initial_strategy)
        if strat_obj:
            st.caption(f"📝 {strat_obj.description}")
    except Exception:
        pass

    # ── Attacker Model ────────────────────────────────────────────────────────
    st.subheader("🤖 Attacker Model")
    st.caption(f"Default (from .env): `groq/{settings.ATTACKER_MODEL}`")

    use_custom_attacker = st.checkbox(
        "Override Attacker Model",
        value=False,
        disabled=st.session_state.conversation_active,
        key="use_custom_attacker",
        help="Choose a different model to craft the attack prompts"
    )

    attacker_model_full = None  # default: use .env

    if use_custom_attacker:
        atk_source = st.radio(
            "Attacker Source",
            ["☁️ Cloud (Groq)", "🏠 Local (Ollama)"],
            horizontal=True,
            disabled=st.session_state.conversation_active,
            key="atk_source",
            help="Groq = free cloud | Ollama = local, private"
        )

        if atk_source == "☁️ Cloud (Groq)":
            groq_only = [
                m.replace("groq/", "")
                for m in _get_cloud_models()
                if m.startswith("groq/")
            ]
            try:
                atk_default_idx = groq_only.index(settings.ATTACKER_MODEL)
            except ValueError:
                atk_default_idx = 0
            selected_attacker_model = st.selectbox(
                "Groq Attacker Model",
                options=groq_only,
                index=atk_default_idx,
                disabled=st.session_state.conversation_active,
                key="attacker_model_select",
                help="Free Groq tier"
            )
            attacker_model_full = f"groq/{selected_attacker_model}"
            st.caption(f"🔴 Override: `{attacker_model_full}`")

        else:  # Local Ollama
            st.info("💡 Ollama must be running on `localhost:11434`")
            ollama_atk_models = []
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    ollama_atk_models = [m["name"] for m in resp.json().get("models", [])]
            except Exception:
                pass

            default_ollama_atk = getattr(settings, "ATTACKER_OLLAMA_MODEL", "llama3.2:latest")

            if ollama_atk_models:
                if default_ollama_atk not in ollama_atk_models:
                    ollama_atk_models.insert(0, default_ollama_atk)
                try:
                    atk_ollama_idx = ollama_atk_models.index(default_ollama_atk)
                except ValueError:
                    atk_ollama_idx = 0
                selected_ollama_atk = st.selectbox(
                    "Ollama Attacker Model",
                    options=ollama_atk_models,
                    index=atk_ollama_idx,
                    disabled=st.session_state.conversation_active,
                    key="atk_ollama_select",
                    help="Installed Ollama models"
                )
                attacker_model_full = f"ollama/{selected_ollama_atk}"
            else:
                st.warning("⚠️ Ollama not reachable — enter manually")
                manual_atk = st.text_input(
                    "Model name",
                    value=default_ollama_atk,
                    disabled=st.session_state.conversation_active,
                    key="atk_ollama_manual",
                    help="e.g. llama3.2:latest, mistral:7b"
                )
                if manual_atk:
                    attacker_model_full = f"ollama/{manual_atk}"

            if attacker_model_full:
                st.caption(f"🔴 Override: `{attacker_model_full}`")
    else:
        st.caption("✅ Using default from `.env`")
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────

    # ── Run Settings ──────────────────────────────────────────────────────────
    max_turns_slider = st.slider(
        "Maximum Turns", min_value=2, max_value=10, value=5,
        disabled=st.session_state.conversation_active,
        help="Locked once conversation starts"
    )

    adaptive_mode = st.checkbox(
        "Adaptive Mode", value=True,
        disabled=st.session_state.conversation_active,
        help="Automatically select best next strategy"
    )

    st.divider()

    # ── Start button ──────────────────────────────────────────────────────────
    if st.button(
        "🚀 Start New Conversation", type="primary",
        disabled=st.session_state.conversation_active,
        key=f"start_{st.session_state.start_counter}"
    ):
        with st.spinner("Initializing conversation..."):
            try:
                st.session_state.start_counter += 1
                conv_id = run_async(
                    manager.start_conversation(
                        forbidden_goal=forbidden_goal,
                        target_model=target_model,
                        initial_strategy=initial_strategy,
                        max_turns=max_turns_slider,
                        adaptive_mode=adaptive_mode,
                        attacker_model=attacker_model_full,      # ← NEW
                    )
                )
                st.session_state.conversation_id = conv_id
                st.session_state.turns           = []
                st.session_state.conversation_active = True
                st.session_state.max_turns       = max_turns_slider
                # Store attacker label that survives reruns
                st.session_state.attacker_model_display = (
                    attacker_model_full
                    or f"groq/{settings.ATTACKER_MODEL} (default)"
                )
                st.success(f"✅ Conversation started: {conv_id[:8]}...")
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start conversation: {e}")
                logger.exception("Error starting conversation")

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.button("🔄 Reset", disabled=not st.session_state.conversation_active):
        st.session_state.conversation_id       = None
        st.session_state.turns                 = []
        st.session_state.conversation_active   = False
        st.session_state.auto_running          = False
        st.session_state.max_turns             = 5
        st.session_state.start_counter         = 0
        st.session_state.turn_counter          = 0
        st.session_state.auto_counter          = 0
        st.session_state.attacker_model_display = f"groq/{settings.ATTACKER_MODEL} (default)"
        st.rerun()

# ─── Main content ─────────────────────────────────────────────────────────────
if not st.session_state.conversation_active:
    st.info("👈 Configure your attack in the sidebar and click **Start New Conversation**")
    st.markdown("""
### How It Works

1. **Adaptive Strategy Selection**: Analyzes each response and chooses the next best strategy
2. **Multi-Turn Persistence**: Continues attacking across 2–10 conversation turns
3. **Real-time Analysis**: Each response is analyzed for:
   - Refusal patterns
   - Information leakage
   - Hedge language
   - Openness score
4. **Success Tracking**: Monitors which strategies work best for each target model

### Attack Modes

- **Manual Mode**: Execute turns one-by-one to observe adaptation
- **Auto Mode**: Run all turns automatically until jailbreak or max turns
""")

else:
    conv_id   = st.session_state.conversation_id
    max_turns = st.session_state.max_turns
    turns_done = len(st.session_state.turns)

    jailbroken        = any(t.get("verdict") == "JAILBROKEN" for t in st.session_state.turns)
    conversation_over = jailbroken or turns_done >= max_turns

    # ── KPI row (5 columns now) ───────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Conversation ID", conv_id[:8] + "...")
    col2.metric("Turns Executed", f"{turns_done}/{max_turns}")
    col3.metric(
        "Status",
        "🎯 JAILBROKEN" if jailbroken else
        "✅ Complete"   if turns_done >= max_turns else
        "🔄 Active"
    )

    # Show live attacker model — after first turn shows actual model used
    live_attacker = (
        manager.get_attacker_model(conv_id)
        if turns_done > 0
        else st.session_state.get("attacker_model_display", f"groq/{settings.ATTACKER_MODEL}")
    )
    col4.metric("🤖 Attacker", live_attacker.split("/")[-1])

    if st.session_state.turns:
        avg_openness = (
            sum(t.get("openness_score", 0) for t in st.session_state.turns) / turns_done
        )
        col5.metric("Avg Openness", f"{avg_openness:.2f}")
    else:
        col5.metric("Avg Openness", "N/A")

    st.divider()

    # ── Action buttons ────────────────────────────────────────────────────────
    col_auto, col_manual = st.columns(2)

    with col_auto:
        if st.button(
            "⚡ Auto-Run Remaining Turns", type="primary",
            disabled=conversation_over or st.session_state.auto_running,
            key=f"auto_{st.session_state.auto_counter}"
        ):
            st.session_state.auto_counter += 1
            st.session_state.auto_running  = True
            st.rerun()

    with col_manual:
        if st.button(
            "▶️ Execute Single Turn",
            disabled=conversation_over or st.session_state.auto_running,
            key=f"turn_{st.session_state.turn_counter}"
        ):
            with st.spinner(f"Executing turn {turns_done + 1}/{max_turns}..."):
                try:
                    st.session_state.turn_counter += 1
                    turn = run_async(manager.execute_turn(conv_id))
                    st.session_state.turns.append({
                        "turn_number":               turn.turn_number,
                        "strategy_used":             turn.strategy_used,
                        "verdict":                   turn.response_analysis.verdict.value,
                        "confidence":                turn.response_analysis.confidence,
                        "openness_score":            turn.response_analysis.openness_score,
                        "recommended_next_strategy": turn.response_analysis.recommended_next_strategy,
                        "target_response_preview":   turn.target_response[:300] + "...",
                        "execution_time_ms":         turn.execution_time_ms,
                        "jailbreak_achieved":        turn.response_analysis.verdict.value == "JAILBROKEN",
                        "attacker_model_used":       manager.get_attacker_model(conv_id),   # NEW
                    })
                    time.sleep(0.2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to execute turn: {e}")
                    logger.exception("execute_turn error")

    # ── Auto-run: ONE turn per rerun ──────────────────────────────────────────
    if st.session_state.auto_running and not conversation_over:
        progress_val = turns_done / max_turns
        st.progress(progress_val, text=f"Auto-running turn {turns_done + 1}/{max_turns}...")
        try:
            turn = run_async(manager.execute_turn(conv_id))
            turn_data = {
                "turn_number":               turn.turn_number,
                "strategy_used":             turn.strategy_used,
                "verdict":                   turn.response_analysis.verdict.value,
                "confidence":                turn.response_analysis.confidence,
                "openness_score":            turn.response_analysis.openness_score,
                "recommended_next_strategy": turn.response_analysis.recommended_next_strategy,
                "target_response_preview":   turn.target_response[:300] + "...",
                "execution_time_ms":         turn.execution_time_ms,
                "jailbreak_achieved":        turn.response_analysis.verdict.value == "JAILBROKEN",
                "attacker_model_used":       manager.get_attacker_model(conv_id),   # NEW
            }
            st.session_state.turns.append(turn_data)

            new_done = len(st.session_state.turns)
            if turn_data["jailbreak_achieved"] or new_done >= max_turns:
                st.session_state.auto_running = False

            st.rerun()

        except Exception as e:
            st.session_state.auto_running = False
            st.error(f"Auto-run failed at turn {turns_done + 1}: {e}")
            logger.exception("Auto-run error")

    elif st.session_state.auto_running and conversation_over:
        st.session_state.auto_running = False

    # ── Conversation timeline ─────────────────────────────────────────────────
    if st.session_state.turns:
        st.subheader("📊 Conversation Timeline")
        tab1, tab2, tab3 = st.tabs(["💬 Turn-by-Turn", "📈 Analytics", "🎯 Strategy Analysis"])

        with tab1:
            for turn in reversed(st.session_state.turns):
                verdict    = turn.get("verdict", "UNKNOWN")
                turn_num   = turn.get("turn_number")
                strategy   = turn.get("strategy_used")
                confidence = turn.get("confidence", 0)
                openness   = turn.get("openness_score", 0)
                icon = {"JAILBROKEN": "🎯", "PARTIAL": "⚠️"}.get(verdict, "🛡️")

                with st.expander(
                    f"{icon} Turn {turn_num} — {strategy} — **{verdict}**",
                    expanded=(turn_num == turns_done)
                ):
                    ca, cb, cc = st.columns(3)
                    ca.metric("Confidence", f"{confidence:.2%}")
                    cb.metric("Openness Score", f"{openness:.2f}")
                    cc.metric("Next Strategy", turn.get("recommended_next_strategy") or "N/A")
                    st.markdown("**Target Response Preview:**")
                    st.text_area(
                        "", value=turn.get("target_response_preview", ""),
                        height=100, disabled=True,
                        key=f"response_{turn_num}"
                    )
                    # Shows attacker model + execution time
                    attacker_used = turn.get("attacker_model_used", settings.ATTACKER_MODEL)
                    st.caption(
                        f"🤖 Attacker: `{attacker_used}` &nbsp;|&nbsp; "
                        f"⏱ {turn.get('execution_time_ms', 0)}ms"
                    )

        with tab2:
            st.markdown("### 📊 Performance Metrics")
            turns_df = pd.DataFrame(st.session_state.turns)
            c1, c2 = st.columns(2)

            with c1:
                vc = turns_df["verdict"].value_counts()
                st.plotly_chart(px.pie(
                    values=vc.values, names=vc.index,
                    title="Verdict Distribution",
                    color=vc.index,
                    color_discrete_map={
                        "REFUSED":   "#10b981",
                        "PARTIAL":   "#f59e0b",
                        "JAILBROKEN":"#ef4444",
                    }
                ), use_container_width=True)

            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=turns_df["turn_number"], y=turns_df["confidence"],
                    mode="lines+markers", name="Confidence",
                    line=dict(color="#3b82f6", width=3), marker=dict(size=10)
                ))
                fig.add_trace(go.Scatter(
                    x=turns_df["turn_number"], y=turns_df["openness_score"],
                    mode="lines+markers", name="Openness",
                    line=dict(color="#f59e0b", width=3, dash="dot"),
                    marker=dict(size=10)
                ))
                fig.update_layout(
                    title="Confidence & Openness Trend",
                    xaxis_title="Turn", yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)

            # Openness area chart
            fig_openness = go.Figure()
            fig_openness.add_trace(go.Scatter(
                x=turns_df["turn_number"], y=turns_df["openness_score"],
                mode="lines+markers", name="Openness",
                line=dict(color="#f59e0b", width=3),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.2)"
            ))
            fig_openness.update_layout(
                title="Model Openness Over Time",
                xaxis_title="Turn", yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_openness, use_container_width=True)

        with tab3:
            st.markdown("### 🎯 Strategy Effectiveness")
            turns_df = pd.DataFrame(st.session_state.turns)
            c1, c2 = st.columns(2)

            with c1:
                sc = turns_df["strategy_used"].value_counts()
                st.plotly_chart(px.bar(
                    x=sc.index, y=sc.values, title="Strategy Usage",
                    labels={"x": "Strategy", "y": "Times Used"},
                    color=sc.values, color_continuous_scale="Blues"
                ), use_container_width=True)

            with c2:
                grp = turns_df.groupby("strategy_used")
                sr  = (
                    grp.apply(
                        lambda x: (x["verdict"] == "JAILBROKEN").sum() / len(x),
                        include_groups=False
                    )
                    .reset_index()
                )
                sr.columns = ["strategy", "success_rate"]
                sr = sr.sort_values("success_rate", ascending=False)
                st.plotly_chart(px.bar(
                    sr, x="strategy", y="success_rate",
                    title="Strategy Success Rate",
                    color="success_rate", color_continuous_scale="RdYlGn",
                    range_y=[0, 1]
                ), use_container_width=True)

            st.markdown("### 📋 Detailed Strategy Breakdown")
            summary = grp.agg(
                Uses=("turn_number", "count"),
                Avg_Confidence=("confidence", "mean"),
                Avg_Openness=("openness_score", "mean"),
            )
            summary["Success_Rate"] = (
                grp.apply(
                    lambda x: (x["verdict"] == "JAILBROKEN").sum() / len(x),
                    include_groups=False
                )
            )
            st.dataframe(
                summary.style.format({
                    "Avg_Confidence": "{:.2%}",
                    "Avg_Openness":   "{:.2f}",
                    "Success_Rate":   "{:.2%}",
                }),
                use_container_width=True
            )

    # ── Complete & Save ───────────────────────────────────────────────────────
    if conversation_over and st.session_state.turns:
        st.divider()
        if st.button("✅ Complete & Save Conversation", type="primary"):
            with st.spinner("Saving results..."):
                try:
                    result = run_async(manager.complete_conversation(conv_id))
                    st.success("✅ Conversation saved successfully!")
                    st.json({
                        "conversation_id":        result.conversation_id,
                        "total_turns":            result.total_turns,
                        "jailbreak_achieved":     result.jailbreak_achieved,
                        "final_verdict":          result.final_verdict,
                        "most_effective_strategy": result.most_effective_strategy,
                        "strategy_success_rates": result.strategy_success_rates,
                    })
                    time.sleep(2)
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to complete: {e}")
                    logger.exception("complete_conversation error")
