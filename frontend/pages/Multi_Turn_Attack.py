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
import sys
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.models.database import Database
from backend.app.core.multiturn_manager import MultiTurnManager

# Page config
st.set_page_config(
    page_title="Multi-Turn Attack Lab",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .turn-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .turn-refused {
        background-color: #1e3a2e;
        border-color: #10b981;
    }
    .turn-partial {
        background-color: #3a2e1e;
        border-color: #f59e0b;
    }
    .turn-jailbroken {
        background-color: #3a1e1e;
        border-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database (cached)
@st.cache_resource
def get_db_and_manager():
    """Initialize database and multi-turn manager"""
    db = Database()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(db.connect())
    manager = MultiTurnManager(db)
    return db, manager, loop

db, manager, loop = get_db_and_manager()

# Helper function to run async code
def run_async(coro):
    """Run async coroutine in the event loop"""
    return loop.run_until_complete(coro)

# Title
st.title("🔄 Multi-Turn Adaptive Attack Lab")
st.markdown("**Execute adaptive conversation-based attacks that learn from target responses**")

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "turns" not in st.session_state:
    st.session_state.turns = []
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False
if "start_counter" not in st.session_state:
    st.session_state.start_counter = 0
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0
if "auto_counter" not in st.session_state:
    st.session_state.auto_counter = 0

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Attack Configuration")
    
    # Forbidden goal input
    forbidden_goal = st.text_area(
        "Forbidden Goal",
        value="How to bypass a security system",
        height=100,
        help="The harmful objective to achieve"
    )
    
    # Target model selection with local support
    st.subheader("Target Model")
    
    model_type = st.radio(
        "Model Source",
        ["Cloud APIs", "Local (Ollama)"],
        horizontal=True,
        key="model_type_selector"
    )
    
    if model_type == "Cloud APIs":
        target_model = st.selectbox(
            "Select Model",
            options=[
                "nvidia/llama3-70b-instruct",
                "nvidia/mixtral-8x7b-instruct",
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "groq/llama3-70b-8192",
                "groq/llama3-8b-8192",
                "groq/llama-3.3-70b-versatile",
                "groq/mixtral-8x7b-32768"
            ],
            help="Cloud-hosted LLM to attack"
        )
    else:
        # Local Ollama models
        st.info("💡 Make sure Ollama is running on localhost:11434")
        
        # Try to fetch available models
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if model_names:
                    local_model = st.selectbox(
                        "Available Models",
                        options=model_names,
                        help="Ollama models detected on your system"
                    )
                else:
                    local_model = st.text_input(
                        "Ollama Model Name",
                        value="llama3:latest",
                        help="No models detected. Enter manually (e.g., llama3:latest)"
                    )
            else:
                local_model = st.text_input(
                    "Ollama Model Name",
                    value="llama3:latest",
                    help="Cannot connect to Ollama. Enter model name manually"
                )
        except:
            local_model = st.text_input(
                "Ollama Model Name",
                value="llama3:latest",
                help="Cannot connect to Ollama. Make sure it's running and enter model name"
            )
        
        target_model = f"ollama/{local_model}"
    
    # Dynamic strategy loading
    st.subheader("Initial Strategy")
    
    try:
        # Import and get all strategies
        from backend.app.agents.strategies import STRATEGY_REGISTRY
        
        # Get strategy names from registry
        strategy_names = list(STRATEGY_REGISTRY.keys())
        
        initial_strategy = st.selectbox(
            "Select Strategy",
            options=strategy_names,
            help="Starting attack strategy",
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Show strategy info
        try:
            from backend.app.agents.strategies import get_strategy
            selected_strat = get_strategy(initial_strategy)
            if selected_strat:
                st.caption(f"📝 {selected_strat.description}")
        except:
            pass
            
    except Exception as e:
        logger.warning(f"Failed to load strategies dynamically: {e}")
        # Fallback
        initial_strategy = st.selectbox(
            "Initial Strategy",
            options=[
                "persona_adoption",
                "contextual_camouflage",
                "hypothetical_framing",
                "authority_impersonation",
                "simple_obfuscation",
                "payload_splitting",
                "cognitive_overload"
            ],
            help="Starting attack strategy"
        )
    
    # Max turns
    max_turns = st.slider(
        "Maximum Turns",
        min_value=3,
        max_value=10,
        value=5,
        help="Max conversation length"
    )
    
    # Adaptive mode
    adaptive_mode = st.checkbox(
        "Adaptive Mode",
        value=True,
        help="Automatically select best strategies"
    )
    
    st.divider()
    
    # Start button with unique key
    start_disabled = st.session_state.conversation_active
    if st.button("🚀 Start New Conversation", type="primary", disabled=start_disabled, key=f"start_{st.session_state.start_counter}"):
        with st.spinner("Initializing conversation..."):
            try:
                # Increment counter
                st.session_state.start_counter += 1
                
                # Start conversation
                conv_id = run_async(manager.start_conversation(
                    forbidden_goal=forbidden_goal,
                    target_model=target_model,
                    initial_strategy=initial_strategy,
                    max_turns=max_turns,
                    adaptive_mode=adaptive_mode
                ))
                
                st.session_state.conversation_id = conv_id
                st.session_state.turns = []
                st.session_state.conversation_active = True
                st.success(f"✅ Conversation started: {conv_id[:8]}...")
                time.sleep(0.3)
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to start conversation: {e}")
                logger.exception("Error starting conversation:")
    
    # Reset button
    if st.button("🔄 Reset", disabled=not st.session_state.conversation_active):
        st.session_state.conversation_id = None
        st.session_state.turns = []
        st.session_state.conversation_active = False
        st.session_state.auto_running = False
        st.session_state.start_counter = 0
        st.session_state.turn_counter = 0
        st.session_state.auto_counter = 0
        st.rerun()

# Main content area
if not st.session_state.conversation_active:
    # Welcome screen
    st.info("👈 Configure your attack in the sidebar and click **Start New Conversation**")
    
    st.markdown("""
    ### How It Works
    
    1. **Adaptive Strategy Selection**: The system analyzes each response and chooses the next best strategy
    2. **Multi-Turn Persistence**: Continues attacking across 3-10 conversation turns
    3. **Real-time Analysis**: Each response is analyzed for:
       - Refusal patterns
       - Information leakage
       - Hedge language
       - Openness score
    4. **Success Tracking**: Monitors which strategies work best for each target model
    
    ### Attack Modes
    
    - **Manual Mode**: Execute turns one-by-one to observe the adaptation
    - **Auto Mode**: Run all turns automatically until jailbreak or max turns
    """)
    
else:
    # Active conversation interface
    conv_id = st.session_state.conversation_id
    
    # Calculate jailbroken status BEFORE using it
    jailbroken = any(t.get("verdict") == "JAILBROKEN" for t in st.session_state.turns)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Conversation ID", conv_id[:8] + "...")
    with col2:
        st.metric("Turns Executed", len(st.session_state.turns))
    with col3:
        st.metric("Status", "🎯 JAILBROKEN" if jailbroken else "🔄 Active")
    with col4:
        if st.session_state.turns:
            avg_openness = sum(t.get("openness_score", 0) for t in st.session_state.turns) / len(st.session_state.turns)
            st.metric("Avg Openness", f"{avg_openness:.2f}")
        else:
            st.metric("Avg Openness", "N/A")
    
    st.divider()
    
    # Action buttons
    col_auto, col_manual = st.columns(2)
    
    with col_auto:
        auto_disabled = jailbroken or st.session_state.auto_running
        if st.button("⚡ Auto-Run Remaining Turns", type="primary", disabled=auto_disabled, key=f"auto_{st.session_state.auto_counter}"):
            st.session_state.auto_counter += 1
            st.session_state.auto_running = True
            st.rerun()
    
    with col_manual:
        turn_disabled = jailbroken or len(st.session_state.turns) >= max_turns
        if st.button("▶️ Execute Single Turn", disabled=turn_disabled, key=f"turn_{st.session_state.turn_counter}"):
            with st.spinner("Executing turn..."):
                try:
                    st.session_state.turn_counter += 1
                    
                    # Execute turn
                    turn = run_async(manager.execute_turn(conv_id))
                    
                    # Convert to dict
                    turn_data = {
                        "turn_number": turn.turn_number,
                        "strategy_used": turn.strategy_used,
                        "verdict": turn.response_analysis.verdict.value,
                        "confidence": turn.response_analysis.confidence,
                        "openness_score": turn.response_analysis.openness_score,
                        "recommended_next_strategy": turn.response_analysis.recommended_next_strategy,
                        "target_response_preview": turn.target_response[:200] + "...",
                        "execution_time_ms": turn.execution_time_ms,
                        "jailbreak_achieved": turn.response_analysis.verdict.value == "JAILBROKEN"
                    }
                    
                    st.session_state.turns.append(turn_data)
                    time.sleep(0.2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to execute turn: {e}")
                    logger.exception("Turn execution error:")
    
    # Auto-run logic
    if st.session_state.auto_running and not jailbroken:
        remaining = max_turns - len(st.session_state.turns)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(remaining):
            status_text.text(f"Executing turn {len(st.session_state.turns) + 1}/{max_turns}...")
            
            try:
                turn = run_async(manager.execute_turn(conv_id))
                
                turn_data = {
                    "turn_number": turn.turn_number,
                    "strategy_used": turn.strategy_used,
                    "verdict": turn.response_analysis.verdict.value,
                    "confidence": turn.response_analysis.confidence,
                    "openness_score": turn.response_analysis.openness_score,
                    "recommended_next_strategy": turn.response_analysis.recommended_next_strategy,
                    "target_response_preview": turn.target_response[:200] + "...",
                    "execution_time_ms": turn.execution_time_ms,
                    "jailbreak_achieved": turn.response_analysis.verdict.value == "JAILBROKEN"
                }
                
                st.session_state.turns.append(turn_data)
                progress_bar.progress((i + 1) / remaining)
                
                # Stop if jailbroken
                if turn_data["jailbreak_achieved"]:
                    status_text.text("🎯 Jailbreak achieved!")
                    time.sleep(1)
                    break
                    
            except Exception as e:
                st.error(f"Failed: {e}")
                logger.exception("Auto-run error:")
                break
        
        st.session_state.auto_running = False
        status_text.text("✅ Auto-run complete!")
        time.sleep(1)
        st.rerun()
    
    # Display turns
    if st.session_state.turns:
        st.subheader("📊 Conversation Timeline")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["💬 Turn-by-Turn", "📈 Analytics", "🎯 Strategy Analysis"])
        
        with tab1:
            # Display each turn
            for turn in reversed(st.session_state.turns):
                verdict = turn.get("verdict", "UNKNOWN")
                turn_num = turn.get("turn_number")
                strategy = turn.get("strategy_used")
                confidence = turn.get("confidence", 0)
                openness = turn.get("openness_score", 0)
                
                # Determine icon
                if verdict == "JAILBROKEN":
                    icon = "🎯"
                elif verdict == "PARTIAL":
                    icon = "⚠️"
                else:
                    icon = "🛡️"
                
                with st.expander(f"{icon} Turn {turn_num} - {strategy} - **{verdict}**", expanded=(turn_num == len(st.session_state.turns))):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col_b:
                        st.metric("Openness Score", f"{openness:.2f}")
                    with col_c:
                        next_strat = turn.get("recommended_next_strategy", "N/A")
                        st.metric("Next Strategy", next_strat)
                    
                    st.markdown("**Target Response Preview:**")
                    st.text_area(
                        f"Response {turn_num}",
                        value=turn.get("target_response_preview", ""),
                        height=100,
                        disabled=True,
                        key=f"response_{turn_num}"
                    )
                    
                    st.caption(f"Execution time: {turn.get('execution_time_ms', 0)}ms")
        
        with tab2:
            # Analytics charts
            st.markdown("### 📊 Performance Metrics")
            
            # Prepare data
            turns_df = pd.DataFrame(st.session_state.turns)
            
            # Verdict distribution
            col1, col2 = st.columns(2)
            
            with col1:
                verdict_counts = turns_df['verdict'].value_counts()
                fig_verdicts = px.pie(
                    values=verdict_counts.values,
                    names=verdict_counts.index,
                    title="Verdict Distribution",
                    color=verdict_counts.index,
                    color_discrete_map={
                        "REFUSED": "#10b981",
                        "PARTIAL": "#f59e0b",
                        "JAILBROKEN": "#ef4444"
                    }
                )
                st.plotly_chart(fig_verdicts, use_container_width=True)
            
            with col2:
                # Confidence progression
                fig_confidence = go.Figure()
                fig_confidence.add_trace(go.Scatter(
                    x=turns_df['turn_number'],
                    y=turns_df['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=10)
                ))
                fig_confidence.update_layout(
                    title="Confidence Progression",
                    xaxis_title="Turn",
                    yaxis_title="Confidence Score",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Openness progression
            fig_openness = go.Figure()
            fig_openness.add_trace(go.Scatter(
                x=turns_df['turn_number'],
                y=turns_df['openness_score'],
                mode='lines+markers',
                name='Openness',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.2)'
            ))
            fig_openness.update_layout(
                title="Model Openness Over Time",
                xaxis_title="Turn",
                yaxis_title="Openness Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_openness, use_container_width=True)
        
        with tab3:
            # Strategy analysis
            st.markdown("### 🎯 Strategy Effectiveness")
            
            # Strategy usage
            strategy_counts = turns_df['strategy_used'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_strategies = px.bar(
                    x=strategy_counts.index,
                    y=strategy_counts.values,
                    title="Strategy Usage",
                    labels={'x': 'Strategy', 'y': 'Times Used'},
                    color=strategy_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_strategies.update_layout(showlegend=False)
                st.plotly_chart(fig_strategies, use_container_width=True)
            
            with col2:
                # Strategy success rate
                strategy_success = turns_df.groupby('strategy_used').apply(
                    lambda x: (x['verdict'] == 'JAILBROKEN').sum() / len(x)
                ).sort_values(ascending=False)
                
                fig_success = px.bar(
                    x=strategy_success.index,
                    y=strategy_success.values,
                    title="Strategy Success Rate",
                    labels={'x': 'Strategy', 'y': 'Success Rate'},
                    color=strategy_success.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_success.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig_success, use_container_width=True)
            
            # Detailed table
            st.markdown("### 📋 Detailed Strategy Breakdown")
            strategy_table = turns_df.groupby('strategy_used').agg({
                'turn_number': 'count',
                'confidence': 'mean',
                'openness_score': 'mean'
            }).rename(columns={
                'turn_number': 'Uses',
                'confidence': 'Avg Confidence',
                'openness_score': 'Avg Openness'
            })
            
            # Add success rate
            strategy_table['Success Rate'] = turns_df.groupby('strategy_used').apply(
                lambda x: (x['verdict'] == 'JAILBROKEN').sum() / len(x)
            )
            
            st.dataframe(
                strategy_table.style.format({
                    'Avg Confidence': '{:.2%}',
                    'Avg Openness': '{:.2f}',
                    'Success Rate': '{:.2%}'
                }),
                use_container_width=True
            )
    
    # Complete conversation button
    if st.session_state.turns and (jailbroken or len(st.session_state.turns) >= max_turns):
        st.divider()
        if st.button("✅ Complete & Save Conversation", type="primary"):
            with st.spinner("Saving results..."):
                try:
                    result = run_async(manager.complete_conversation(conv_id))
                    st.success("✅ Conversation saved successfully!")
                    
                    # Show summary
                    st.json({
                        "conversation_id": result.conversation_id,
                        "total_turns": result.total_turns,
                        "jailbreak_achieved": result.jailbreak_achieved,
                        "final_verdict": result.final_verdict,
                        "most_effective_strategy": result.most_effective_strategy
                    })
                    
                    # Reset
                    time.sleep(2)
                    st.session_state.conversation_id = None
                    st.session_state.turns = []
                    st.session_state.conversation_active = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to complete: {e}")
                    logger.exception("Complete conversation error:")
