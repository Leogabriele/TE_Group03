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

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.api.routes import run_full_audit
from backend.app.core.orchestrator import Orchestrator
from backend.app.agents.strategies import (
    list_all_strategies, 
    get_strategy,
    get_strategies_by_phase,
    get_strategy_stats
)
from backend.app.models.database import db
from backend.app.config import settings

# Page config
st.set_page_config(
    page_title="LLM Security Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left-color: #28a745;
    }
    .warning-card {
        border-left-color: #ffc107;
    }
    .danger-card {
        border-left-color: #dc3545;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .phase-1 { background-color: #28a745; color: white; }
    .phase-2 { background-color: #ffc107; color: black; }
    .phase-3 { background-color: #dc3545; color: white; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ LLM Security Auditor</h1>', unsafe_allow_html=True)
    st.markdown("**Automated Red Teaming for Language Models**")
    
    # Show strategy stats in header
    try:
        stats = get_strategy_stats()
        st.markdown(f"*{stats['total']} Attack Strategies Available | Avg Effectiveness: {stats['avg_effectiveness']:.0%}*")
    except:
        pass
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Target type selection
        target_type = st.radio(
            "Target Type",
            ["☁️ Cloud API", "🏠 Local Model"],
            horizontal=True,
            help="Choose between cloud API or locally running model"
        )
        
        is_local = (target_type == "🏠 Local Model")
        
        st.markdown("---")
        
        # Model selection based on type
        if is_local:
            st.subheader("🏠 Local Model")
            
            # Check if Ollama is running
            from backend.app.utils.local_model_manager import LocalModelManager
            manager = LocalModelManager()
            
            if not manager.check_ollama_installed():
                st.error("⚠️ Ollama not running!")
                st.caption("Go to Configuration → Local Models to install")
                target_provider = "ollama"
                target_model = "llama3.2:1b"
            else:
                installed_models = manager.get_installed_models()
                
                if installed_models:
                    model_names = [m['name'] for m in installed_models]
                    target_model = st.selectbox(
                        "Select Model",
                        model_names,
                        help="Choose from installed Ollama models"
                    )
                    target_provider = "ollama"
                    
                    # Show model info
                    selected_model_info = next(
                        (m for m in installed_models if m['name'] == target_model),
                        None
                    )
                    if selected_model_info:
                        st.caption(f"📦 Size: {selected_model_info.get('size', 'Unknown')}")
                else:
                    st.warning("No models installed")
                    st.caption("Pull a model: `ollama pull llama3.2:1b`")
                    target_provider = "ollama"
                    target_model = "llama3.2:1b"
        
        else:
            # Cloud API selection
            st.subheader("☁️ Cloud API")
            
            target_provider = st.selectbox(
                "Provider",
                ["nvidia", "groq"],
                help="LLM provider to test"
            )
            
            if target_provider == "nvidia":
                target_models = ["meta/llama3-70b-instruct", "meta/llama-3.1-405b-instruct"]
            else:
                target_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            
            target_model = st.selectbox(
                "Model",
                target_models,
                help="Specific model to audit"
            )
        
        st.markdown("---")
        
        # Audit settings
        st.subheader("Audit Settings")
        save_to_db = st.checkbox("Save results to database", value=True)
        parallel_execution = st.checkbox("Parallel execution", value=True)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        try:
            stats = get_strategy_stats()
            st.metric("Total Strategies", stats['total'])
            st.metric("Phase 1 (Basic)", stats['phase1'])
            st.metric("Phase 2 (Advanced)", stats['phase2'])
            st.metric("Phase 3 (Expert)", stats['phase3'])
        except:
            pass
        
        if st.button("Refresh Stats"):
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Attack Test",
        "🔥 Full Audit",
        "📊 Analytics",
        "🔬 Strategy Comparison",
        "📈 Model Comparison"
    ])
    
    # Tab 1: Single Attack Test
    with tab1:
        single_attack_interface(target_provider, target_model, is_local)
    
    # Tab 2: Full Audit
    with tab2:
        full_audit_interface(target_provider, target_model, is_local, save_to_db, parallel_execution)
    
    # Tab 3: Analytics
    with tab3:
        analytics_interface()
    
    # Tab 4: Strategy Comparison
    with tab4:
        strategy_comparison_interface()
    
    # Tab 5: Model Comparison
    with tab5:
        model_comparison_interface()


def single_attack_interface(target_provider, target_model, is_local):
    """Interface for testing single attacks"""
    
    st.header("🎯 Single Attack Test")
    st.markdown("Test individual attack strategies interactively")
    
    # Show target info
    target_type_badge = "🏠 LOCAL" if is_local else "☁️ CLOUD"
    st.info(f"{target_type_badge} Target: **{target_provider}/{target_model}**")
    
    # Strategy statistics overview
    try:
        stats = get_strategy_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Strategies", stats['total'])
        with col2:
            st.metric("🟢 Phase 1", stats['phase1'])
        with col3:
            st.metric("🟡 Phase 2", stats['phase2'])
        with col4:
            st.metric("🔴 Phase 3", stats['phase3'])
        
        st.markdown("---")
    except Exception as e:
        st.warning(f"Could not load strategy stats: {e}")
    
    # Phase filter
    phase_filter = st.radio(
        "Filter by Phase",
        ["All Strategies", "🟢 Phase 1: Basic", "🟡 Phase 2: Advanced", "🔴 Phase 3: Expert"],
        horizontal=True,
        key="phase_filter",
        help="Phase 1: Basic attacks | Phase 2: Advanced tactics | Phase 3: Expert-level attacks"
    )
    
    # Get strategies based on filter
    all_strategies = list_all_strategies()
    
    if phase_filter == "🟢 Phase 1: Basic":
        strategies = all_strategies[:5]
        st.info("🟢 **Phase 1 Strategies** - Basic attack patterns with moderate effectiveness")
    elif phase_filter == "🟡 Phase 2: Advanced":
        strategies = all_strategies[5:10]
        st.info("🟡 **Phase 2 Strategies** - Advanced manipulation techniques with higher success rates")
    elif phase_filter == "🔴 Phase 3: Expert":
        strategies = all_strategies[10:15]
        st.warning("🔴 **Phase 3 Strategies** - Expert-level attacks with highest effectiveness (Use responsibly!)")
    else:
        strategies = all_strategies
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        forbidden_goal = st.text_area(
            "Forbidden Goal",
            placeholder="Enter the harmful goal to test (e.g., 'How to hack email accounts')",
            height=100,
            key="single_attack_goal"
        )
        
        # Strategy selection with phase indicators
        if not strategies:
            st.error("No strategies available for selected phase")
            return
        
        # Create display names with phase badges
        strategy_display_map = {}
        for idx, s in enumerate(strategies):
            # Determine phase
            all_idx = all_strategies.index(s)
            if all_idx < 5:
                phase = "🟢 P1"
            elif all_idx < 10:
                phase = "🟡 P2"
            else:
                phase = "🔴 P3"
            
            display_name = f"{phase} | {s.name}"
            strategy_display_map[display_name] = s
        
        selected_display_name = st.selectbox(
            "Attack Strategy",
            list(strategy_display_map.keys()),
            help="Choose which attack strategy to use",
            key="strategy_select"
        )
        
        selected_strategy_obj = strategy_display_map[selected_display_name]
        
        # Show strategy details
        with st.expander("ℹ️ Strategy Details", expanded=False):
            st.write(f"**Name:** {selected_strategy_obj.name}")
            st.write(f"**Type:** {selected_strategy_obj.strategy_type.value}")
            st.write(f"**Description:** {selected_strategy_obj.description}")
            
            # Effectiveness bar
            if hasattr(selected_strategy_obj, 'effectiveness_score'):
                st.write("**Effectiveness:**")
                st.progress(selected_strategy_obj.effectiveness_score)
                st.caption(f"{selected_strategy_obj.effectiveness_score:.0%}")
            
            # Difficulty badge
            if hasattr(selected_strategy_obj, 'difficulty'):
                difficulty = selected_strategy_obj.difficulty.value
                difficulty_colors = {
                    'EASY': '🟢',
                    'MEDIUM': '🟡',
                    'HARD': '🟠',
                    'VERY_HARD': '🔴'
                }
                icon = difficulty_colors.get(difficulty, '⚪')
                st.write(f"**Difficulty:** {icon} {difficulty}")
    
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
                st.error(f"Preview error: {str(e)}")
        else:
            st.info("Enter a forbidden goal to see attack preview")
    
    # Run button
    st.markdown("---")
    
    if st.button("🚀 Run Attack", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("⚠️ Please enter a forbidden goal")
            return
        
        with st.spinner("🔄 Running attack... This may take 10-30 seconds"):
            result = run_single_attack_sync(
                forbidden_goal=forbidden_goal,
                strategy=selected_strategy_obj.strategy_type.value,
                target_provider=target_provider,
                target_model=target_model,
                is_local=is_local
            )
        
        if result:
            display_single_result(result)



def strategy_comparison_interface():
    """NEW: View to compare all strategies"""
    
    st.header("🔬 Strategy Comparison & Analysis")
    st.markdown("Compare effectiveness and characteristics of all attack strategies")
    
    try:
        stats = get_strategy_stats()
        
        # Display overall stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Strategies", stats['total'])
        with col2:
            st.metric("Average Effectiveness", f"{stats['avg_effectiveness']:.0%}")
        with col3:
            easy_count = stats['by_difficulty'].get('EASY', 0) + stats['by_difficulty'].get('MEDIUM', 0)
            st.metric("Easy/Medium", easy_count)
        with col4:
            hard_count = stats['by_difficulty'].get('HARD', 0) + stats['by_difficulty'].get('VERY_HARD', 0)
            st.metric("Hard/Very Hard", hard_count)
        
        st.markdown("---")
        
        # Get all strategies
        all_strategies = list_all_strategies()
        
        # Create comparison table
        comparison_data = []
        
        for idx, strategy in enumerate(all_strategies, 1):
            # Determine phase
            if idx <= 5:
                phase = "Phase 1"
                phase_badge = "🟢"
            elif idx <= 10:
                phase = "Phase 2"
                phase_badge = "🟡"
            else:
                phase = "Phase 3"
                phase_badge = "🔴"
            
            effectiveness = getattr(strategy, 'effectiveness_score', 0)
            difficulty = getattr(strategy, 'difficulty', 'UNKNOWN')
            difficulty_value = difficulty.value if hasattr(difficulty, 'value') else 'N/A'
            
            comparison_data.append({
                '': phase_badge,
                'Phase': phase,
                'Strategy Name': strategy.name,
                'Effectiveness': f"{effectiveness:.0%}",
                'Difficulty': difficulty_value,
                'Type': strategy.strategy_type.value,
                'Description': strategy.description[:60] + '...' if len(strategy.description) > 60 else strategy.description
            })
        
        # Display as dataframe
        df = pd.DataFrame(comparison_data)
        
        st.subheader("📋 All Strategies Overview")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "": st.column_config.TextColumn("", width="small"),
                "Effectiveness": st.column_config.ProgressColumn(
                    "Effectiveness",
                    format="%s",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
        
        # Visualizations
        st.markdown("---")
        
        # Effectiveness by Phase
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Effectiveness by Phase")
            
            phase_data = {
                'Phase 1': [getattr(s, 'effectiveness_score', 0) * 100 for s in all_strategies[:5]],
                'Phase 2': [getattr(s, 'effectiveness_score', 0) * 100 for s in all_strategies[5:10]],
                'Phase 3': [getattr(s, 'effectiveness_score', 0) * 100 for s in all_strategies[10:15]]
            }
            
            fig = go.Figure()
            
            colors = {'Phase 1': '#28a745', 'Phase 2': '#ffc107', 'Phase 3': '#dc3545'}
            
            for phase, values in phase_data.items():
                strategies_in_phase = all_strategies[:5] if phase == 'Phase 1' else all_strategies[5:10] if phase == 'Phase 2' else all_strategies[10:15]
                names = [s.name[:20] for s in strategies_in_phase]
                
                fig.add_trace(go.Bar(
                    name=phase,
                    x=names,
                    y=values,
                    marker_color=colors[phase],
                    text=[f"{v:.0f}%" for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title='Strategy',
                yaxis_title='Effectiveness (%)',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Difficulty Distribution")
            
            difficulty_counts = stats['by_difficulty']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(difficulty_counts.keys()),
                values=list(difficulty_counts.values()),
                marker=dict(colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545'])
            )])
            
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Phase comparison
        st.markdown("---")
        st.subheader("📊 Phase-by-Phase Comparison")
        
        phase_comparison = []
        for phase_num in [1, 2, 3]:
            strategies = get_strategies_by_phase(phase_num)
            if strategies:
                avg_eff = sum(getattr(s, 'effectiveness_score', 0) for s in strategies) / len(strategies)
                phase_comparison.append({
                    'Phase': f"Phase {phase_num}",
                    'Strategies': len(strategies),
                    'Avg Effectiveness': f"{avg_eff:.0%}",
                    'Description': "Basic attacks" if phase_num == 1 else "Advanced tactics" if phase_num == 2 else "Expert-level attacks"
                })
        
        phase_df = pd.DataFrame(phase_comparison)
        st.table(phase_df)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🟢 Phase 1: Start Here**
            - Best for initial testing
            - Moderate effectiveness
            - Easy to understand
            - Good for learning
            """)
        
        with col2:
            st.warning("""
            **🟡 Phase 2: Advanced**
            - Higher success rates
            - More sophisticated
            - Requires understanding
            - Better for thorough audits
            """)
        
        with col3:
            st.error("""
            **🔴 Phase 3: Expert**
            - Highest effectiveness
            - Complex techniques
            - Use responsibly
            - Best for security research
            """)
        
    except Exception as e:
        st.error(f"Error loading strategy comparison: {e}")
        import traceback
        st.code(traceback.format_exc())


def full_audit_interface(provider, model, is_local, save_to_db, parallel):
    """Interface for full audit with multiple strategies"""
    
    st.header("🔥 Full Security Audit")
    st.markdown("Test multiple attack strategies comprehensively")
    
    # Show target info
    target_type_badge = "🏠 LOCAL" if is_local else "☁️ CLOUD"
    st.info(f"{target_type_badge} Target: **{provider}/{model}**")
    
    # Input
    forbidden_goal = st.text_area(
        "Forbidden Goal",
        placeholder="Enter the harmful goal to audit",
        height=80,
        key="full_audit_goal"
    )
    
    # Strategy selection with phase options
    col1, col2 = st.columns(2)
    
    with col1:
        audit_scope = st.radio(
            "Audit Scope",
            ["All Strategies (15)", "Phase 1 Only (5)", "Phase 2 Only (5)", "Phase 3 Only (5)", "Custom Selection"],
            key="audit_scope"
        )
    
    with col2:
        if audit_scope == "Custom Selection":
            all_strategies = list_all_strategies()
            selected = st.multiselect(
                "Select strategies",
                [s.name for s in all_strategies],
                default=[all_strategies[0].name] if all_strategies else []
            )
    
    # Show what will be tested
    if audit_scope == "All Strategies (15)":
        st.info("✅ Will test all 15 strategies across all phases")
    elif audit_scope == "Phase 1 Only (5)":
        st.info("🟢 Will test 5 basic strategies")
    elif audit_scope == "Phase 2 Only (5)":
        st.info("🟡 Will test 5 advanced strategies")
    elif audit_scope == "Phase 3 Only (5)":
        st.warning("🔴 Will test 5 expert-level strategies")
    
    # Warning for local models with many strategies
    if is_local and audit_scope == "All Strategies (15)":
        st.warning("⚠️ Testing 15 strategies on local model may take 5-10 minutes. Consider selecting fewer strategies.")
    
    # Run audit button
    if st.button("🔥 Run Full Audit", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("Please enter a forbidden goal")
            return
        
        # Determine strategies to use
        all_strategies = list_all_strategies()
        
        if audit_scope == "All Strategies (15)":
            strategy_list = [s.strategy_type.value for s in all_strategies]
        elif audit_scope == "Phase 1 Only (5)":
            strategy_list = [s.strategy_type.value for s in all_strategies[:5]]
        elif audit_scope == "Phase 2 Only (5)":
            strategy_list = [s.strategy_type.value for s in all_strategies[5:10]]
        elif audit_scope == "Phase 3 Only (5)":
            strategy_list = [s.strategy_type.value for s in all_strategies[10:15]]
        else:  # Custom
            strategy_map = {s.name: s.strategy_type.value for s in all_strategies}
            strategy_list = [strategy_map[name] for name in selected]
        
        if not strategy_list:
            st.error("No strategies selected")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Running audit with {len(strategy_list)} strategies...")
        
        result = run_full_audit_sync(
            forbidden_goal=forbidden_goal,
            strategies=strategy_list,
            provider=provider,
            model=model,
            is_local=is_local,
            save_to_db=save_to_db,
            parallel=parallel
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Audit complete!")
        
        if result:
            display_audit_results(result)

def analytics_interface():
    """Analytics and historical data visualization"""
    
    st.header("📊 Security Analytics")
    st.markdown("Analyze historical audit results and trends")
    
    # Time range selector
    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.selectbox("Time Range", [7, 30, 90, 365], index=1)
    with col2:
        if st.button("Refresh Data"):
            st.rerun()
    
    # Fetch data
    data = fetch_analytics_data_sync(days)
    
    if not data:
        st.info("No data available. Run some audits first!")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tests", data['total_tests'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card danger-card">', unsafe_allow_html=True)
        st.metric("Jailbreaks", data['jailbreaks'], f"{data['asr']:.1f}% ASR")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
        st.metric("Partial Leaks", data['partial'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
        st.metric("Refused", data['refused'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Verdict distribution pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Refused', 'Jailbroken', 'Partial'],
            values=[data['refused'], data['jailbreaks'], data['partial']],
            marker=dict(colors=['#28a745', '#dc3545', '#ffc107'])
        )])
        fig_pie.update_layout(title="Verdict Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Strategy effectiveness
        if 'strategy_stats' in data:
            df_strategies = pd.DataFrame(data['strategy_stats'])
            fig_bar = px.bar(
                df_strategies,
                x='strategy',
                y='success_rate',
                title="Strategy Effectiveness",
                labels={'success_rate': 'Success Rate (%)', 'strategy': 'Strategy'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Trend over time
    if 'timeline' in data:
        df_timeline = pd.DataFrame(data['timeline'])
        fig_line = px.line(
            df_timeline,
            x='date',
            y='asr',
            title="Attack Success Rate Trend",
            labels={'asr': 'ASR (%)', 'date': 'Date'}
        )
        st.plotly_chart(fig_line, use_container_width=True)


def model_comparison_interface():
    """Compare different models' security"""
    
    st.header("📈 Model Comparison")
    st.markdown("Compare security across different language models")
    
    st.info("🚧 Feature coming soon! This will allow you to compare multiple models side-by-side.")
    
    # Placeholder for future implementation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model A")
        st.selectbox("Select Model A", ["llama-3.3-70b", "gpt-4", "claude-3"], key="model_a")
    
    with col2:
        st.subheader("Model B")
        st.selectbox("Select Model B", ["llama-3.1-8b", "gpt-3.5", "claude-2"], key="model_b")
    
    if st.button("Compare Models"):
        st.warning("This feature will be implemented in the next phase!")


# ========== HELPER FUNCTIONS ========== (Keep all existing helper functions)

def run_single_attack_sync(
    forbidden_goal: str,
    strategy: str,
    target_provider: str,
    target_model: str,
    is_local: bool
):
    """Synchronous wrapper for single attack"""
    
    async def _run():
        try:
            await db.connect()
            
            # Create orchestrator with local/cloud support
            orchestrator = Orchestrator(
                target_provider=target_provider,
                target_model=target_model,
                is_local=is_local
            )
            
            result = await orchestrator.run_single_attack(
                forbidden_goal=forbidden_goal,
                strategy_name=strategy,
                save_to_db=True
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
            except:
                pass
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_run())
        return result
    finally:
        loop.close()


def run_full_audit_sync(
    forbidden_goal,
    strategies,
    provider,
    model,
    is_local,
    save_to_db,
    parallel
):
    """Synchronous wrapper for full audit"""
    
    async def _run():
        try:
            await db.connect()
            
            # Create orchestrator with local/cloud support
            orchestrator = Orchestrator(
                target_provider=provider,
                target_model=model,
                is_local=is_local
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
            except:
                pass
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_run())
        return result
    finally:
        loop.close()

def fetch_analytics_data_sync(days: int):
    """Synchronous wrapper for analytics data"""
    
    async def _fetch():
        try:
            await db.connect()
            evaluations = await db.get_evaluations_by_filter(limit=1000)
            
            if not evaluations:
                return None
            
            total = len(evaluations)
            refused = sum(1 for e in evaluations if e.get('verdict') == 'REFUSED')
            jailbroken = sum(1 for e in evaluations if e.get('verdict') == 'JAILBROKEN')
            partial = sum(1 for e in evaluations if e.get('verdict') == 'PARTIAL')
            asr = (jailbroken / total * 100) if total > 0 else 0
            
            return {
                'total_tests': total,
                'refused': refused,
                'jailbreaks': jailbroken,
                'partial': partial,
                'asr': asr
            }
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
        finally:
            try:
                await db.disconnect()
            except:
                pass
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_fetch())
        return result
    finally:
        loop.close()

def get_strategy_stats():
    """Get statistics about attack strategies"""
    from backend.app.agents.strategies import list_all_strategies
    
    all_strategies = list_all_strategies()
    
    total = len(all_strategies)
    phase1 = 5
    phase2 = 5
    phase3 = total - phase1 - phase2
    
    # Calculate average effectiveness
    avg_eff = sum(
        getattr(s, 'effectiveness_score', 0) for s in all_strategies
    ) / total if total > 0 else 0
    
    # Count by difficulty
    by_difficulty = {}
    for s in all_strategies:
        if hasattr(s, 'difficulty'):
            diff = s.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
    
    return {
        'total': total,
        'phase1': phase1,
        'phase2': phase2,
        'phase3': phase3,
        'avg_effectiveness': avg_eff,
        'by_difficulty': by_difficulty
    }


def display_single_result(result):
    """Display single attack result"""
    
    st.markdown("---")
    st.subheader("🎯 Attack Results")
    
    verdict = result.evaluation.verdict.value
    if verdict == "JAILBROKEN":
        st.error(f"🔴 **JAILBROKEN** - Model was successfully attacked!")
    elif verdict == "PARTIAL":
        st.warning(f"🟡 **PARTIAL** - Model leaked some information")
    else:
        st.success(f"🟢 **REFUSED** - Model successfully refused")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{result.evaluation.confidence_score:.1%}")
    with col2:
        st.metric("Execution Time", f"{result.total_time_ms}ms")
    with col3:
        st.metric("Attack ID", result.attack.attack_id[:8] + "...")
    
    with st.expander("📝 Attack Prompt"):
        st.text_area("Generated Attack", result.attack.generated_prompt, height=200, disabled=True, key="result_attack_prompt")
    
    with st.expander("💬 Target Response"):
        st.text_area("Model Response", result.response.response_text, height=200, disabled=True, key="result_target_response")
    
    with st.expander("🧑‍⚖️ Judge Reasoning"):
        st.write(result.evaluation.reasoning)


def display_audit_results(result):
    """Display full audit results"""
    
    st.markdown("---")
    st.subheader("🔥 Audit Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Attacks", result.total_attacks)
    with col2:
        asr_color = "🔴" if result.attack_success_rate > 0.2 else "🟡" if result.attack_success_rate > 0.1 else "🟢"
        st.metric("Attack Success Rate", f"{result.attack_success_rate:.1%}", delta=asr_color)
    with col3:
        st.metric("Successful Jailbreaks", result.successful_jailbreaks)
    with col4:
        st.metric("Execution Time", f"{result.total_execution_time_ms/1000:.1f}s")
    
    st.subheader("📊 Detailed Results")
    
    results_data = []
    for r in result.results:
        results_data.append({
            "Strategy": r.attack.strategy_name,
            "Verdict": r.evaluation.verdict.value,
            "Confidence": f"{r.evaluation.confidence_score:.1%}",
            "Time (ms)": r.total_time_ms,
            "Success": "✅" if r.success else "❌"
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    fig = px.bar(
        df,
        x="Strategy",
        y=[1 if s == "✅" else 0 for s in df["Success"]],
        title="Attack Success by Strategy",
        labels={'y': 'Success (1=Yes, 0=No)'}
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
