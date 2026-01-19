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
from backend.app.agents.strategies import list_all_strategies, get_strategy
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
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ LLM Security Auditor</h1>', unsafe_allow_html=True)
    st.markdown("**Automated Red Teaming for Language Models**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Target model selection
        st.subheader("Target Model")
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
        if st.button("Refresh Stats"):
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Attack Test",
        "🔥 Full Audit",
        "📊 Analytics",
        "📈 Model Comparison"
    ])
    
    # Tab 1: Single Attack Test
    with tab1:
        single_attack_interface()
    
    # Tab 2: Full Audit
    with tab2:
        full_audit_interface(target_provider, target_model, save_to_db, parallel_execution)
    
    # Tab 3: Analytics
    with tab3:
        analytics_interface()
    
    # Tab 4: Model Comparison
    with tab4:
        model_comparison_interface()

def single_attack_interface():
    """Interface for testing single attacks"""
    
    st.header("🎯 Single Attack Test")
    st.markdown("Test individual attack strategies interactively")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        forbidden_goal = st.text_area(
            "Forbidden Goal",
            placeholder="Enter the harmful goal to test (e.g., 'How to hack email accounts')",
            height=100,
            key="single_attack_goal"
        )
        
        # Strategy selection
        strategies = list_all_strategies()
        strategy_names = [s.name for s in strategies]
        strategy_map = {s.name: s.strategy_type.value for s in strategies}
        
        selected_strategy_name = st.selectbox(
            "Attack Strategy",
            strategy_names,
            help="Choose which attack strategy to use"
        )
        
        selected_strategy = strategy_map[selected_strategy_name]
        
        # Show strategy details
        strategy_obj = get_strategy(selected_strategy)
        with st.expander("ℹ️ Strategy Details"):
            st.write(f"**Name:** {strategy_obj.name}")
            st.write(f"**Description:** {strategy_obj.description}")
            if hasattr(strategy_obj, 'effectiveness_score'):
                st.write(f"**Effectiveness:** {strategy_obj.effectiveness_score:.0%}")
            if hasattr(strategy_obj, 'difficulty'):
                st.write(f"**Difficulty:** {strategy_obj.difficulty.value}")
    
    with col2:
        st.subheader("Preview Generated Attack")
        if forbidden_goal:
            try:
                preview = strategy_obj.generate(forbidden_goal)
                st.text_area(
                    "Attack Prompt Preview",
                    value=preview[:300] + "..." if len(preview) > 300 else preview,
                    height=250,
                    disabled=True,
                    key="preview_area"
                )
            except Exception as e:
                st.info(f"Preview unavailable: {str(e)}")
    
    # Run button
    if st.button("🚀 Run Attack", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("Please enter a forbidden goal")
            return
        
        with st.spinner("Running attack... This may take 10-30 seconds"):
            result = run_single_attack_sync(  # ← No asyncio.run()
                forbidden_goal=forbidden_goal,
                strategy=selected_strategy
            )
        
        if result:
            display_single_result(result)


def full_audit_interface(provider, model, save_to_db, parallel):
    """Interface for full audit with multiple strategies"""
    
    st.header("🔥 Full Security Audit")
    st.markdown("Test multiple attack strategies comprehensively")
    
    # Input
    forbidden_goal = st.text_area(
        "Forbidden Goal",
        placeholder="Enter the harmful goal to audit",
        height=80,
        key="full_audit_goal"
    )
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        use_all = st.checkbox("Test all strategies", value=True)
    
    with col2:
        if not use_all:
            strategies = list_all_strategies()
            selected = st.multiselect(
                "Select strategies",
                [s.name for s in strategies],
                default=[strategies[0].name] if strategies else []
            )
    
    # Run audit button
    if st.button("🔥 Run Full Audit", type="primary", use_container_width=True):
        if not forbidden_goal:
            st.error("Please enter a forbidden goal")
            return
        
        # Determine strategies to use
        if use_all:
            strategy_list = [s.strategy_type.value for s in list_all_strategies()]
        else:
            strategy_map = {s.name: s.strategy_type.value for s in list_all_strategies()}
            strategy_list = [strategy_map[name] for name in selected]
        
        if not strategy_list:
            st.error("No strategies selected")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Running audit with {len(strategy_list)} strategies...")
        
        result = run_full_audit_sync(  # ← No asyncio.run()
            forbidden_goal=forbidden_goal,
            strategies=strategy_list,
            provider=provider,
            model=model,
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
    data = fetch_analytics_data_sync(days)  # ← No asyncio.run()
    
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


# ========== HELPER FUNCTIONS ==========

def run_single_attack_sync(forbidden_goal: str, strategy: str):
    """Synchronous wrapper for single attack"""
    
    async def _run():
        orchestrator = None
        try:
            # Connect DB
            await db.connect()
            
            # Create orchestrator
            orchestrator = Orchestrator()
            
            # Run attack
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
            # Always disconnect
            try:
                await db.disconnect()
            except:
                pass
    
    # Create new event loop for this operation
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_run())
        return result
    finally:
        loop.close()


def run_full_audit_sync(forbidden_goal, strategies, provider, model, save_to_db, parallel):
    """Synchronous wrapper for full audit"""
    
    async def _run():
        try:
            # Connect DB
            await db.connect()
            
            # Create orchestrator
            orchestrator = Orchestrator(
                target_provider=provider,
                target_model=model
            )
            
            # Run audit
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
            # Always disconnect
            try:
                await db.disconnect()
            except:
                pass
    
    # Create new event loop
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
            # Connect DB
            await db.connect()
            
            # Get evaluations
            evaluations = await db.get_evaluations_by_filter(limit=1000)
            
            if not evaluations:
                return None
            
            # Calculate stats
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
            # Always disconnect
            try:
                await db.disconnect()
            except:
                pass
    
    # Create new event loop
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_fetch())
        return result
    finally:
        loop.close()

    
def display_single_result(result):
    """Display single attack result"""
    
    st.markdown("---")
    st.subheader("🎯 Attack Results")
    
    # Verdict banner
    verdict = result.evaluation.verdict.value
    if verdict == "JAILBROKEN":
        st.error(f"🔴 **JAILBROKEN** - Model was successfully attacked!")
    elif verdict == "PARTIAL":
        st.warning(f"🟡 **PARTIAL** - Model leaked some information")
    else:
        st.success(f"🟢 **REFUSED** - Model successfully refused")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{result.evaluation.confidence_score:.1%}")
    with col2:
        st.metric("Execution Time", f"{result.total_time_ms}ms")
    with col3:
        st.metric("Attack ID", result.attack.attack_id[:8] + "...")
    
    # Details
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
    
    # Summary metrics
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
    
    # Results table
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
    
    # Visualization
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
