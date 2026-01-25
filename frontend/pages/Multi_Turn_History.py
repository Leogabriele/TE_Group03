"""
Multi-Turn History - View past conversations and analytics
Direct backend integration (no API server needed)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import asyncio
import sys
import json
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.models.database import Database

# Page config
st.set_page_config(
    page_title="Multi-Turn History",
    page_icon="📜",
    layout="wide"
)

# Initialize database (cached)
@st.cache_resource
def get_db_and_loop():
    """Initialize database connection"""
    db = Database()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(db.connect())
    return db, loop

db, loop = get_db_and_loop()

# Helper function to run async code
def run_async(coro):
    """Run async coroutine in the event loop"""
    return loop.run_until_complete(coro)

# Helper to clean data for JSON export
def clean_for_json(obj):
    """Convert MongoDB objects to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return clean_for_json(obj.__dict__)
    else:
        return obj

# Title
st.title("📜 Multi-Turn Attack History")
st.markdown("**Browse and analyze past multi-turn conversation attacks**")

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    show_jailbroken_only = st.checkbox("🎯 Show Only Jailbreaks", value=False)

with col2:
    limit = st.selectbox("Results per page", options=[10, 20, 50, 100], index=1)

with col3:
    if st.button("🔄 Refresh"):
        st.cache_resource.clear()
        st.rerun()

# Fetch history
try:
    # Get conversation history directly from database
    conversations = run_async(
        db.get_multiturn_history(
            limit=limit,
            jailbroken_only=show_jailbroken_only
        )
    )
    
    if conversations:
        # Convert ObjectId and datetime to strings for DataFrame
        cleaned_conversations = []
        for conv in conversations:
            cleaned_conv = {}
            for key, value in conv.items():
                if key == '_id':
                    cleaned_conv['_id'] = str(value)
                elif isinstance(value, datetime):
                    cleaned_conv[key] = value
                else:
                    cleaned_conv[key] = value
            cleaned_conversations.append(cleaned_conv)
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_conversations)
        
        # Summary metrics
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", len(df))
        
        with col2:
            jailbreak_rate = (df['jailbreak_achieved'].sum() / len(df)) * 100
            st.metric("Jailbreak Rate", f"{jailbreak_rate:.1f}%")
        
        with col3:
            avg_turns = df['total_turns'].mean()
            st.metric("Avg Turns", f"{avg_turns:.1f}")
        
        with col4:
            if df['jailbreak_achieved'].any():
                jailbroken_df = df[df['jailbreak_achieved'] == True]
                if 'jailbreak_turn' in jailbroken_df.columns:
                    avg_jb_turn = jailbroken_df['jailbreak_turn'].mean()
                    st.metric("Avg Jailbreak Turn", f"{avg_jb_turn:.1f}")
                else:
                    st.metric("Avg Jailbreak Turn", "N/A")
            else:
                st.metric("Avg Jailbreak Turn", "N/A")
        
        st.divider()
        
        # Visualizations
        tab1, tab2 = st.tabs(["📋 Conversation List", "📈 Analytics"])
        
        with tab1:
            # Display conversations
            for idx, conv in enumerate(cleaned_conversations):
                # Format timestamp
                timestamp = conv.get('timestamp', 'Unknown')
                if isinstance(timestamp, datetime):
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, str):
                    try:
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        timestamp_str = str(timestamp)
                else:
                    timestamp_str = "Unknown"
                
                conv_id = conv.get('conversation_id', 'Unknown')[:8]
                forbidden_goal = conv.get('forbidden_goal', 'N/A')[:50]
                
                with st.expander(
                    f"{'🎯' if conv.get('jailbreak_achieved') else '🛡️'} "
                    f"{conv_id}... - {forbidden_goal}...",
                    expanded=(idx == 0)
                ):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Turns", conv.get('total_turns', 0))
                    
                    with col_b:
                        status = "JAILBROKEN" if conv.get('jailbreak_achieved') else "REFUSED"
                        st.metric("Final Verdict", status)
                    
                    with col_c:
                        jb_turn = conv.get('jailbreak_turn')
                        st.metric("Jailbreak Turn", jb_turn if jb_turn else 'N/A')
                    
                    with col_d:
                        best_strat = conv.get('most_effective_strategy', 'N/A')
                        st.metric("Best Strategy", best_strat if best_strat else 'N/A')
                    
                    st.markdown(f"**Forbidden Goal:** {conv.get('forbidden_goal', 'N/A')}")
                    st.markdown(f"**Target Model:** `{conv.get('target_model', 'N/A')}`")
                    st.markdown(f"**Timestamp:** {timestamp_str}")
                    
                    # Show strategies tried
                    strategies_tried = conv.get('strategies_tried', [])
                    if strategies_tried:
                        st.markdown(f"**Strategies Tried:** {', '.join(strategies_tried)}")
                    
                    # Show success rates if available
                    success_rates = conv.get('strategy_success_rates', {})
                    if success_rates:
                        st.markdown("**Strategy Success Rates:**")
                        for strategy, rate in success_rates.items():
                            st.write(f"  - {strategy}: {rate:.1%}")
        
        with tab2:
            # Analytics charts
            st.markdown("### 📈 Historical Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate by model
                if 'target_model' in df.columns:
                    model_success = df.groupby('target_model')['jailbreak_achieved'].mean() * 100
                    fig_model = px.bar(
                        x=model_success.index,
                        y=model_success.values,
                        title="Success Rate by Target Model",
                        labels={'x': 'Model', 'y': 'Success Rate (%)'},
                        color=model_success.values,
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_model, use_container_width=True)
                else:
                    st.info("No model data available")
            
            with col2:
                # Turns distribution
                fig_turns = px.histogram(
                    df,
                    x='total_turns',
                    title="Distribution of Conversation Lengths",
                    labels={'total_turns': 'Total Turns'},
                    nbins=10,
                    color_discrete_sequence=['#3b82f6']
                )
                st.plotly_chart(fig_turns, use_container_width=True)
            
            # Jailbreak rate over time
            if 'timestamp' in df.columns:
                st.markdown("### 📅 Jailbreak Rate Over Time")
                
                # Ensure timestamp is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Group by date
                df['date'] = df['timestamp'].dt.date
                daily_stats = df.groupby('date').agg({
                    'jailbreak_achieved': ['sum', 'count', 'mean']
                }).reset_index()
                daily_stats.columns = ['date', 'jailbreaks', 'total', 'rate']
                daily_stats['rate'] = daily_stats['rate'] * 100
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['rate'],
                    mode='lines+markers',
                    name='Jailbreak Rate',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=10)
                ))
                fig_timeline.update_layout(
                    title="Jailbreak Rate Over Time",
                    xaxis_title="Date",
                    yaxis_title="Jailbreak Rate (%)",
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Strategy effectiveness
            st.markdown("### 🎯 Strategy Effectiveness Across All Conversations")
            
            # Aggregate strategy data
            all_strategies = []
            for conv in cleaned_conversations:
                if conv.get('most_effective_strategy'):
                    all_strategies.append(conv['most_effective_strategy'])
            
            if all_strategies:
                strategy_counts = pd.Series(all_strategies).value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_strat = px.pie(
                        values=strategy_counts.values,
                        names=strategy_counts.index,
                        title="Most Effective Strategies"
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
                
                with col2:
                    # Bar chart of strategy usage
                    fig_strat_bar = px.bar(
                        x=strategy_counts.index,
                        y=strategy_counts.values,
                        title="Strategy Effectiveness Count",
                        labels={'x': 'Strategy', 'y': 'Times Most Effective'},
                        color=strategy_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_strat_bar, use_container_width=True)
            else:
                st.info("No strategy effectiveness data available yet")
            
            # Average turns to jailbreak
            st.markdown("### ⚡ Performance Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if df['jailbreak_achieved'].any():
                    jailbroken_df = df[df['jailbreak_achieved']]
                    if 'total_duration_ms' in jailbroken_df.columns:
                        avg_duration = jailbroken_df['total_duration_ms'].mean()
                        st.metric(
                            "Avg Time to Jailbreak",
                            f"{avg_duration/1000:.1f}s" if avg_duration > 0 else "N/A"
                        )
                    else:
                        st.metric("Avg Time to Jailbreak", "N/A")
                else:
                    st.metric("Avg Time to Jailbreak", "N/A")
            
            with col2:
                total_tested = len(df)
                unique_models = df['target_model'].nunique() if 'target_model' in df.columns else 0
                st.metric("Models Tested", unique_models)
            
            with col3:
                total_turns = df['total_turns'].sum()
                st.metric("Total Attack Turns", int(total_turns))
        
        # Detailed data table
        st.divider()
        st.markdown("### 📋 Detailed Conversation Data")
        
        # Prepare display dataframe
        display_columns = ['conversation_id', 'forbidden_goal', 'target_model', 'total_turns', 'jailbreak_achieved', 'final_verdict', 'timestamp']
        available_columns = [col for col in display_columns if col in df.columns]
        
        display_df = df[available_columns].copy()
        
        # Format columns
        if 'conversation_id' in display_df.columns:
            display_df['conversation_id'] = display_df['conversation_id'].str[:8] + '...'
        if 'forbidden_goal' in display_df.columns:
            display_df['forbidden_goal'] = display_df['forbidden_goal'].str[:40] + '...'
        if 'jailbreak_achieved' in display_df.columns:
            display_df['jailbreak_achieved'] = display_df['jailbreak_achieved'].map({
                True: '✅ Yes',
                False: '❌ No'
            })
        
        # Rename columns
        column_rename = {
            'conversation_id': 'Conv ID',
            'forbidden_goal': 'Goal',
            'target_model': 'Model',
            'total_turns': 'Turns',
            'jailbreak_achieved': 'Jailbroken',
            'final_verdict': 'Verdict',
            'timestamp': 'Timestamp'
        }
        display_df = display_df.rename(columns={k: v for k, v in column_rename.items() if k in display_df.columns})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        st.markdown("### 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"multiturn_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export as JSON (properly handle datetime)
            try:
                # Clean data for JSON export
                export_data = []
                for conv in cleaned_conversations:
                    clean_conv = {}
                    for key, value in conv.items():
                        if isinstance(value, datetime):
                            clean_conv[key] = value.isoformat()
                        elif key == '_id':
                            clean_conv[key] = str(value)
                        else:
                            clean_conv[key] = value
                    export_data.append(clean_conv)
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name=f"multiturn_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"JSON export error: {e}")
                logger.exception("JSON export failed:")
    
    else:
        st.info("📭 No conversations found. Start a new multi-turn attack to see results here!")
        
        # Quick start button
        if st.button("🚀 Start Multi-Turn Attack"):
            st.switch_page("pages/Multi_Turn_Attack.py")

except Exception as e:
    st.error(f"❌ Error loading history: {e}")
    logger.exception("History loading error:")
