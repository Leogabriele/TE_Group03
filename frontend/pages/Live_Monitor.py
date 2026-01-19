"""
Real-time Monitoring Dashboard
Live view of ongoing audits and system metrics
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.models.database import db

st.set_page_config(
    page_title="Live Monitor",
    page_icon="🔴",
    layout="wide"
)

st.title("🔴 Real-Time Security Monitor")
st.markdown("**Live monitoring of security audits and system health**")
st.markdown("---")

# Initialize session state for live data
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

def fetch_data_sync():
    """Synchronous wrapper for async data fetching"""
    async def _fetch():
        try:
            result = await fetch_live_data_sync()  # or generate_report_data()
            return result
        finally:
            try:
                await db.disconnect()
            except:
                pass
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch())
    finally:
        loop.close()

def main():
    # Control panel
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
    
    with col2:
        if st.button("▶️ Start" if not st.session_state.monitoring_active else "⏸️ Pause"):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
    
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col4:
        st.metric("Updates", st.session_state.refresh_count)
    
    st.markdown("---")
    
    # Fetch live data
    data = fetch_live_data_sync()
    
    if not data:
        st.info("📡 No live data available. System is idle.")
        st.stop()
    
    # Real-time metrics
    st.subheader("📊 Current System Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Active Tests",
            data.get('active_tests', 0),
            delta=None,
            help="Currently running audit tests"
        )
    
    with col2:
        st.metric(
            "Tests Today",
            data.get('tests_today', 0),
            help="Total tests completed today"
        )
    
    with col3:
        recent_asr = data.get('recent_asr', 0)
        asr_color = "🔴" if recent_asr > 20 else "🟡" if recent_asr > 10 else "🟢"
        st.metric(
            "Recent ASR",
            f"{recent_asr:.1f}%",
            delta=asr_color,
            help="Attack Success Rate (last 24h)"
        )
    
    with col4:
        avg_response = data.get('avg_response_time', 0)
        st.metric(
            "Avg Response",
            f"{avg_response:.0f}ms",
            help="Average target model response time"
        )
    
    with col5:
        system_health = data.get('system_health', 'Unknown')
        health_emoji = "🟢" if system_health == "Healthy" else "🟡" if system_health == "Warning" else "🔴"
        st.metric(
            "System Health",
            f"{health_emoji} {system_health}",
            help="Overall system health status"
        )
    
    st.markdown("---")
    
    # Live activity feed
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📡 Live Activity Feed")
        display_activity_feed(data.get('recent_activities', []))
    
    with col2:
        st.subheader("🎯 Recent Verdicts")
        display_verdict_distribution(data.get('recent_verdicts', {}))
    
    st.markdown("---")
    
    # Real-time charts
    st.subheader("📈 Real-Time Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_response_time_chart(data.get('response_times', []))
    
    with col2:
        display_success_rate_gauge(recent_asr)
    
    # Auto-refresh logic
    if st.session_state.monitoring_active:
        st.session_state.refresh_count += 1
        time.sleep(refresh_rate)
        st.rerun()


def fetch_live_data_sync():
    """Synchronous wrapper for live data fetching"""
    
    async def _fetch():
        try:
            await db.connect()
            
            evaluations = await db.get_evaluations_by_filter(limit=100)
            
            if not evaluations:
                await db.disconnect()
                return None
            
            # Calculate metrics
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            tests_today = 0
            recent_24h = []
            
            for e in evaluations:
                try:
                    timestamp = e.get('timestamp')
                    if isinstance(timestamp, str):
                        eval_time = datetime.fromisoformat(timestamp)
                    elif isinstance(timestamp, datetime):
                        eval_time = timestamp
                    else:
                        continue
                    
                    if eval_time >= today:
                        tests_today += 1
                    
                    if (now - eval_time).total_seconds() < 86400:
                        recent_24h.append(e)
                        
                except Exception:
                    continue
            
            # Recent ASR
            if recent_24h:
                jailbreaks_24h = sum(1 for e in recent_24h if e.get('verdict') == 'JAILBROKEN')
                recent_asr = (jailbreaks_24h / len(recent_24h)) * 100
            else:
                recent_asr = 0
            
            # Response times
            response_times = []
            for e in recent_24h:
                latency = e.get('metadata', {}).get('judge_latency_ms')
                if latency and isinstance(latency, (int, float)):
                    response_times.append(latency)
            
            avg_response = sum(response_times) / len(response_times) if response_times else 0
            
            # Verdict distribution
            verdict_counts = {'REFUSED': 0, 'JAILBROKEN': 0, 'PARTIAL': 0}
            for e in recent_24h:
                verdict = e.get('verdict', 'REFUSED')
                if verdict in verdict_counts:
                    verdict_counts[verdict] += 1
            
            # Recent activities
            recent_activities = []
            for e in evaluations[:10]:
                try:
                    timestamp = e.get('timestamp')
                    if isinstance(timestamp, str):
                        eval_time = datetime.fromisoformat(timestamp)
                    elif isinstance(timestamp, datetime):
                        eval_time = timestamp
                    else:
                        eval_time = now
                    
                    recent_activities.append({
                        'time': eval_time,
                        'verdict': e.get('verdict', 'UNKNOWN'),
                        'confidence': e.get('confidence_score', 0),
                        'id': e.get('evaluation_id', 'N/A')[:8]
                    })
                except:
                    continue
            
            # System health
            if recent_asr > 30:
                system_health = "Critical"
            elif recent_asr > 15:
                system_health = "Warning"
            else:
                system_health = "Healthy"
            
            result = {
                'active_tests': 0,
                'tests_today': tests_today,
                'recent_asr': recent_asr,
                'avg_response_time': avg_response,
                'system_health': system_health,
                'recent_activities': recent_activities,
                'recent_verdicts': verdict_counts,
                'response_times': response_times[-20:] if len(response_times) > 20 else response_times
            }
            
            await db.disconnect()
            return result
            
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            try:
                await db.disconnect()
            except:
                pass
            return None
    
    # Create new event loop
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch())
    finally:
        loop.close()




def display_activity_feed(activities):
    """Display live activity feed"""
    
    if not activities:
        st.info("No recent activity")
        return
    
    now = datetime.now()
    
    for activity in activities:
        verdict = activity['verdict']
        
        if verdict == 'JAILBROKEN':
            icon = "🔴"
            color = "#dc3545"
        elif verdict == 'PARTIAL':
            icon = "🟡"
            color = "#ffc107"
        else:
            icon = "🟢"
            color = "#28a745"
        
        # Calculate time ago
        try:
            time_diff = (now - activity['time']).total_seconds()
            if time_diff < 60:
                time_str = f"{int(time_diff)}s ago"
            elif time_diff < 3600:
                time_str = f"{int(time_diff // 60)}m ago"
            else:
                time_str = f"{int(time_diff // 3600)}h ago"
        except:
            time_str = "recently"
        
        st.markdown(
            f"""
            <div style="padding: 0.5rem; margin: 0.5rem 0; border-left: 3px solid {color}; background-color: #f8f9fa;">
                <strong>{icon} {verdict}</strong> - Confidence: {activity['confidence']:.1%}<br>
                <small>ID: {activity['id']} • {time_str}</small>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_verdict_distribution(verdicts):
    """Display verdict distribution pie chart"""
    
    if not verdicts or sum(verdicts.values()) == 0:
        st.info("No verdict data")
        return
    
    fig = go.Figure(data=[go.Pie(
        labels=list(verdicts.keys()),
        values=list(verdicts.values()),
        marker=dict(colors=['#28a745', '#dc3545', '#ffc107']),
        hole=0.4
    )])
    
    fig.update_layout(
        showlegend=True,
        height=300,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_response_time_chart(response_times):
    """Display response time trend"""
    
    if not response_times:
        st.info("No response time data")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=response_times,
        mode='lines+markers',
        name='Response Time',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Response Time Trend (Last 20 Tests)",
        xaxis_title="Test Number",
        yaxis_title="Time (ms)",
        height=300,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_success_rate_gauge(asr):
    """Display ASR gauge chart"""
    
    # Determine color
    if asr > 20:
        color = "red"
    elif asr > 10:
        color = "yellow"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=asr,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Attack Success Rate (%)"},
        delta={'reference': 10, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "lightyellow"},
                {'range': [20, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
