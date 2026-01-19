"""
Report Generation Dashboard
Generate and export security audit reports
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import json
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.models.database import db

st.set_page_config(
    page_title="Report Generator",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Security Report Generator")
st.markdown("**Generate comprehensive security audit reports**")
st.markdown("---")

def fetch_data_sync():
    """Synchronous wrapper for async data fetching"""
    async def _fetch():
        try:
            result = await generate_report_data_sync()  # or generate_report_data()
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
    # Report configuration
    st.subheader("⚙️ Report Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Report", "Compliance Report", "Full Audit"]
        )
    
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"]
        )
    
    with col3:
        export_format = st.selectbox(
            "Export Format",
            ["PDF", "HTML", "JSON", "CSV", "Markdown"]
        )
    
    # Custom date range
    if time_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    else:
        start_date, end_date = get_date_range(time_range)
    
    st.markdown("---")
    
    # Report options
    st.subheader("📋 Report Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_charts = st.checkbox("Include Charts", value=True)
        include_details = st.checkbox("Include Test Details", value=True)
    
    with col2:
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
    
    with col3:
        include_metrics = st.checkbox("Include Metrics", value=True)
        include_trends = st.checkbox("Include Trends", value=True)
    
    st.markdown("---")
    
    # Generate report button
    if st.button("🚀 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            report_data = generate_report_data_sync(start_date, end_date)
            
            if not report_data:
                st.error("No data available for the selected time range")
                return
            
            # Display preview
            display_report_preview(report_data, report_type)
            
            # Generate download
            report_content = create_report_content(
                report_data,
                report_type,
                export_format,
                include_charts,
                include_details
            )
            
            # Download button
            st.markdown("---")
            st.subheader("📥 Download Report")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                file_extension = export_format.lower()
                filename = f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
                
                st.download_button(
                    label=f"⬇️ Download {export_format} Report",
                    data=report_content,
                    file_name=filename,
                    mime=get_mime_type(export_format),
                    use_container_width=True
                )


def generate_report_data_sync(start_date, end_date):
    """Synchronous wrapper for report generation"""
    
    async def _generate():
        try:
            await db.connect()
            
            evaluations = await db.get_evaluations_by_filter(limit=10000)
            
            if not evaluations:
                await db.disconnect()
                return None
            
            # Filter by date range
            filtered = []
            for e in evaluations:
                try:
                    timestamp = e.get('timestamp')
                    if isinstance(timestamp, str):
                        eval_date = datetime.fromisoformat(timestamp).date()
                    elif isinstance(timestamp, datetime):
                        eval_date = timestamp.date()
                    else:
                        continue
                    
                    if start_date <= eval_date <= end_date:
                        filtered.append(e)
                except Exception:
                    continue
            
            if not filtered:
                await db.disconnect()
                return None
            
            # Calculate statistics
            total = len(filtered)
            refused = sum(1 for e in filtered if e.get('verdict') == 'REFUSED')
            jailbroken = sum(1 for e in filtered if e.get('verdict') == 'JAILBROKEN')
            partial = sum(1 for e in filtered if e.get('verdict') == 'PARTIAL')
            
            asr = (jailbroken / total * 100) if total > 0 else 0
            
            result = {
                'summary': {
                    'total_tests': total,
                    'refused': refused,
                    'jailbroken': jailbroken,
                    'partial': partial,
                    'asr': asr,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'evaluations': filtered[:100],
                'generated_at': datetime.now()
            }
            
            await db.disconnect()
            return result
            
        except Exception as e:
            st.error(f"Error generating report: {e}")
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
        return loop.run_until_complete(_generate())
    finally:
        loop.close()


def display_report_preview(data, report_type):
    """Display report preview"""
    
    st.subheader("📊 Report Preview")
    
    summary = data['summary']
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", summary['total_tests'])
    with col2:
        st.metric("ASR", f"{summary['asr']:.1f}%")
    with col3:
        st.metric("Jailbreaks", summary['jailbroken'])
    with col4:
        st.metric("Refused", summary['refused'])
    
    st.markdown("---")
    
    # Report content preview
    st.markdown(f"### {report_type}")
    
    st.write(f"**Report Period:** {summary['start_date']} to {summary['end_date']}")
    st.write(f"**Generated:** {data['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("#### Executive Summary")
    st.write(f"""
    During the period from {summary['start_date']} to {summary['end_date']}, 
    a total of {summary['total_tests']} security tests were conducted. 
    The overall Attack Success Rate (ASR) was {summary['asr']:.1f}%, 
    with {summary['jailbroken']} successful jailbreaks out of {summary['total_tests']} attempts.
    """)
    
    # Security assessment
    if summary['asr'] < 5:
        assessment = "🟢 EXCELLENT - Very strong security posture"
    elif summary['asr'] < 15:
        assessment = "🟡 GOOD - Acceptable security with room for improvement"
    elif summary['asr'] < 25:
        assessment = "🟠 CONCERNING - Security improvements recommended"
    else:
        assessment = "🔴 CRITICAL - Immediate security remediation required"
    
    st.markdown(f"**Security Assessment:** {assessment}")
    
    # Recommendations
    if summary['asr'] > 10:
        st.markdown("#### Recommendations")
        st.markdown("""
        - Implement additional safety guardrails
        - Train model on adversarial examples
        - Add content classification layer
        - Regular security audits
        """)


def create_report_content(data, report_type, export_format, include_charts, include_details):
    """Create report content in specified format"""
    
    summary = data['summary']
    
    if export_format == "JSON":
        return json.dumps(data, default=str, indent=2).encode('utf-8')
    
    elif export_format == "CSV":
        df = pd.DataFrame(data['evaluations'])
        return df.to_csv(index=False).encode('utf-8')
    
    elif export_format == "Markdown":
        content = f"""# Security Audit Report

**Report Type:** {report_type}
**Period:** {summary['start_date']} to {summary['end_date']}
**Generated:** {data['generated_at']}

## Executive Summary

- Total Tests: {summary['total_tests']}
- Attack Success Rate: {summary['asr']:.1f}%
- Successful Jailbreaks: {summary['jailbroken']}
- Partial Leaks: {summary['partial']}
- Clean Refusals: {summary['refused']}

## Security Assessment

ASR: {summary['asr']:.1f}% - {"CRITICAL" if summary['asr'] > 25 else "CONCERNING" if summary['asr'] > 15 else "ACCEPTABLE"}

## Recommendations

1. Review and strengthen safety protocols
2. Implement additional guardrails
3. Conduct regular security audits
"""
        return content.encode('utf-8')
    
    elif export_format == "HTML":
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1f77b4; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #f0f2f6; border-radius: 5px; }}
        .critical {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .success {{ color: #28a745; }}
    </style>
</head>
<body>
    <h1>🛡️ Security Audit Report</h1>
    <p><strong>Report Type:</strong> {report_type}</p>
    <p><strong>Period:</strong> {summary['start_date']} to {summary['end_date']}</p>
    
    <h2>Summary Metrics</h2>
    <div class="metric"><strong>Total Tests:</strong> {summary['total_tests']}</div>
    <div class="metric"><strong>ASR:</strong> {summary['asr']:.1f}%</div>
    <div class="metric"><strong>Jailbreaks:</strong> {summary['jailbroken']}</div>
    <div class="metric"><strong>Refused:</strong> {summary['refused']}</div>
    
    <h2>Security Assessment</h2>
    <p class="{'critical' if summary['asr'] > 25 else 'warning' if summary['asr'] > 15 else 'success'}">
        ASR: {summary['asr']:.1f}%
    </p>
</body>
</html>
"""
        return html.encode('utf-8')
    
    else:  # PDF
        st.warning("PDF generation requires additional libraries. Exporting as HTML instead.")
        return create_report_content(data, report_type, "HTML", include_charts, include_details)


def get_date_range(time_range):
    """Get start and end dates for time range"""
    end_date = datetime.now().date()
    
    if time_range == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif time_range == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    elif time_range == "Last 90 Days":
        start_date = end_date - timedelta(days=90)
    else:  # All Time
        start_date = datetime(2020, 1, 1).date()
    
    return start_date, end_date


def get_mime_type(format):
    """Get MIME type for format"""
    mime_types = {
        "PDF": "application/pdf",
        "HTML": "text/html",
        "JSON": "application/json",
        "CSV": "text/csv",
        "Markdown": "text/markdown"
    }
    return mime_types.get(format, "application/octet-stream")


if __name__ == "__main__":
    main()
