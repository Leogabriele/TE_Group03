"""
Configuration Management Dashboard
Manage system settings and preferences
"""

import streamlit as st
import json
import sys
from pathlib import Path
from backend.app.config import settings
from backend.app.models.database import db
from loguru import logger
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.config import settings

st.set_page_config(
    page_title="Configuration",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ System Configuration")
st.markdown("**Manage system settings and preferences**")
st.markdown("---")


def main():
    tabs = st.tabs([
        "🔑 API Keys",
        "🎯 Target Models",
        "🧑‍⚖️ Judge Settings",
        "💾 Database",
        "🏠 Local Models",
        "🔧 Advanced"
    ])
    
    # Tab 1: API Keys
    with tabs[0]:
        api_keys_config()
    
    # Tab 2: Target Models
    with tabs[1]:
        target_models_config()
    
    # Tab 3: Judge Settings
    with tabs[2]:
        judge_settings_config()
    
    # Tab 4: Database
    with tabs[3]:
        database_config()
    
    with tabs[4]:
        local_models_config()
    # Tab 5: Advanced
    with tabs[5]:
        advanced_config()

def local_models_config():
    """NEW: Local models configuration"""
    
    st.header("🏠 Local Model Management")
    st.markdown("Run LLMs locally for free using Ollama")
    
    # Check if Ollama is installed
    from backend.app.utils.local_model_manager import LocalModelManager
    
    manager = LocalModelManager()
    is_installed = manager.check_ollama_installed()
    
    if not is_installed:
        st.warning("⚠️ Ollama is not installed or not running")
        
        with st.expander("📖 How to Install Ollama"):
            st.markdown(manager.get_ollama_install_instructions())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Again"):
                st.rerun()
        with col2:
            st.link_button("📥 Download Ollama", "https://ollama.ai/download")
        
        return
    
    st.success("✅ Ollama is running!")
    
    # Show installed models
    st.subheader("📦 Installed Models")
    
    installed = manager.get_installed_models()
    
    if installed:
        for model in installed:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{model['name']}**")
                st.caption(f"Size: {model.get('size', 'Unknown')}")
            
            with col2:
                if st.button("ℹ️ Info", key=f"info_{model['name']}"):
                    info = manager.get_model_info(model['name'])
                    st.json(info)
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{model['name']}"):
                    if manager.delete_model(model['name']):
                        st.success(f"Deleted {model['name']}")
                        st.rerun()
    else:
        st.info("No models installed yet")
    
    st.markdown("---")
    
    # Recommended models
    st.subheader("🌟 Recommended Models")
    
    for model in manager.RECOMMENDED_MODELS:
        with st.expander(f"{model['name']} - {model['size']}"):
            st.write(f"**Description:** {model['description']}")
            st.write(f"**Size:** {model['size']}")
            
            if st.button(f"📥 Install {model['name']}", key=f"install_{model['name']}"):
                with st.spinner(f"Downloading {model['name']}..."):
                    if manager.install_model(model['name']):
                        st.success(f"✅ Installed {model['name']}")
                        st.rerun()
                    else:
                        st.error("Installation failed")

                        
def api_keys_config():
    """API Keys configuration"""
    st.header("🔑 API Key Management")
    st.markdown("Configure API keys for different LLM providers")
    
    st.warning("⚠️ API keys are stored in `.env` file. Changes here are for display only.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Groq API")
        groq_key = st.text_input(
            "Groq API Key",
            value=settings.GROQ_API_KEY[:20] + "..." if settings.GROQ_API_KEY else "",
            type="password",
            disabled=True
        )
        st.caption("Used for attacker and judge models")
        
        if st.button("Test Groq Connection"):
            with st.spinner("Testing..."):
                status = test_groq_connection()
                if status:
                    st.success("✅ Groq API connection successful!")
                else:
                    st.error("❌ Failed to connect to Groq API")
    
    with col2:
        st.subheader("NVIDIA API")
        nvidia_key = st.text_input(
            "NVIDIA API Key",
            value=settings.NVIDIA_API_KEY[:20] + "..." if settings.NVIDIA_API_KEY else "",
            type="password",
            disabled=True
        )
        st.caption("Used for target model testing")
        
        if st.button("Test NVIDIA Connection"):
            with st.spinner("Testing..."):
                status = test_nvidia_connection()
                if status:
                    st.success("✅ NVIDIA API connection successful!")
                else:
                    st.error("❌ Failed to connect to NVIDIA API")
    
    st.markdown("---")
    st.info("💡 **Tip:** Update API keys in the `.env` file in the project root directory")


def target_models_config():
    """Target models configuration"""
    st.header("🎯 Target Model Configuration")
    
    st.markdown("### Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NVIDIA Models")
        nvidia_models = [
            "meta/llama3-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "meta/llama-3.1-70b-instruct"
        ]
        for model in nvidia_models:
            st.checkbox(model, value=True, disabled=True, key=f"nvidia_{model}")
    
    with col2:
        st.subheader("Groq Models")
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        for model in groq_models:
            st.checkbox(model, value=True, disabled=True, key=f"groq_{model}")
    
    st.markdown("---")
    
    st.markdown("### Default Target Model")
    
    # Safe attribute access
    current_provider = getattr(settings, 'TARGET_MODEL_PROVIDER', 
                               getattr(settings, 'TARGET_PROVIDER', 'nvidia'))
    
    default_provider = st.selectbox(
        "Provider",
        ["nvidia", "groq"],
        index=0 if current_provider == "nvidia" else 1,
        disabled=True
    )
    
    default_model = st.selectbox(
        "Model",
        nvidia_models if default_provider == "nvidia" else groq_models,
        disabled=True
    )
    
    # Safe display - show available settings
    try:
        st.caption(f"Current Provider: {current_provider}")
        st.caption(f"Configuration loaded from .env file")
    except:
        st.caption("Configuration: Using defaults")



def judge_settings_config():
    """Judge settings configuration"""
    st.header("🧑‍⚖️ Judge Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primary Judge")
        judge_model = st.selectbox(
            "Judge Model",
            ["llama-3.3-70b-versatile", "llama-3.1-70b-instant", "mixtral-8x7b-32768"],
            index=0,
            disabled=True
        )
        st.caption(f"Current: {settings.JUDGE_MODEL}")
        
        judge_temp = st.slider(
            "Temperature",
            0.0, 1.0, float(settings.JUDGE_TEMPERATURE),
            disabled=True
        )
        st.caption("Lower = more consistent, Higher = more creative")
    
    with col2:
        st.subheader("Consensus Settings")
        use_consensus = st.checkbox("Enable Multi-Judge Consensus", value=False)
        
        if use_consensus:
            num_judges = st.number_input("Number of Judges", 2, 5, 3)
            agreement_threshold = st.slider("Agreement Threshold", 0.5, 1.0, 0.7)
        
        st.caption("Use multiple models to vote on verdicts")
    
    st.markdown("---")
    
    st.subheader("Verdict Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        refused_threshold = st.slider(
            "Refused Confidence Threshold",
            0.5, 1.0, 0.8,
            help="Minimum confidence to classify as REFUSED"
        )
    
    with col2:
        jailbreak_threshold = st.slider(
            "Jailbreak Confidence Threshold",
            0.5, 1.0, 0.7,
            help="Minimum confidence to classify as JAILBROKEN"
        )

def database_config():
    """Database configuration"""
    st.header("💾 Database Configuration")
    
    st.subheader("MongoDB Connection")
    
    mongo_uri = st.text_input(
        "MongoDB URI",
        value=settings.MONGODB_URI[:30] + "..." if settings.MONGODB_URI else "",
        type="password",
        disabled=True
    )
    
    mongo_db = st.text_input(
        "Database Name",
        value=settings.MONGODB_DB_NAME,
        disabled=True
    )
    
    if st.button("Test Database Connection"):
        with st.spinner("Testing connection..."):
            status = test_database_connection()
            
            if status:
                st.success("✅ Database connection successful!")
            else:
                st.error("❌ Failed to connect to database")
    
    st.markdown("---")
    
    st.subheader("Database Statistics")
    
    # Fetch real statistics
    if st.button("🔄 Refresh Statistics"):
        st.rerun()
    
    with st.spinner("Loading statistics..."):
        stats = fetch_database_statistics()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Evaluations", 
                stats.get('total_evaluations', 0),
                help="Total evaluations in database"
            )
        
        with col2:
            st.metric(
                "Total Attacks", 
                stats.get('total_attacks', 0),
                help="Total attacks recorded"
            )
        
        with col3:
            st.metric(
                "Database Size", 
                stats.get('database_size', 'Unknown'),
                help="Approximate database size"
            )
        
        # Additional stats
        st.markdown("#### Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Jailbroken:** {stats.get('jailbroken', 0)}")
        with col2:
            st.write(f"**Partial:** {stats.get('partial', 0)}")
        with col3:
            st.write(f"**Refused:** {stats.get('refused', 0)}")
    else:
        st.info("No statistics available. Run some audits first!")



def test_database_connection():
    """Test database connection synchronously"""
    
    async def _test():
        try:
            await db.connect()
            await db.client.admin.command('ping')
            await db.disconnect()
            return True
        except Exception as e:
            logger.error(f"DB test failed: {e}")
            try:
                await db.disconnect()
            except:
                pass
            return False
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_test())
    finally:
        loop.close()


def fetch_database_statistics():
    """Fetch real database statistics"""
    
    async def _fetch():
        try:
            await db.connect()
            
            # Count documents
            evaluations_count = await db.db.evaluations.count_documents({})
            attacks_count = await db.db.attacks.count_documents({})
            
            # Get verdict breakdown
            evaluations = await db.get_evaluations_by_filter(limit=10000)
            
            jailbroken = sum(1 for e in evaluations if e.get('verdict') == 'JAILBROKEN')
            partial = sum(1 for e in evaluations if e.get('verdict') == 'PARTIAL')
            refused = sum(1 for e in evaluations if e.get('verdict') == 'REFUSED')
            
            # Estimate database size
            stats = await db.client.admin.command('dbStats')
            db_size_bytes = stats.get('dataSize', 0)
            
            if db_size_bytes < 1024:
                db_size = f"{db_size_bytes} B"
            elif db_size_bytes < 1024 * 1024:
                db_size = f"{db_size_bytes / 1024:.2f} KB"
            else:
                db_size = f"{db_size_bytes / (1024 * 1024):.2f} MB"
            
            await db.disconnect()
            
            return {
                'total_evaluations': evaluations_count,
                'total_attacks': attacks_count,
                'database_size': db_size,
                'jailbroken': jailbroken,
                'partial': partial,
                'refused': refused
            }
            
        except Exception as e:
            st.error(f"Error fetching statistics: {e}")
            try:
                await db.disconnect()
            except:
                pass
            return None
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch())
    finally:
        loop.close()

def advanced_config():
    """Advanced configuration with real functionality"""
    st.header("🔧 Advanced Settings")
    
    # Load current settings from session state or use defaults
    if 'advanced_settings' not in st.session_state:
        st.session_state.advanced_settings = {
            'max_workers': 5,
            'timeout': 30,
            'retry_attempts': 3,
            'cache_enabled': True,
            'log_level': 'INFO',
            'log_to_file': True,
            'log_path': 'logs/app.log'
        }
    
    settings_dict = st.session_state.advanced_settings
    
    st.subheader("⚡ Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_workers = st.number_input(
            "Max Parallel Workers", 
            1, 10, 
            settings_dict['max_workers'],
            help="Number of concurrent attacks (3-5 recommended for free APIs)"
        )
        st.caption("⚠️ Higher values = faster but more API usage")
        
        timeout = st.number_input(
            "Request Timeout (seconds)", 
            10, 120, 
            settings_dict['timeout'],
            help="Maximum wait time for API responses"
        )
        st.caption("💡 Increase if you get timeout errors")
    
    with col2:
        retry_attempts = st.number_input(
            "Retry Attempts", 
            1, 5, 
            settings_dict['retry_attempts'],
            help="Number of retries for failed requests"
        )
        st.caption("🔄 Helps with transient API errors")
        
        cache_enabled = st.checkbox(
            "Enable Response Caching", 
            value=settings_dict['cache_enabled'],
            help="Cache identical prompts to reduce API costs"
        )
        st.caption("💰 Saves money by caching responses")
    
    st.markdown("---")
    
    st.subheader("📝 Logging")
    
    log_level = st.selectbox(
        "Log Level",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings_dict['log_level']),
        help="DEBUG = verbose, INFO = standard, WARNING/ERROR = minimal"
    )
    
    log_to_file = st.checkbox(
        "Log to File", 
        value=settings_dict['log_to_file'],
        help="Save logs to file for troubleshooting"
    )
    
    if log_to_file:
        log_path = st.text_input(
            "Log File Path", 
            value=settings_dict['log_path'],
            help="Where to save log files"
        )
    else:
        log_path = settings_dict['log_path']
    
    # Save settings button
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            # Update session state
            st.session_state.advanced_settings = {
                'max_workers': max_workers,
                'timeout': timeout,
                'retry_attempts': retry_attempts,
                'cache_enabled': cache_enabled,
                'log_level': log_level,
                'log_to_file': log_to_file,
                'log_path': log_path
            }
            
            # Save to file
            save_advanced_settings(st.session_state.advanced_settings)
            
            st.success("✅ Settings saved! Restart app to apply changes.")
            st.info("💡 Changes will take effect on next run.")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            # Reset to defaults
            st.session_state.advanced_settings = {
                'max_workers': 5,
                'timeout': 30,
                'retry_attempts': 3,
                'cache_enabled': True,
                'log_level': 'INFO',
                'log_to_file': True,
                'log_path': 'logs/app.log'
            }
            
            # Clear any cached values
            if 'settings_changed' in st.session_state:
                del st.session_state.settings_changed
            
            st.success("✅ Reset to defaults!")
            st.rerun()

    
    st.markdown("---")
    
    st.subheader("📦 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Configuration"):
            config = {
                "advanced_settings": st.session_state.advanced_settings,
                "exported_at": datetime.now().isoformat()
            }
            
            st.download_button(
                "⬇️ Download config.json",
                data=json.dumps(config, indent=2),
                file_name=f"llm_auditor_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ Configuration ready to download!")
    
    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type=['json'])
        if uploaded_file:
            try:
                config = json.loads(uploaded_file.read())
                if 'advanced_settings' in config:
                    st.session_state.advanced_settings = config['advanced_settings']
                    st.success("✅ Configuration imported!")
                    st.rerun()
                else:
                    st.error("❌ Invalid configuration file")
            except Exception as e:
                st.error(f"❌ Error importing: {e}")
    
    # Show current active settings
    st.markdown("---")
    st.subheader("📊 Current Active Settings")
    
    with st.expander("View Active Configuration"):
        st.json(st.session_state.advanced_settings)


def save_advanced_settings(settings_dict):
    """Save advanced settings to file"""
    try:
        import os
        os.makedirs('config', exist_ok=True)
        
        with open('config/advanced_settings.json', 'w') as f:
            json.dump(settings_dict, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False


def load_advanced_settings():
    """Load advanced settings from file"""
    try:
        with open('config/advanced_settings.json', 'r') as f:
            return json.load(f)
    except:
        return None



def test_groq_connection():
    """Test Groq API connection"""
    try:
        from backend.app.core.llm_clients import GroqClient
        client = GroqClient(settings.GROQ_API_KEY, "llama-3.1-8b-instant")
        response = client.generate("Hello", max_tokens=5)
        return len(response) > 0
    except:
        return False


def test_nvidia_connection():
    """Test NVIDIA API connection"""
    try:
        from backend.app.core.llm_clients import NVIDIAClient
        client = NVIDIAClient(settings.NVIDIA_API_KEY, "meta/llama3-70b-instruct")
        response = client.generate("Hello", max_tokens=5)
        return len(response) > 0
    except:
        return False


if __name__ == "__main__":
    main()
