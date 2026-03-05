"""
Configuration Management Dashboard
Manage system settings and preferences
"""

import streamlit as st
import json
import sys
import requests
from pathlib import Path
from backend.app.config import settings
from backend.app.models.database import db
from backend.app.models.enums import LLMProvider
from loguru import logger
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")
st.title("⚙️ System Configuration")
st.markdown("**Manage system settings and preferences**")
st.markdown("---")


# ─── Dynamic Model Fetchers ───────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_groq_models() -> list:
    """Fetch available Groq models from API (cached 5 min)."""
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        logger.warning(f"Could not fetch Groq models: {e}")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_ollama_models() -> list:
    """Fetch locally installed Ollama models (cached 1 min)."""
    try:
        resp = requests.get(
            getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags",
            timeout=3
        )
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def _get_nvidia_models() -> list:
    """
    NVIDIA NIM has no public list endpoint.
    Returns known available models — update this list as NVIDIA adds new ones.
    Reads current config model first so it always appears as an option.
    """
    known = [
        "meta/llama3-70b-instruct",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.2-3b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "mistralai/mistral-7b-instruct-v0.3",
        "microsoft/phi-3-mini-128k-instruct",
        "nvidia/mistral-nemo-minitron-8b-instruct",
    ]
    # Always include the currently configured model even if not in list
    current = getattr(settings, "TARGET_MODEL_NAME", "")
    if current and current not in known:
        known.insert(0, current)
    return known


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    tabs = st.tabs([
        "🔑 API Keys",
        "🎯 Target Models",
        "🧑‍⚖️ Judge Settings",
        "💾 Database",
        "🏠 Local Models",
        "🔧 Advanced"
    ])

    with tabs[0]:
        api_keys_config()
    with tabs[1]:
        target_models_config()
    with tabs[2]:
        judge_settings_config()
    with tabs[3]:
        database_config()
    with tabs[4]:
        local_models_config()
    with tabs[5]:
        advanced_config()


# ─── Tab: Target Models ───────────────────────────────────────────────────────

def target_models_config():
    """Target models configuration — fully dynamic."""
    st.header("🎯 Target Model Configuration")

    # ── Provider Selection ──────────────────────────────────────────────────
    st.markdown("### Default Target Model")

    current_provider = getattr(settings, "TARGET_MODEL_PROVIDER", "nvidia").lower()

    # All supported providers from LLMProvider enum — no hardcoding
    all_providers = [p.value for p in LLMProvider]

    try:
        provider_index = all_providers.index(current_provider)
    except ValueError:
        provider_index = 0

    selected_provider = st.selectbox(
        "Provider",
        options=all_providers,
        index=provider_index,
        format_func=lambda x: x.upper(),
        help="Select the LLM provider for the target model"
    )

    # ── Model List — dynamic per provider ──────────────────────────────────
    if selected_provider == "groq":
        with st.spinner("Fetching Groq models..."):
            model_list = _fetch_groq_models()
        st.caption(f"✅ {len(model_list)} Groq models available")

    elif selected_provider == "nvidia":
        model_list = _get_nvidia_models()
        st.caption(f"ℹ️ {len(model_list)} NVIDIA NIM models listed")

    elif selected_provider == "ollama":
        with st.spinner("Fetching local Ollama models..."):
            model_list = _fetch_ollama_models()
        if not model_list:
            st.warning("⚠️ No Ollama models found. Is Ollama running?")
            manual = st.text_input(
                "Enter model name manually",
                value=getattr(settings, "ATTACKER_OLLAMA_MODEL", "llama3.2:latest")
            )
            model_list = [manual] if manual else ["llama3.2:latest"]
        else:
            st.caption(f"✅ {len(model_list)} local models detected")

    else:
        model_list = [getattr(settings, "TARGET_MODEL_NAME", "")]
        st.info(f"No model list available for provider: {selected_provider}")

    # Current model from settings
    current_model = getattr(settings, "TARGET_MODEL_NAME", "")
    try:
        model_index = model_list.index(current_model)
    except ValueError:
        model_index = 0

    selected_model = st.selectbox(
        "Model",
        options=model_list,
        index=model_index,
        help="Target model to audit"
    )

    st.caption(f"**Active config:** `{current_provider}/{current_model}` (from .env)")
    st.info("💡 To change the default, update `TARGET_MODEL_PROVIDER` and `TARGET_MODEL_NAME` in `.env`")

    st.markdown("---")

    # ── Available Models Overview ───────────────────────────────────────────
    st.markdown("### Available Models Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("☁️ NVIDIA")
        nvidia_list = _get_nvidia_models()
        for m in nvidia_list:
            st.caption(f"• {m}")

    with col2:
        st.subheader("⚡ Groq")
        if st.button("🔄 Refresh Groq Models", key="refresh_groq"):
            st.cache_data.clear()
            st.rerun()
        groq_list = _fetch_groq_models()
        for m in groq_list:
            st.caption(f"• {m}")

    with col3:
        st.subheader("🏠 Ollama (Local)")
        if st.button("🔄 Refresh Local Models", key="refresh_ollama"):
            st.cache_data.clear()
            st.rerun()
        ollama_list = _fetch_ollama_models()
        if ollama_list:
            for m in ollama_list:
                st.caption(f"• {m}")
        else:
            st.caption("No models detected")


# ─── Tab: Judge Settings ──────────────────────────────────────────────────────

def judge_settings_config():
    """Judge settings — model list fetched dynamically from Groq."""
    st.header("🧑‍⚖️ Judge Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primary Judge")

        # Fetch Groq models dynamically (judge always uses Groq)
        with st.spinner("Loading available judge models..."):
            judge_model_options = _fetch_groq_models()

        # Ensure current judge model is in the list
        current_judge = getattr(settings, "JUDGE_MODEL", "llama-3.3-70b-versatile")
        if current_judge not in judge_model_options:
            judge_model_options.insert(0, current_judge)

        try:
            judge_index = judge_model_options.index(current_judge)
        except ValueError:
            judge_index = 0

        judge_model = st.selectbox(
            "Judge Model",
            options=judge_model_options,
            index=judge_index,
            disabled=True,
            help="Configure JUDGE_MODEL in .env to change"
        )
        st.caption(f"Current: {current_judge}")

        judge_temp = st.slider(
            "Temperature",
            0.0, 1.0,
            float(getattr(settings, "JUDGE_TEMPERATURE", 0.0)),
            disabled=True
        )
        st.caption("Lower = more consistent verdicts")

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
            "Refused Confidence Threshold", 0.5, 1.0, 0.8,
            help="Minimum confidence to classify as REFUSED"
        )
    with col2:
        jailbreak_threshold = st.slider(
            "Jailbreak Confidence Threshold", 0.5, 1.0, 0.7,
            help="Minimum confidence to classify as JAILBROKEN"
        )


# ─── Unchanged functions (kept exactly as-is) ─────────────────────────────────

def local_models_config():
    """Local models configuration"""
    st.header("🏠 Local Model Management")
    st.markdown("Run LLMs locally for free using Ollama")

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
                    st.json(manager.get_model_info(model['name']))
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{model['name']}"):
                    if manager.delete_model(model['name']):
                        st.success(f"Deleted {model['name']}")
                        st.rerun()
    else:
        st.info("No models installed yet")

    st.markdown("---")
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
        st.text_input(
            "Groq API Key",
            value=settings.GROQ_API_KEY[:20] + "..." if settings.GROQ_API_KEY else "",
            type="password", disabled=True
        )
        st.caption("Used for attacker and judge models")
        if st.button("Test Groq Connection"):
            with st.spinner("Testing..."):
                if test_groq_connection():
                    st.success("✅ Groq API connection successful!")
                else:
                    st.error("❌ Failed to connect to Groq API")

    with col2:
        st.subheader("NVIDIA API")
        st.text_input(
            "NVIDIA API Key",
            value=settings.NVIDIA_API_KEY[:20] + "..." if settings.NVIDIA_API_KEY else "",
            type="password", disabled=True
        )
        st.caption("Used for target model testing")
        if st.button("Test NVIDIA Connection"):
            with st.spinner("Testing..."):
                if test_nvidia_connection():
                    st.success("✅ NVIDIA API connection successful!")
                else:
                    st.error("❌ Failed to connect to NVIDIA API")

    st.markdown("---")
    st.info("💡 **Tip:** Update API keys in the `.env` file in the project root directory")


def database_config():
    """Database configuration"""
    st.header("💾 Database Configuration")
    st.subheader("MongoDB Connection")

    st.text_input(
        "MongoDB URI",
        value=settings.MONGODB_URI[:30] + "..." if settings.MONGODB_URI else "",
        type="password", disabled=True
    )
    st.text_input("Database Name", value=settings.MONGODB_DB_NAME, disabled=True)

    if st.button("Test Database Connection"):
        with st.spinner("Testing connection..."):
            if test_database_connection():
                st.success("✅ Database connection successful!")
            else:
                st.error("❌ Failed to connect to database")

    st.markdown("---")
    st.subheader("Database Statistics")

    if st.button("🔄 Refresh Statistics"):
        st.rerun()

    with st.spinner("Loading statistics..."):
        stats = fetch_database_statistics()

    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Evaluations", stats.get('total_evaluations', 0))
        with col2:
            st.metric("Total Attacks", stats.get('total_attacks', 0))
        with col3:
            st.metric("Database Size", stats.get('database_size', 'Unknown'))

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


def advanced_config():
    """Advanced configuration"""
    st.header("🔧 Advanced Settings")

    if 'advanced_settings' not in st.session_state:
        st.session_state.advanced_settings = {
            'max_workers': 5, 'timeout': 30, 'retry_attempts': 3,
            'cache_enabled': True, 'log_level': 'INFO',
            'log_to_file': True, 'log_path': 'logs/app.log'
        }

    settings_dict = st.session_state.advanced_settings
    st.subheader("⚡ Performance")

    col1, col2 = st.columns(2)
    with col1:
        max_workers = st.number_input("Max Parallel Workers", 1, 10, settings_dict['max_workers'])
        st.caption("⚠️ Higher values = faster but more API usage")
        timeout = st.number_input("Request Timeout (seconds)", 10, 120, settings_dict['timeout'])
        st.caption("💡 Increase if you get timeout errors")
    with col2:
        retry_attempts = st.number_input("Retry Attempts", 1, 5, settings_dict['retry_attempts'])
        st.caption("🔄 Helps with transient API errors")
        cache_enabled = st.checkbox("Enable Response Caching", value=settings_dict['cache_enabled'])
        st.caption("💰 Saves money by caching responses")

    st.markdown("---")
    st.subheader("📝 Logging")

    # Log level from enum — no hardcoding
    from backend.app.config import Settings
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = st.selectbox(
        "Log Level",
        log_levels,
        index=log_levels.index(settings_dict['log_level'])
    )
    log_to_file = st.checkbox("Log to File", value=settings_dict['log_to_file'])
    log_path = st.text_input("Log File Path", value=settings_dict['log_path']) if log_to_file else settings_dict['log_path']

    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            st.session_state.advanced_settings = {
                'max_workers': max_workers, 'timeout': timeout,
                'retry_attempts': retry_attempts, 'cache_enabled': cache_enabled,
                'log_level': log_level, 'log_to_file': log_to_file, 'log_path': log_path
            }
            save_advanced_settings(st.session_state.advanced_settings)
            st.success("✅ Settings saved!")
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.session_state.advanced_settings = {
                'max_workers': 5, 'timeout': 30, 'retry_attempts': 3,
                'cache_enabled': True, 'log_level': 'INFO',
                'log_to_file': True, 'log_path': 'logs/app.log'
            }
            st.success("✅ Reset to defaults!")
            st.rerun()

    st.markdown("---")
    st.subheader("📦 Export/Import Configuration")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Configuration"):
            config = {"advanced_settings": st.session_state.advanced_settings, "exported_at": datetime.now().isoformat()}
            st.download_button(
                "⬇️ Download config.json",
                data=json.dumps(config, indent=2),
                file_name=f"llm_auditor_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type=['json'])
        if uploaded_file:
            try:
                config = json.loads(uploaded_file.read())
                if 'advanced_settings' in config:
                    st.session_state.advanced_settings = config['advanced_settings']
                    st.success("✅ Configuration imported!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error importing: {e}")

    st.markdown("---")
    with st.expander("View Active Configuration"):
        st.json(st.session_state.advanced_settings)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_advanced_settings(settings_dict):
    try:
        import os
        os.makedirs('config', exist_ok=True)
        with open('config/advanced_settings.json', 'w') as f:
            json.dump(settings_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return False


def test_database_connection():
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
    async def _fetch():
        try:
            await db.connect()
            evaluations_count = await db.db.evaluations.count_documents({})
            attacks_count = await db.db.attacks.count_documents({})
            evaluations = await db.get_evaluations_by_filter(limit=10000)
            jailbroken = sum(1 for e in evaluations if e.get('verdict') == 'JAILBROKEN')
            partial = sum(1 for e in evaluations if e.get('verdict') == 'PARTIAL')
            refused = sum(1 for e in evaluations if e.get('verdict') == 'REFUSED')
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
                'total_evaluations': evaluations_count, 'total_attacks': attacks_count,
                'database_size': db_size, 'jailbroken': jailbroken,
                'partial': partial, 'refused': refused
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


def test_groq_connection():
    try:
        from backend.app.core.llm_clients import GroqClient
        client = GroqClient(settings.GROQ_API_KEY, "llama-3.1-8b-instant")
        response = client.generate("Hello", max_tokens=5)
        return len(response) > 0
    except:
        return False


def test_nvidia_connection():
    try:
        from backend.app.core.llm_clients import NVIDIAClient
        client = NVIDIAClient(settings.NVIDIA_API_KEY, "meta/llama3-70b-instruct")
        response = client.generate("Hello", max_tokens=5)
        return len(response) > 0
    except:
        return False


if __name__ == "__main__":
    main()
