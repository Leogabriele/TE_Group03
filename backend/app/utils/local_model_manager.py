"""
Local model management utilities
"""

from pathlib import Path
from typing import List, Dict
import requests
from loguru import logger


class LocalModelManager:
    """Manage local LLM models"""
    
    # Recommended small models for testing (free)
    RECOMMENDED_MODELS = [
        {
            'name': 'llama3.2:1b',
            'size': '1.3 GB',
            'description': 'Meta Llama 3.2 1B - Fast, small model',
            'provider': 'ollama'
        },
        {
            'name': 'llama3.2:3b',
            'size': '2.0 GB',
            'description': 'Meta Llama 3.2 3B - Balanced performance',
            'provider': 'ollama'
        },
        {
            'name': 'phi3:mini',
            'size': '2.3 GB',
            'description': 'Microsoft Phi-3 Mini - Efficient',
            'provider': 'ollama'
        },
        {
            'name': 'gemma:2b',
            'size': '1.7 GB',
            'description': 'Google Gemma 2B - Lightweight',
            'provider': 'ollama'
        },
        {
            'name': 'mistral:7b',
            'size': '4.1 GB',
            'description': 'Mistral 7B - High quality',
            'provider': 'ollama'
        }
    ]
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
    
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_installed_models(self) -> List[Dict]:
        """Get list of installed local models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
        return []
    
    def install_model(self, model_name: str) -> bool:
        """Install a model via Ollama"""
        try:
            logger.info(f"📥 Installing {model_name}...")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes for download
            )
            
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if 'status' in data:
                        logger.info(data['status'])
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a local model"""
        try:
            response = requests.delete(
                f"{self.ollama_base_url}/api/delete",
                json={"name": model_name}
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed info about a model"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/show",
                json={"name": model_name}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
        return {}
    
    @staticmethod
    def get_ollama_install_instructions() -> str:
        """Get instructions for installing Ollama"""
        return """
# Install Ollama (Free & Easy)

## Windows:
1. Download from: https://ollama.ai/download/windows
2. Run the installer
3. Ollama will start automatically

## Linux:
curl -fsSL https://ollama.ai/install.sh | sh

## macOS:
brew install ollama

## Verify Installation:
ollama --version

## Start Ollama:
ollama serve
"""
