"""
Local LLM clients for running models locally
Uses Ollama for free local inference
"""

import requests
import subprocess
import json
from typing import Optional, Dict, List
from loguru import logger


class OllamaClient:
    """Client for Ollama local LLM inference"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self._check_ollama_status()
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                logger.info(f"✅ Ollama running with {len(self.available_models)} models")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Ollama not running: {e}")
            return False
        return False
    
    def list_models(self) -> List[str]:
        """List available local models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def pull_model(self, model_name: str) -> bool:
        """Download a model from Ollama library"""
        try:
            logger.info(f"📥 Downloading model: {model_name}")
            
            # Use streaming to show progress
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        logger.debug(data['status'])
            
            logger.info(f"✅ Downloaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60
    ) -> Dict:
        """Generate response from local model"""
        
        try:
            start_time = __import__('time').time()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                latency = int((__import__('time').time() - start_time) * 1000)
                
                return {
                    'response': data.get('response', ''),
                    'model': model,
                    'latency_ms': latency,
                    'tokens': data.get('eval_count', 0),
                    'success': True
                }
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return {'success': False, 'error': f"Status {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        return self._check_ollama_status()


class HuggingFaceLocalClient:
    """Client for Hugging Face Transformers (local inference)"""
    
    def __init__(self):
        self.pipeline = None
        self.current_model = None
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed"""
        try:
            import transformers
            import torch
            logger.info("✅ Hugging Face Transformers available")
            return True
        except ImportError:
            logger.warning("⚠️ Transformers not installed. Run: pip install transformers torch")
            return False
    
    def load_model(self, model_name: str = "gpt2") -> bool:
        """Load a model from Hugging Face"""
        try:
            from transformers import pipeline
            
            logger.info(f"📥 Loading model: {model_name}")
            
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=-1  # CPU only for free tier
            )
            
            self.current_model = model_name
            logger.info(f"✅ Loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict:
        """Generate response using loaded model"""
        
        if not self.pipeline:
            return {'success': False, 'error': 'No model loaded'}
        
        try:
            start_time = __import__('time').time()
            
            result = self.pipeline(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )
            
            latency = int((__import__('time').time() - start_time) * 1000)
            
            return {
                'response': result[0]['generated_text'],
                'model': self.current_model,
                'latency_ms': latency,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {'success': False, 'error': str(e)}


# Factory function
def get_local_llm_client(provider: str = "ollama"):
    """Get local LLM client"""
    
    if provider == "ollama":
        return OllamaClient()
    elif provider == "huggingface":
        return HuggingFaceLocalClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
