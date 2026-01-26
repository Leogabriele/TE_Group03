"""
LLM Client Abstraction Layer
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
from loguru import logger
from groq import Groq
from openai import OpenAI
import httpx
from backend.app.config import settings
from backend.app.core.local_llm_clients import OllamaClient, get_local_llm_client


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = ""):
        self.api_key = api_key
        self.model_name = model_name
        self.request_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text synchronously"""
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text asynchronously"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "model": self.model_name,
            "requests": self.request_count,
            "total_tokens": self.total_tokens
        }


class GroqClient(BaseLLMClient):
    """Groq API client"""
    
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        super().__init__(api_key, model_name)
        self.client = Groq(api_key=api_key)
        logger.info(f"✅ Initialized Groq client: {model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text using Groq API"""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency = int((time.time() - start_time) * 1000)
            self.request_count += 1
            self.total_tokens += response.usage.total_tokens
            
            content = response.choices[0].message.content
            logger.debug(f"Groq call: {self.model_name} | Tokens: {response.usage.total_tokens} | {latency}ms")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text asynchronously using thread pool executor"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, temperature, max_tokens, **kwargs)
            )
            return response
        except Exception as e:
            logger.error(f"❌ Groq API error (async): {e}")
            raise


class NVIDIAClient(BaseLLMClient):
    """NVIDIA NIM API client"""
    
    def __init__(self, api_key: str, model_name: str = "meta/llama3-70b-instruct"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        logger.info(f"✅ Initialized NVIDIA client: {model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text using NVIDIA NIM API"""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency = int((time.time() - start_time) * 1000)
            self.request_count += 1
            
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            content = response.choices[0].message.content
            logger.debug(f"NVIDIA call: {self.model_name} | {latency}ms")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ NVIDIA API error: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text asynchronously"""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **kwargs
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                latency = int((time.time() - start_time) * 1000)
                self.request_count += 1
                
                if "usage" in data and data["usage"]:
                    self.total_tokens += data["usage"]["total_tokens"]
                
                content = data["choices"][0]["message"]["content"]
                logger.debug(f"NVIDIA async: {self.model_name} | {latency}ms")
                
                return content
                
        except Exception as e:
            logger.error(f"❌ NVIDIA API error (async): {e}")
            raise
class OllamaClient(BaseLLMClient):
    """Ollama local model client"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = None
        logger.info(f"✅ Initialized Ollama client: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Ollama"""
        import requests
        
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generate (uses sync for now)"""
        return self.generate(prompt, **kwargs)

class LocalLLMClient(BaseLLMClient):
    """Wrapper for local LLM clients (Ollama, Hugging Face)"""
    
    def __init__(self, provider: str = "ollama", model_name: str = "llama3.2:1b"):
        super().__init__(api_key=None, model_name=model_name)
        self.provider = provider
        self.local_client = None
        
        # Initialize local client
        if provider == "ollama":
            self.local_client = OllamaClient(model_name)
            logger.info(f"✅ Initialized Ollama client: {model_name}")
        elif provider == "huggingface":
            self.local_client = get_local_llm_client("huggingface")
            self.local_client.load_model(model_name)
            logger.info(f"✅ Initialized HuggingFace client: {model_name}")
        else:
            raise ValueError(f"Unknown local provider: {provider}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text using local LLM"""
        try:
            if self.provider == "ollama":
                result = self.local_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:  # huggingface
                result = self.local_client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if result.get('success'):
                self.request_count += 1
                self.total_tokens += result.get('tokens', 0)
                return result['response']
            else:
                raise Exception(f"Local generation failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Local LLM error: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text asynchronously (runs sync in executor)"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, temperature, max_tokens, **kwargs)
            )
            return response
        except Exception as e:
            logger.error(f"❌ Local LLM error (async): {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        is_local: bool = False  # NEW PARAMETER
    ) -> BaseLLMClient:
        """Create LLM client based on provider"""
        
        # Local models
        if is_local:
            return LocalLLMClient(provider=provider, model_name=model_name)
        
        # Cloud providers
        if provider.lower() == "groq":
            api_key = api_key or settings.GROQ_API_KEY
            return GroqClient(api_key=api_key, model_name=model_name)
        elif provider.lower() == "nvidia":
            api_key = api_key or settings.NVIDIA_API_KEY
            return NVIDIAClient(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def create_attacker() -> BaseLLMClient:
        """Create attacker LLM client"""
        return GroqClient(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.ATTACKER_MODEL
        )
    
    @staticmethod
    def create_judge() -> BaseLLMClient:
        """Create judge LLM client"""
        return GroqClient(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.JUDGE_MODEL
        )
    
    @staticmethod
    def create_target(is_local: bool = False) -> BaseLLMClient:
        """Create target LLM client (cloud or local)"""
        return LLMClientFactory.create(
            provider=settings.TARGET_MODEL_PROVIDER,
            model_name=settings.TARGET_MODEL_NAME,
            is_local=is_local
        )
