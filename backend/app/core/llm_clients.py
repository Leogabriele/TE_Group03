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
from backend.app.core.local_llm_clients import OllamaClient as LocalOllamaClient, get_local_llm_client


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


class LocalLLMClient(BaseLLMClient):
    """Wrapper for local LLM clients (Ollama, Hugging Face)"""
    
    def __init__(self, provider: str = "ollama", model_name: str = "llama3.2:1b"):
        super().__init__(api_key=None, model_name=model_name)
        self.provider = provider
        self.local_client = None
        
        # Initialize local client
        if provider == "ollama":
            self.local_client = LocalOllamaClient()  # ✅ Use the correct one from local_llm_clients.py
            logger.info(f"✅ Initialized Local Ollama client: {model_name}")
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
                # ✅ FIX: Call with correct parameters matching local_llm_clients.py
                result = self.local_client.generate(
                    model=self.model_name,  # ✅ This is correct for LocalOllamaClient
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
            
            # ✅ FIX: result is a Dict with 'success' key
            if isinstance(result, dict) and result.get('success'):
                self.request_count += 1
                self.total_tokens += result.get('tokens', 0)
                return result['response']
            elif isinstance(result, dict):
                raise Exception(f"Local generation failed: {result.get('error', 'Unknown error')}")
            else:
                # Fallback if somehow it's a string
                return str(result)
                
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
    def _resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
        """Resolve the correct API key for a cloud provider."""
        if api_key:
            return api_key
        return settings.get_api_key_for_provider(provider)
    
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        is_local: bool = False
    ) -> BaseLLMClient:
        """Create LLM client based on provider"""
        # Local models
        if is_local:
            return LocalLLMClient(provider=provider, model_name=model_name)
        
        # Cloud providers
        if provider.lower() == "groq":
            api_key = LLMClientFactory._resolve_api_key("groq", api_key)
            return GroqClient(api_key=api_key, model_name=model_name)
        
        elif provider.lower() == "nvidia":
            api_key = LLMClientFactory._resolve_api_key("nvidia", api_key)
            return NVIDIAClient(api_key=api_key, model_name=model_name)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def create_attacker() -> BaseLLMClient:
        """Create attacker LLM client"""
        return LLMClientFactory.create(
            provider=settings.ATTACKER_PROVIDER,
            model_name=settings.ATTACKER_MODEL,
            api_key=settings.get_api_key_for_provider(settings.ATTACKER_PROVIDER)
        )
    
    @staticmethod
    def create_judge() -> BaseLLMClient:
        """Create judge LLM client"""
        return LLMClientFactory.create(
            provider=settings.JUDGE_PROVIDER,
            model_name=settings.JUDGE_MODEL,
            api_key=settings.get_api_key_for_provider(settings.JUDGE_PROVIDER)
        )
    
    @staticmethod
    def create_target(is_local: bool = False) -> BaseLLMClient:
        """Create target LLM client (cloud or local)"""
        return LLMClientFactory.create(
            provider=settings.TARGET_MODEL_PROVIDER,
            model_name=settings.TARGET_MODEL_NAME,
            is_local=is_local
        )


class UnslothClient(BaseLLMClient):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = "unsloth-local"
        self._request_count = 0

    def generate(self, prompt: str, temperature=0.7, max_tokens=200,**kwargs) -> str:
        self._request_count += 1

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        prompt_length = inputs.input_ids.shape[1]
        response_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        return response_text.strip()

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
                lambda: self.generate(prompt, temperature, max_tokens)
            )
            return response
        except Exception as e:
            logger.error(f"❌ Unsloth  error (async): {e}")
            raise

    def get_stats(self):
        return {
            "provider": "unsloth",
            "model": self.model_name,
            "requests": self._request_count
        }
    