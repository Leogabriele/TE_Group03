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


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, api_key: str, model_name: str):
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
        """
        Generate text asynchronously using thread pool executor
        This is simpler and more reliable than manual HTTP calls
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            # Run synchronous method in executor to avoid blocking
            response = await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, temperature, max_tokens, **kwargs)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Groq API error (async): {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Local Ollama API client"""
    
    def __init__(self, model_name: str = "phi3:latest", base_url: str = "http://localhost:11434"):
        # API key is not required for local Ollama, so we pass a placeholder
        super().__init__(api_key="local", model_name=model_name)
        self.base_url = base_url
        self._initialized = False
        logger.info(f"✅ Initialized Ollama client (Local): {model_name}")
        import asyncio
    async def ensure_model_ready(self):
        """Ensure model is downloaded and ready (call this before first use)"""
        if self._initialized:
            return
        
        if not await self.check_model_exists():
            logger.warning(f"⚠️ Model {self.model_name} not available, downloading...")
            await self.pull_model()
        
        self._initialized = True
    async def check_model_exists(self) -> bool:
        """Check if the model is downloaded and available"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                # Check if model exists in the list
                model_names = [model["name"] for model in data.get("models", [])]
                exists = self.model_name in model_names
                
                if exists:
                    logger.info(f"✅ Model {self.model_name} is available")
                else:
                    logger.warning(f"⚠️ Model {self.model_name} not found. Available: {model_names}")
                
                return exists
                
        except Exception as e:
            logger.error(f"❌ Error checking model availability: {e}")
            return False
    async def pull_model(self, model_name: str = None) -> bool:
        """Download a model from Ollama registry"""
        model_to_pull = model_name or self.model_name
        
        try:
            logger.info(f"📥 Downloading model: {model_to_pull}")
            
            async with httpx.AsyncClient(timeout=600.0) as client:  # Longer timeout for downloads
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"model": model_to_pull, "stream": True}
                )
                response.raise_for_status()
                print("started the model download")
                # Stream the download progress
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        status = data.get("status", "")
                        print("getting the model")
                        
                        # Log progress updates
                        if "total" in data and "completed" in data:
                            percent = (data["completed"] / data["total"]) * 100
                            logger.info(f"⏳ {status}: {percent:.1f}%")
                        else:
                            logger.info(f"⏳ {status}")
                        
                        # Check if download is complete
                        if status == "success":
                            logger.info(f"✅ Successfully downloaded {model_to_pull}")
                            return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_to_pull}: {e}")
            return False


    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text using local Ollama API (Synchronous)"""
        import asyncio
        # Reusing the async logic for sync calls to maintain consistency
        return asyncio.run(self.generate_async(prompt, temperature, max_tokens, **kwargs))

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text asynchronously using Ollama's /api/chat endpoint"""
        try:
            start_time = time.time()
            await self.ensure_model_ready()
            
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            **kwargs
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()

            latency = int((time.time() - start_time) * 1000)
            
            self.request_count += 1
            # Note: Ollama provides 'prompt_eval_count' and 'eval_count' for tokens
            if "prompt_eval_count" in data:
                self.total_tokens += (data["prompt_eval_count"] + data["eval_count"])

            content = data["message"]["content"]
            logger.debug(f"Ollama async: {self.model_name} | {latency}ms")
            
            return content

        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}")
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


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None
    ) -> BaseLLMClient:
        """Create LLM client based on provider"""
        
        if provider.lower() == "groq":
            api_key = api_key or settings.GROQ_API_KEY
            return GroqClient(api_key=api_key, model_name=model_name)
        
        elif provider.lower() == "nvidia":
            api_key = api_key or settings.NVIDIA_API_KEY
            return NVIDIAClient(api_key=api_key, model_name=model_name)
        elif provider.lower() == "ollama":
            # Ollama uses local URL instead of API Key
            return OllamaClient(model_name=model_name)
        
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
    def create_target() -> BaseLLMClient:
        """Create target LLM client"""
        return LLMClientFactory.create(
            provider=settings.TARGET_MODEL_PROVIDER,
            model_name=settings.TARGET_MODEL_NAME
        )
if __name__ == "__main__":
    # Example usage
    ollama_client = LLMClientFactory.create(
        provider="ollama",
        model_name="phi3:latest"
    )
    response = ollama_client.generate("Hello, how are you?", temperature=0.5, max_tokens=50)
    print("Ollama Response:", response)