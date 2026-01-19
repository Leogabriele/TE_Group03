"""
Setup configuration for LLM Security Auditor
"""

from setuptools import setup, find_packages

setup(
    name="llm-security-auditor",
    version="0.1.0",
    description="Automated adversarial testing framework for LLMs",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "groq>=0.4.0",
        "openai>=1.0.0",
        "httpx>=0.25.0",
        "motor>=3.3.0",
        "pymongo>=4.6.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
