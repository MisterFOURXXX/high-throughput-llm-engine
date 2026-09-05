from setuptools import setup, find_packages

setup(
    name="high-throughput-llm-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "vllm>=0.2.0",
        "polars>=0.19.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "boto3>=1.26.0",
        "docker>=6.0.0",
    ],
)