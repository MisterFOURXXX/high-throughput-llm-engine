from setuptools import setup, find_packages

setup(
    name="high-throughput-llm-engine",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "artifacts*"]),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "polars>=0.19.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "docker>=6.0.0",
            "kubernetes>=28.1.0",
        ],
        "test": ["pytest>=7.0.0"],
    },
    python_requires=">=3.10",
)