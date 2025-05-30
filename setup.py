#!/usr/bin/env python3
"""
LlamaDistributor安装脚本

基于QLLM的Llama模型自定义分层与分布式推理系统
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# 读取依赖文件
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="llamadist",
    version="0.1.0",
    description="基于QLLM的Llama模型自定义分层与分布式推理系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="LlamaDistributor Team",
    author_email="contact@llamadist.dev",
    url="https://github.com/llamadist/LlamaDistributor",
    
    packages=find_packages(),
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        "gpu": [
            "torch>=1.12.0",
        ],
        "quantization": [
            "bitsandbytes>=0.40.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "llamadist-partition=llamadist.cli.partition:main",
            "llamadist-inference=llamadist.cli.inference:main",
            "llamadist-analyze=llamadist.cli.analyze:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords=[
        "llama",
        "transformer",
        "distributed-inference",
        "model-partition",
        "quantization",
        "pytorch",
        "qllm",
        "nlp",
        "large-language-model"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/llamadist/LlamaDistributor/issues",
        "Source": "https://github.com/llamadist/LlamaDistributor",
        "Documentation": "https://llamadist.readthedocs.io/",
    },
) 