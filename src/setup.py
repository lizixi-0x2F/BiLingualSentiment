#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# 读取README作为长描述
with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 主版本号
__version__ = '0.1.0'

setup(
    name="ltc-ncp-va",
    version=__version__,
    author="LiZiXi",
    author_email="your.email@example.com",
    description="基于液态时间常数网络与神经电路策略的文本情感价效度回归模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ltc-ncp-va",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Chinese (Simplified)"
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "tqdm>=4.45.0",
        "scikit-learn>=0.22.0",
        "matplotlib>=3.2.0",
        "jieba>=0.42.1",
        "pyyaml>=5.3.0",
        "tensorboard>=2.2.0"
    ],
    entry_points={
        "console_scripts": [
            "ltc-train=src.train:main",
            "ltc-predict=src.emotion_predict:main",
            "ltc-evaluate=src.evaluate:main",
        ],
    },
    include_package_data=True,
) 