#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Computing-Intelligence-Contest Code",
    author="zhuyuedluter",
    author_email="516872091@qq.com",
    url="https://github.com/zhuyuedlut/Computing-Intelligence-Contest",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
