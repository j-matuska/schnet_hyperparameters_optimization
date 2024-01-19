#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:43:25 2023

@author: jamat
"""

import setuptools

setuptools.setup(
    name="kniznica_evaluation",
    version="0.1",
    packages=setuptools.find_packages("."),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas"        ]
    )