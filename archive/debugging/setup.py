#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="pokemon_crystal_rl",
    version="2.0.0",
    description="Pokemon Crystal RL Training System",
    author="",
    packages=['pokemon_crystal_rl'] + [
        'pokemon_crystal_rl.' + pkg for pkg in [
            'trainer',
            'core',
            'monitoring',
            'vision',
            'utils',
            'agents'
        ]
    ],
    package_dir={'': '.'},
    install_requires=[
        "numpy",
        "pyboy",
        "pillow",
        "opencv-python",
        "gymnasium",  # Updated replacement for gym
        "ollama",    # For local LLM support
    ],
    python_requires=">=3.8",
)
