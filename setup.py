from setuptools import setup, find_packages

setup(
    name='pokemon_crystal_rl',
    version='0.1.0',
    description='Pokemon Crystal Reinforcement Learning Agent',
    author='AI Research Team',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'pyboy',
        'pillow',
        'pytesseract',
        'opencv-python',
        'psutil',
    ],
    python_requires='>=3.8',
)
