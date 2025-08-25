from setuptools import setup, find_packages



with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lit_ecology_classifier',
    version='2.1.0',
    description='Image Classifier optimised for ecology use-cases',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'ImageHash',
        'torch==2.5.1',
        'torchvision',
        'torchaudio',
        'wandb==0.18.7',
        'lightning==2.2.5',
        'pytorch-lightning==2.5.1',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'timm==0.9.2',
        'safetensors',
        'scikit-learn',
        'distinctipy',   
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
            'isort',
            'pre-commit',
            'typeguard'
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
    },
    entry_points={
        'console_scripts': [
            'lit_ecology_classifier=lit_ecology_classifier.main:main',
        ],
    },
    author='Juan Ruiz, Benno Kaech',
    author_email='your.email@example.com',
    url='https://github.com/JuanFelipeRuiz/lit_ecology_classifier',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
