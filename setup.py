from setuptools import setup, find_packages

setup(
    name="context-palace",
    version="0.1.0",
    description="Federated Context Lake & TUI CRUD Memory Palace with Auto-Detected Model Context Windows",
    author="Alejandro Londono",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "context-palace=context_palace.cli:main",
        ],
    },
)
