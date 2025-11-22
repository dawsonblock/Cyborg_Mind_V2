from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="capsule_brain",
    version="0.1.0",
    author="Dawson Block",
    author_email="",
    description="Capsule Brain AGI framework with multi‑skill support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "temp_cyborg", "slides")),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "gym",
        "flask",
        "prometheus_client",
        "stable-baselines3",
        "minerl>=0.4.4",
        "openai",
        "torchaudio",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "capsule-brain-api=capsule_brain.api.app:main",
            "capsule-brain-gui=capsule_brain.gui.app:main",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)