from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="cosmos-explorer",
    version="1.0.0",
    author="NASA Space Apps Challenge Team",
    author_email="example@example.com",
    description="An exoplanet detection application using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cosmos-explorer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cosmos-explorer=app:main",
        ],
    },
)