from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path: str) -> list[str]:
    requirements = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="open-unlearning",
    version="0.1.0",
    author="Vineeth Dorna, Anmol Mekala",
    author_email="vineethdorna@gmail.com, m.anmolreddy@gmail.com",
    description="DualCF extensions for selective factual unlearning, built on OpenUnlearning.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/vValkroVv/unlearning-acl",
    license="MIT",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install .[lm-eval]`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.15.8",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11",
)
