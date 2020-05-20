from io import open
from setuptools import setup, find_packages

# from pip.req import parse_requirements

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="fast_bert",
    version="1.6.2",
    description="AI Library using BERT",
    author="Kaushal Trivedi",
    author_email="kaushaltrivedi@me.com",
    license="Apache2",
    url="https://github.com/kaushaltrivedi/fast-bert",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="BERT NLP deep learning google",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
