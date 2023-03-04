from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "black<=21.12b0",
    "datasets>=1.7.0",
    "flake8",
    "isort==5.8.0",
    "pytest",
    "pyyaml>=5",
    "streamlit==0.82",
    "jinja2",
    "plotly",
    "requests",
    "pandas",
    ##############################################################
    # Dependencies in this section are added for specific datasets
    ##############################################################
    "py7zr",
    ##############################################################
    # End of dataset-specific dependencies
    ##############################################################
]

setup(
    name='promptsource',
    version='0.2.3',
    url='https://github.com/bigscience-workshop/promptsource.git',
    author='BigScience - Prompt Engineering Working Group',
    author_email='sbach@cs.brown.edu,victor@huggingface.co',
    python_requires='>=3.7,<3.10',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description='An Integrated Development Environment and Repository for Natural Language Prompts.',
    packages=find_packages(),
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_data={"": [
        "templates/*/*.yaml",
        "templates/*/*/*.yaml",
    ]}
)
