from setuptools import find_packages, setup

NAME = 'kaggle_tools'
VERSION = '0.0.3'
AUTHOR = 'Yassine Alouini'
DESCRIPTION = """This is a suite of tools to help you participate in various Kaggle competitions"""
EMAIL = "yassinealouini@outlook.com"
URL = ""

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    url=URL,
    license="MIT",
    keywords="kaggle machine-learning",
)
