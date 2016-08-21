from setuptools import find_packages, setup

NAME = 'kaggle_tools'
VERSION = '0.0.1'
AUTHOR = 'Yassine Alouini'
DESCRIPTION = """This is a suite of tools to help you participate in various
Kaggle competitions"""


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    description=DESCRIPTION,
    license="MIT",
    keywords="kaggle machine-learning",
)
