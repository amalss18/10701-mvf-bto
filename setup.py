from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


# get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()


# base requirements
install_requires = list(
    filter(
        lambda x: "github.com" not in x,
        open(path.join(here, "requirements.txt")).read().strip().split("\n"),
    )
)


# version tag
version = "0.0.0"


setup(
    name="mvf-bto",
    version=version,
    description="Multivariate time series forecasting project for 10-701.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    include_package_data=True,
)
