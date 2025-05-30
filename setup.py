# make a setup.py file for this package

from setuptools import setup, find_packages
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="skel_features",
    version=find_version("src", "skel_features", "__init__.py"),
    description="Feature extraction for meshwork neurons",
    author="Casey Schneider-Mizell",
    author_email="caseys@alleninstitute.org",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=required,
)