from setuptools import setup, find_packages

try:
    from meld_graph import __author__, __maintainer__, __email__, __version__
except ImportError:
    __author__ = __maintainer__ = "MELD development team"
    __email__ = "meld.study@gmail.com"
    __version__ = "2.2.2"

setup(
    name="meld_graph",
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description="Graph-based lesion segmentation for the MELD project.",
    license="MIT",
    packages=find_packages(),
    install_requires=[],
    package_dir={"meld_graph": "meld_graph"},
)
