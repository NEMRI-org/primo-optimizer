#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################
# Standard lib
from pathlib import Path

# Installed lib
from setuptools import setup, find_packages

# User defined lib
from primo import VERSION


NAME = "primo-optimizer"
CURRENT_DIRECTORY = Path(__file__).parent
LONG_DESCRIPTION = (CURRENT_DIRECTORY / "README.md").read_text()

# All requirements for the package to work successfully
REQUIREMENTS = [
    "appengine-python-standard",
    "censusgeocode",
    "folium",
    "geopandas",
    "gurobipy",
    "highspy",
    "haversine",
    "ipyfilechooser",
    "ipyleaflet",
    "ipywidgets",
    "kneed",
    "matplotlib",
    "notebook",
    "numpy<=1.28",
    "openpyxl",
    "pandas",
    "pyarrow",
    "pyomo",
    "pyscipopt",
    "python-dotenv",
    "scikit-learn",
    "rasterio",
    "xlsxwriter",
]

# All requirements for developers
DEV_REQUIREMENTS = [
    "addheader",
    "black",
    "myst-parser",
    "nbsphinx",
    "pre-commit",
    "sphinx",
    "sphinx-rtd-theme",
]

# All requirements for testing
TEST_REQUIREMENTS = ["pytest", "pytest-cov"]

setup(
    author="PRIMO team",
    author_email="primo@netl.doe.gov",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    description="PRIMO - The P&A Project Optimizer",
    entry_points={
        "console_scripts": [
            "stagedfright=stagedfright:main",
        ]
    },
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "test": TEST_REQUIREMENTS,
    },
    include_package_data=True,
    install_requires=REQUIREMENTS,
    keywords=[
        "PRIMO",
        "MERP",
        "NEMRI",
        "methane emissions",
        "optimization",
        "process modeling",
        "operations research",
        "well plugging",
    ],
    license="BSD",
    license_files=("LICENSE.md",),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    maintainer="PRIMO team",
    maintainer_email="primo@netl.doe.gov",
    name=NAME,
    package_data={
        # If any package contains these files, include them:
        "": [
            "*.xlsx",
        ]
    },
    packages=find_packages(),
    platforms=[
        "windows",
        "linux",
    ],
    project_urls={
        "Documentation": "https://primo.readthedocs.io/en/latest/",
        "Download": "https://github.com/NEMRI-org/primo-optimizer/releases",
        "Homepage": "https://edx.netl.doe.gov/nemri/",
        "Source": "https://github.com/NEMRI-org/primo-optimizer",
        "Tracker": "https://github.com/NEMRI-org/primo-optimizer/issues",
    },
    python_requires=">=3.8, <4.0",
    py_modules=["stagedfright"],
    version=VERSION,
)
