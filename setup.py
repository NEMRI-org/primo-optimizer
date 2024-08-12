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

# Installed lib
from setuptools import setup, find_packages

# User defined lib
from primo import VERSION


NAME = "primo-optimizer"


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
    name=NAME,
    version=VERSION,
    description="PRIMO - The P&A Project Optimizer",
    long_description=(
        "PRIMO - The P&A Project Optimizer toolkit was produced under the "
        " Methane Emissions Reduction Program (MERP) "
        " and National Energy Technology Laboratory's (NETL) "
        "National Emissions Reduction Initiative (NEMRI). "
        "The major deliverable of this project will be an open-source, optimization-based, "
        "downloadable and executable decision-support application, PRIMO, "
        "that can be run by companies, technology providers, research organizations, "
        "and regulators."
    ),
    long_description_content_type="text/plain",
    url="https://edx.netl.doe.gov/nemri/",
    author="PRIMO team",
    license="BSD",
    maintainer="Miguel Zamarripa",
    maintainer_email="Miguel.Zamarripa-Perez@netl.doe.gov",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3 :: Only",
    ],
    platforms=[
        "windows",
        "linux",
    ],
    python_requires=">=3.8, <=4.0",
    packages=find_packages(),
    py_modules=["stagedfright"],
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "test": TEST_REQUIREMENTS,
    },
    include_package_data=True,
    package_data={
        # If any package contains these files, include them:
        "": [
            "*.xlsx",
        ]
    },
    entry_points={
        "console_scripts": [
            "stagedfright=stagedfright:main",
        ]
    },
)
