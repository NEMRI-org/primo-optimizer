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

# Standard libs
import logging
import os
import urllib.error
import urllib.request
import zipfile

# User-defined libs
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def unzip_file(zip_path: str, extract_path: str):
    """
    Unzips a file.

    Parameters
    -----------
    zip_path : str
        Path to zip archive

    extract_path : str
        Path where zip archive is to be extracted

    Returns
    --------
    None
    """
    if not os.path.exists(extract_path):
        os.mkdir(extract_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def download_file(local_path: str, url: str):
    """
    Checks whether a file exists at the path specified by local_path;
    if not, attempts to download it from the URL specified.

    Parameters
    -----------
    local_path : str
        The local path where the file is expected, or to be downloaded if not found
    url : str
        The URL from which the resource is to be downloaded

    Returns
    --------
    None

    Raises
    -------
    """

    if os.path.exists(local_path):
        LOGGER.debug(f"File: {local_path} already exists!")
        return

    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.HTTPError as e:
        raise_exception(
            f"Failed to download from: {url} or save to: {local_path}" + str(e),
            RuntimeError,
        )
