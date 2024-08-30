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
import os

# Installed libs
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


def process_notebook_with_config(
    notebook_path: str, config_file: str, output_dir: str, kernel_name: str
):
    """
    Process a Jupyter Notebook with a given configuration file and generate an HTML file of the Notebook and the Excel file that the notebook generates.

    Parameters
    ----------
    notebook_path : str
        Path to the Jupyter Notebook (.ipynb) file
    config_file : str
        Path to the configuration JSON file to be used in the notebook
    output_dir : str
        Directory where the output HTML file and any generated Excel files will be saved

    Returns
    -------
    None
    """
    # Read the notebook
    # The as_version=4 parameter specifies that the notebook should be read using the version 4 of the Jupyter Notebook format.
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    config_filename = os.path.basename(config_file)

    # Replace the config file path in the notebook
    for cell in nb.cells:
        if cell.cell_type == "code" and "config.json" in cell.source:
            cell.source = cell.source.replace("config.json", config_file)
        if cell.cell_type == "code" and "Primo_projects.xlsx" in cell.source:
            cell.source = cell.source.replace(
                "Primo_projects.xlsx",
                os.path.join(
                    output_dir,
                    f'Primo_projects_{config_filename.replace(".json", ".xlsx")}',
                ),
            )

    ep = ExecutePreprocessor(kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})

    # Convert the executed notebook to HTML
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True
    body, _ = html_exporter.from_notebook_node(nb)

    # Save the HTML file
    html_filename = os.path.join(output_dir, config_filename.replace(".json", ".html"))
    with open(html_filename, "w") as f:
        f.write(body)


def main(config_dir: str, notebook_path: str, output_dir: str, kernel_name: str):
    """
    Process a Jupyter Notebook with multiple configuration files and generate HTML reports for each.

    Parameters
    ----------
    config_dir : str
        Directory containing the configuration JSON files
    notebook_path : str
        Path to the Jupyter Notebook (.ipynb) file
    output_dir : str
        Directory where the output HTML files and any generated Excel files will be saved

    Returns
    -------
    None
    """
    config_files = [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".json")
    ]

    for config_file in config_files:
        process_notebook_with_config(
            notebook_path, config_file, output_dir, kernel_name
        )
