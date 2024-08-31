Examples
========

PRIMO currently has the following Jupyter Notebook demo available:

`PRIMO - Example_1 <https://github.com/NEMRI-org/primo-optimizer/blob/main/primo/demo/PRIMO%20-%20Example_1.ipynb>`_

More demos are forthcoming. In order to run the example and check the results, users need to:

- :doc:`Install PRIMO <install>`
- :doc:`Provide necessary inputs to PRIMO <method/primo_input>` and :doc:`update the config file if desirable <method/config_file>`
- Execute all cells in the Jupyter Notebook by clicking "Run"
- Check :ref:`PRIMO outputs`

.. note::
    The current example is developed using randomized data presented on a 2D x-y grid to demonstrate the PRIMO workflow and core capabilities. 
    Users are encouraged to adjust priority metrics, efficiency metrics and constraints. 

    Please note that PRIMO is fully capable of utilizing geospatial data in the form of shapefiles and process and visualize 
    well data based on latitudes and longitudes.

The optimization model in PRIMO is implemented using the Pyomo package. For users unfamiliar with Pyomo or Jupyter Notebook, additional information can be found using the following links:

- `Pyomo <https://www.pyomo.org/>`_
- `Jupyter Notebook <https://docs.jupyter.org/en/latest/>`_ 
