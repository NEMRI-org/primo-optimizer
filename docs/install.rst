Getting Started
===============

To install the PRIMO framework on a Windows operating system, users should follow the instructions below 
that best suit their needs. 
If difficulties are encountered during the installation process, please email primo@netl.doe.gov for assistance.


Installation Prerequisites
--------------------------

While not mandatory, `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
should be installed, and users should use the ``conda`` command to create a separate Python environment for installing the PRIMO Toolkit.

1. Download installer from `Anaconda <https://www.anaconda.com/products/individual#Downloads>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
2. Install Anaconda using the downloaded installer.
3. Open the Anaconda prompt (click on the Start menu and search for the Anaconda prompt).


Installation
------------
The installation process varies depending on how PRIMO will be used. 

For users who want to use PRIMO for their own projects and do not contribute to its development, 
the instructions to :ref:`install PRIMO using a ZIP file <zip_file>` of the PRIMO repository should be followed. 

For developers who are planning to actively contribute to the PRIMO repository, the instructions to 
:ref:`install PRIMO using GitHub <git_hub>` should be followed. 
This includes users who handle protected data and contribute to the development of PRIMO simultaneously.

Step 1: Get Source Code
^^^^^^^^^^^^^^^^^^^^^^^^
.. _zip_file:

Installation using a ZIP File
``````````````````````````````

To download the ZIP file of PRIMO, visit the PRIMO repository on GitHub, click the "Code" icon, 
and select "Download ZIP." 
Once the download is complete, place the ZIP file in a preferred location and extract all files. After extraction, refer to the
:ref:`Create the Python Environment <environment>` section for the next step.


.. _git_hub:

Installation using GitHub
``````````````````````````
**Prerequisites**

Before proceeding with the installation via GitHub, ensure that the following prerequisites are in place:

1. **GitHub Account**: Create an account on `GitHub <https://github.com/>`_. 

2. **Git Installation**: Download and install `Git <https://git-scm.com/download/win>`_ . For detailed instructions, refer to the `Set up Git <https://docs.github.com/en/get-started/getting-started-with-git/set-up-git>`_ page on GitHub.

**Fork the Repo and Clone that Fork**

1. **Create a Fork**: Navigate to the PRIMO repository on GitHub and create a fork.

2. **Navigate to Directory**: Open the Anaconda command prompt and change to the preferred directory to store the source code: ::

        cd your\path

3. **Clone the Fork**: Clone the fork locally using the following command: ::

        git clone https://github.com/<USERNAME>/primo-optimizer.git
        cd primo-optimizer
   
  Users should replace the '*<USERNAME>*' with their GitHub username.

**Add Upstream Remote**

To synchronize the fork with the main PRIMO repository, add it as a remote by executing the following command: ::
    
    git remote add upstream https://github.com/NEMRI-org/primo-optimizer.git

To verify whether the remote has been added correctly, run: ::
    
    git remote -v

Upon successful addition, users should see two remotes listed---origin and upstream---displaying their respective remote names, URLs, and access (fetch or push).

.. _environment:

Step 2: Create the Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the Anaconda command prompt, follow these steps to create and activate a new conda environment. Please skip to the next step if 
you are a developer contributing actively to the PRIMO codebase ::
    
    conda env create -f conda-env.yml
    conda activate primo

For users who are contributing to the development of PRIMO, create a new environment with developer dependencies as follows ::

    conda env create -f conda-env-dev.yml
    conda activate primo

If the user handles protected data and contributes to PRIMO development, it is advisable to maintain separate environments for each task.

.. note::
    After creating the conda environment, each time the user opens a new terminal window and wants to use PRIMO, 
    they should activate the environment using: ``conda activate primo``. This ensures that the user is working within the correct environment setup for PRIMO.

.. _finish:

Step 3: Finish the Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Developers should set up pre-commit checks that will automatically run when using ``git commit``::
    
    pre-commit install

These steps ensure that all necessary dependencies are installed and pre-commit checks are configured for development tasks in PRIMO.

Since PRIMO identifies P&A projects by solving an optimization problem, it requires a suitable Mixed Integer Linear Programming (MILP) solver. 
The environment created in Step 2 already includes the free solver SCIP to solve PRIMO's optimization problems.

Users can also employ other commercial solvers, for example Gurobi, to solve the optimization problem. 
However, users are responsible for configuring and setting up these solvers themselves.