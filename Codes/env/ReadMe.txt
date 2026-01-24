How to setup the environment before you run the code?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. Install Miniconda or Anaconda


2. Type the following commands in the terminal (ubuntu/mac) or anaconda prompt (windows based) to create the virtual-environment in Anaconda
Choose one of the following options in given order: ('ap' is the name of our environment)

# Option 1 (Recommended)
$ conda env create -f environment.yml
$ source activate ap

# Option 2
$ conda create --name ap --file spec-file.txt
$ source activate ap
$ pip install -r requirements.txt

# Option 3 (bruteforce)
$ conda create --name ap anaconda
$ pip install pydot

# Reference for creating/managing environments in anaconda/miniconda:
LINK = https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Everytime you run the code, first activate the environment using command below:
$ source activate ap

To run .ipynb files, use jupyter:
$ jupyter-notebook
