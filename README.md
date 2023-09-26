# bargaining

This repository goes with the paper, "Bargaining between sexes: outside options and leisure time in BaYaka and Agta", by Angarika Deb, Daniel Saunders, Christophe Heintz, Nikhil Chaudhary, Mark Dyble, Abigail Page, Gul Deniz Salali, Dan Smith. You can reproduce all figures and parameter estimates with this repository. We recommend downloading the respository as a zip file. Reproducing the analysis requires [Jupyter notebooks](https://jupyter.org/). The code was written by myself (Daniel Saunders) and all mistakes are mine alone.

The `.ipynb` files contain the main analysis and generate the figures. 
The `.py` files contain functions that are useful to simulating bargaining model or visualizing the results.
The trace files and the likelihood files are not necessary but they store expensive computations. You can optionally regenerate them with the `.ipynb` files or call them to skip the computations.
The `data_couples_Rfile.csv` and `Agta_Analysis2_couples_Rfile.csv` files contain the original data.

## The Generative model analysis

To run the generative model analysis, you'll need an environment with the packages below. We recommend installing them with conda using the following command:

    conda install -c conda-forge pandas matplotlib numpy scipy seaborn

## The Bayesian GLM analysis

The Bayesian analysis needs a seperate environment with the packages below

    conda install -c conda-forge pymc arviz pandas matplotlib numpy
