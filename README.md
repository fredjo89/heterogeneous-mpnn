# Heterogeneous MPNN
This repository contains implementation of a heterogeneous version of the GNN method MPNN with running code. <br>
The notebook `train_model.ipynb` contains the running code-example. <br>
The implementation of Heterogeneous MPNN for 1-4 layers is in `models_hmct.py`. <br>

# Creating the environment
These are instrictions to create a conda environment to run the code in Jupyter Notebook. 

1. Create a conda environment named gnn_env: `conda create -n gnn_env python=3.9`
2. Activate the environment: `conda activate gnn_env`
3. Install Black formatter: `pip install jupyter-black jupyter`
4. Install PyTorch and dependencies: `conda install pyg -c pyg`
5. Install pandas, matplotlib and Iphkernel: `conda install pandas matplotlib ipykernel`
6. Add the virtual environment to jupyter: `python -m ipykernel install --user --name=gnn_env`
  
 


# Remove environment
First deactivate conda environment:`conda deactivate`.
Remove conda enviroment with `conda env remove -n gnn_env` and the jupyter kerlen with `jupyter kernelspec uninstall gnn_env`
