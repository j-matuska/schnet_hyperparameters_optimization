# SchNetPack hyperparameter optimization for a more reliable top docking scores prediction.
Data for article: SchNetPack hyperparameter optimization for a more reliable top docking scores prediction.

## Neural network Schnet

In folder `calculation_NN` are 
* Config YAML files used to set the traning of neural network. YAML is Hydra configuration file used by package [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack). Please refer to manual of SchNetPack on usage.
* Binary files `best_model` containing neural network model trained in SchNetPack. They can by loaded by torch.
* File `split.npz`

Training and testing dataset are stored on [doi.org/10.5061/dryad.zgmsbccg7](https://doi.org/10.5061/dryad.zgmsbccg7)


## Evaluation of loss landscape

In folder `calculation_ll` is version of `ip_exlorer` used in article. It is modification of the original code [ip_explorer](https://github.com/jvita/data_efficiency_in_IAPS.git)
to create interface to analyze docking score prediction.
For instalation just change directory to calculation_ll and install using pip
```
cd calulation_ll
pip install -e .
```
In folder `scripts` are scripts dependend on `ip_explorer` used to calculate loss lanscape.

## Figures

All necesarry scripts and data to plot a figures in article are in folder `figures`
* `figueres`
    * `scripts/`: scripts to plot a figures
    * `modules/`: python modules used by scripts. Instalation by pip install -e .
    * `data/`: csv and npy files with results from calculations
    
# Authors
* Jan Matúška
* Lukáš Bučinský
* Marián Gáll
* Michal Pitonák
* Marek Štekláč


