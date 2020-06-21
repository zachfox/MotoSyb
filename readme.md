# Modeling Tools for Synthetic Biology (MotoSyb) ?
A tool that can be used to go from synthetic biology descriptions to stochastic or deterministic models. See `notebooks/CC_Toolkit_Repressilator.ipynb` for an example with the repressilator.

Biological parts are defined easy, for example:
```plac = {'name':'P_lac', 'start':1, 'end':10, 'type':'Promoter', 'opts': {'color':[0.38, 0.82, 0.32]}}```
Rather than defining "by hand" the interactions between different parts, they are defined in python dictionaries. 
```lac_repress = {'from_part':laci, 'to_part':plac, 'type':'Repression', 'opts':{'linewidth':1, 'color':[0.38, 0.82, 0.32]}}```
The different types of interactions are in the file `codes/cyber_circuits.py`. 

From circuit descriptions, one can draw the circuit using SBOL-compliant symbols using `dnaplotlib`, create deterministic models, and run stochastic simulations. 
### Dependencies
* [`dnaplotlib`](https://github.com/VoigtLab/dnaplotlib)
* `scipy/numpy/matplotlib`


