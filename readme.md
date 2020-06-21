# Modeling Tools for Synthetic Biology (MotoSyb) ?
A tool that can be used to go from synthetic biology descriptions to stochastic or deterministic models. See `notebooks/CC_Toolkit_Repressilator.ipynb` for an example with the repressilator.

Biological parts are defined easy, for example:
`plac = {'name':'P_lac', 'start':1, 'end':10, 'type':'Promoter', 'opts': {'color':[0.38, 0.82, 0.32]}}`


### Dependencies
* [`dnaplotlib`](https://github.com/VoigtLab/dnaplotlib)
* `scipy/numpy/matplotlib`


