# VQM24
Some tools used for generating the VQM24 dataset

The PSI4 input file templates for DFT optimization+frequency calculation are present in the `template_DFT.in` file. The `scf_props.in` input file contains the template for subsequent single point SCF calculations for saving the wavefunction files and a few more one-electron properties which were not saved during optimization.

The python scripts are a few exemplary scripts out of many other which were used for submitting, retrieving, parsing, analyzing etc.

Machine learning scripts for training and prediction on atomization energies used to generate the learning curves presented in the manuscript with Graph Neural Networks (GNN) and Kernel ridge regression (KRR) are available in the `GNN_learning.py` and `KRR_learning.py` scripts.
