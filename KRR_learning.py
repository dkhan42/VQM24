'''
Example script for generating the atomization energy learning curves using Kernel Ridge Regression (KRR) and FCHL19 or cMBDF (https://github.com/dkhan42/cMBDF) representations.
This script relies on the qmlcode-develop library (https://www.qmlcode.org/), the wrapper for using it : https://github.com/dkhan42/QMLwrap and the cMBDF script : https://github.com/dkhan42/cMBDF
'''
import numpy as np
from qml.representations import generate_fchl_acsf
from cMBDF import generate_mbdf
from KernelRidge import GridSearchCV, KRR_global
from KernelRidge import GridSearchCV_local, KRR_local

#Load VQM24 dataset (only most stable conformers)
data = np.load('DFT_uniques.npz',allow_pickle = True)
charges, coords, labels = data['atoms'], data['coordinates'], data['Eatomization']*627.5 #Hartree to kcal/mol conversion of atomization energies

#generate cmbdf representation for all molecules
cmbdf = generate_mbdf(charges,coords,progress_bar=True)

#get list of unique chemical elements in the dataset (required for fchl19)
uniques = np.unique(np.concatenate(charges))

#generate fchl19 representation for all molecules
fchl = np.array([generate_fchl_acsf(q,r,uniques,pad=max([len(arr) for arr in charges]),nRs2=10,nRs3=8) for q,r in zip(charges,coords)])

def return_errs(rep, labels):
  '''
  Returns 5-fold cross-validated learning curves using KRR and the provided representation
  '''
  param_grid = {'length':[0.1*(2**i) for i in range(15)],
                  'lambda':[10**(-3*i) for i in range(1,5)]} #Logarithmic grid for hyper-parameter optimization via simple griid-search

  errsCV = []
  for seed in tqdm([1423,8465,641,56485, 190374]):
    #Seed for random-number generator
    rng = np.random.default_rng(seed)

    #Indices for test set of size 10,000 molecules to be kept aside for measuring error
    testinds = rng.choice(range(samples),size=10000,replace=False)
    xtest, ytest = rep[testinds], labels[testinds]
    qtest = charges[testinds]

    #generating learning curves for the chosen seed
    errs = []
    for size in [500,1000,2000,4000,8000]:
      trainids = rng.choice(np.delete(range(samples),testinds),size=size,replace=False)
      xtrain, ytrain = rep[trainids], labels[trainids]
      qtrain = charges[trainids]

      #Optimized hyper-parameters via grid search
      best = GridSearchCV_local(xtrain,qtrain,ytrain,param_grid,kernel='gaussian')

      #Training KRR model and obtaining predictions with the optimized hyper-parameters
      yss = KRR_local(xtrain,qtrain,ytrain,xtest,qtest,'gaussian',best)
      
      mae = np.mean(np.abs(yss - ytest))
      errs.append(mae)
      errsCV.append(errs)
      
    return np.mean(errsCV,axis=0)

errs = return_errsCV(cmbdf,labels)
print(f'cMBDF mean absolute errors: {errs}')

errs = return_errsCV(fchl,labels)
print(f'FCHL19 mean absolute errors: {errs}')
