import numpy as np
from tqdm import tqdm
from ase.io import write
from ase import Atoms
from cMBDF import generate_mbdf
from itertools import combinations
from openbabel import openbabel
from collections import Counter
data = np.load('qm24_saddles.npz',allow_pickle=True)
charges, coords, scfe = data['atoms'], data['geometries'], data['scf_energies']

inchiob, inchiob2 = [], []
for i in tqdm(range(len(charges))):
    atoms = Atoms(numbers = charges[i], positions=coords[i])
    write('temp.xyz', atoms)
    # Read XYZ file
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, "temp.xyz")
    # Convert to SMILES
    obConversion.SetOutFormat("inchi")
    smiles = obConversion.WriteString(mol)
    smiles = smiles.strip().split('\t')[0]
    inchiob.append(smiles)

    atoms = Atoms(numbers = charges[i], positions=-coords[i])
    write('temp.xyz', atoms)
    # Read XYZ file
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, "temp.xyz")
    # Convert to SMILES
    obConversion.SetOutFormat("inchi")
    smiles = obConversion.WriteString(mol)
    smiles = smiles.strip().split('\t')[0]
    inchiob2.append(smiles)

inchiob, inchiob2 = np.array(inchiob), np.array(inchiob2)
print(inchiob.shape, np.unique(inchiob).shape)
print(inchiob2.shape, np.unique(inchiob2).shape)
np.savez_compressed('qm5_saddles_inchi_chiral.npz',inchiob = inchiob, inchiob2 = inchiob2)