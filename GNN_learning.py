'''
This script trains and returns the prediction error on atomization energies for Graph Neural Network models based on SchNet or PaiNN using the schnetpack library
'''

import os
import numpy as np
from ase import Atoms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping

seed, trainsize, rep = input("Enter seed: "), input("Enter training size: "), input("Choose model type (schnet or painn): ")
seed, trainsize = int(seed), int(trainsize)

# stop if val_loss doesn’t improve for 20 epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    verbose=False
)

class EpochProgressBar(Callback):
    def on_train_start(self, trainer, pl_module):
        self.pbar = tqdm(total=trainer.max_epochs, desc="Training Progress (epochs)", position=0)

    def on_train_epoch_end(self, trainer, pl_module):
        self.pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.pbar.close()

# Device settings
torch.set_float32_matmul_precision('medium')
#print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.get_device_name(0))

# Load VQM24 .npz data
data = np.load('DFT_uniques.npz', allow_pickle=True)
charges, coords, labels = data['atoms'], data['coordinates'], data['Eatomization']
labels *= 27.2114  # Convert Hartree to eV

samples = len(charges)

#setting random-number generator for creating training and test sets
rng = np.random.default_rng(seed)

#indices for test set which will be kep aside for measuring prediction error (size = 10,000 molecules)
testinds = rng.choice(range(samples),size=10000,replace=False)

#indices for training set which excludes the previously selected test set. 
#Validation set of size 1000 molecules is chosen for hyper-parameter optimization
trainids = rng.choice(np.delete(range(samples),testinds),size=trainsize+1000,replace=False)

# Convert into ASE Atoms objects
class VQM24Dataset(Dataset):
    def __init__(self, charges, coords, labels):
        self.atoms = [Atoms(numbers=z, positions=r) for z, r in zip(charges, coords)]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, idx):
        return {
            "atoms": self.atoms[idx],
            "label": self.labels[idx]
        }

# Create dataset and dataloaders
dataset = VQM24Dataset(charges[trainids], coords[trainids], labels[trainids])
total_len = len(dataset)
print("Total dataset size:", total_len)
ntrain = trainsize
nval = 1000
train_data, val_data = torch.utils.data.random_split(dataset, [ntrain, nval])

# Converter for ASE Atoms to model input
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.),
    dtype=torch.float32,
    device='cuda'
)

class VQM24DataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, converter, batch_size=100, num_workers=0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.converter = converter
        self.batch_size = batch_size
        self.num_workers = num_workers

    def collate_fn(self, batch):
        atoms = [item["atoms"] for item in batch]
        y = torch.stack([item["label"] for item in batch])
        inputs = self.converter(atoms)
        inputs["y"] = y
        return inputs

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

datamodule = VQM24DataModule(train_data, val_data, converter)

# PaiNN model setup
cutoff = 5.
n_atom_basis = 128
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

if rep=='painn':
    model = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
elif rep=='schnet'
    model = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=3,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )

output_key = "y"  # name of the key storing labels in the batch
pred_layer = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=output_key)

nnpot = spk.model.NeuralNetworkPotential(
    representation=model,
    input_modules=[spk.atomistic.PairwiseDistances()],
    output_modules=[pred_layer]
)

output_H = spk.task.ModelOutput(
    name=output_key,
    loss_fn=nn.MSELoss(),
    loss_weight=1.,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()}
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_H],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# Training config
save_dir = f"./{rep}_vqm24"
os.makedirs(save_dir, exist_ok=True)

logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(save_dir, "best_model"),
        save_top_k=1,
        monitor="val_loss",
    ),EpochProgressBar(),
    #early_stop,
]

trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=save_dir,
    enable_progress_bar=False
)

trainer.fit(task, datamodule=datamodule,)

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32, device='cuda')
best_model = torch.load(os.path.join(save_dir, "best_model"))

mols = []
for num,pos in tqdm(list(zip(charges[testinds],coords[testinds]))):
    mols.append(Atoms(numbers = num, positions=pos))

preds = []
ev2kcal = 23.0621
for mol in tqdm(mols,desc='Predictions',position=0):  # test set
    inputs = converter(mol)
    with torch.no_grad():
        pred = best_model(inputs)
    preds.append(pred[output_key].item())

preds = np.array(preds)
mae = np.mean(np.abs(preds - labels[testinds]))

file = open(f"{rep}_vqm24.txt", "a")
file.write(f"Seed : {seed}, Train size : {trainsize}, Test size : {len(testinds)}\n")
file.write(f"Test MAE [kcal/mol]: {mae*ev2kcal}\n")
file.write(f"Test MAE [eV]: {mae}\n")
file.write(" \n")
file.close()

print(f"Seed : {seed}, Train size : {trainsize}, Test size : {len(testinds)}")

print("Test MAE [kcal/mol]:", mae*ev2kcal)

mae = np.mean(np.abs(preds - labels[testinds]))
print("Test MAE [eV]:", mae)
print(" ")
