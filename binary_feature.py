# %%
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tqdm
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#%matplotlib inline

# %%
with open("res.txt","r") as f:
    annos = f.readlines()
annos = [np.array([float(x) for x in anno.split()]) for anno in annos]
annos = np.stack(annos)

# %%
pattern = "generate_with_latent/seed{}.npy"
xs = []
for idx in range(1,len(annos)+1):
    x = np.load(pattern.format(str(idx).zfill(4)))
    xs.append(x[:,4])
xs = np.concatenate(xs).reshape(len(annos),-1)


# %%
clf = LDA()
clf.fit(xs[:800], annos[:800,1].astype(int))
pattern_vector = clf.coef_[:,:512]
pattern_vector /= np.linalg.norm(pattern_vector,ord=2)
torch.tensor(pattern_vector)
torch.save(torch.tensor(pattern_vector),"pattern_strength.pt")
