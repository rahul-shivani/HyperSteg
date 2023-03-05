import numpy as np
import os

dir = "Distance Data/HyperSteg ESC50 NPY/"

for f in os.listdir(dir):
    print(f, np.asarray(np.load(os.path.join(dir, f))).mean())