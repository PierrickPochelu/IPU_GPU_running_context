import numpy as np
import time

def W(x):
    np.save("data.npy", x)
def R():
    return np.load("data.npy")

x=np.random.normal(0.,1.,(250,1000,1000)).astype(np.float32)

for i in range(3):
    st=time.time()
    W(x)
    print("Writing time: ", time.time()-st)

for i in range(3):
    st=time.time()
    x=R()
    print("reading time: ", time.time()-st)

import os
os.remove("data.npy")
