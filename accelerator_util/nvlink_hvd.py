# mpirun

import horovod.keras as hvd
import tensorflow as tf
import time
import numpy as np


hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
print("List of visible physical GPUs : ", gpus)

tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True) # Dynamic allocation mode



x = np.random.uniform(0, 1, (250_000_000)).astype(np.float32)
tfx = tf.Variable(x)


hvd.allgather(np.array([0])) # barrier

st=time.time()
hvd.allgather(tfx)
print(1./(time.time()-st))


