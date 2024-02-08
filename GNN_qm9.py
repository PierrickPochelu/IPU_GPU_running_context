import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from spektral.data.loaders import SingleLoader, BatchLoader
from spektral.datasets.citation import Citation, Cora
from spektral.layers import GCNConv, GATConv, ARMAConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess, GCNFilter

VERBOSITY=0
epochs = 1

tf.random.set_seed(seed=0)  # make weight initialization reproducible


from tf_running_context import TFRunningContext

try:
    device_type = sys.argv[1]
    model_id = int(sys.argv[2])
except IndexError:
    device_type = "GPU"
    model_id = 2
print(f"device_type={device_type} model_id={model_id}")

# url: https://www.researchgate.net/figure/List-of-hyperparameters-used-in-our-experiments-on-QM9-and-PDBBind_tbl1_345970245

models_hp={0: {"batch_size": 32, "num_units":128, "num_layers":6, "learning_rate":0.001},
           1:  {"batch_size": 32, "num_units":128, "num_layers": 2, "learning_rate":0.001},
           2: {"batch_size": 1, "num_units": 128, "num_layers": 1, "learning_rate": 0.001}
           }
hp = models_hp[model_id]


context = TFRunningContext(
    device_type,
    global_learning_rate=hp["learning_rate"],
    global_batch_size=hp["batch_size"],
    pipelining=True,
    grad_ac=1)

context.run()

# Load data
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import QM9, tudataset
from spektral.utils import label_to_one_hot

"""
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           amount=1000)  # Set to None to train on whole dataset
"""

nb_train_samples=1024
nb_test_samples=1024

dataset = QM9(n_jobs=16, amount=nb_train_samples + nb_test_samples)
# dataset = tudataset.TUDataset("github_stargazers", clean=False)

#  Usual hyperparameters
# 2layers, 128 units ; url: 

trainDataset = dataset[:nb_train_samples]
testDataset = dataset[nb_train_samples:]
print(f"len(trainDataset)={len(trainDataset)}, len(testDataset)={len(testDataset)} ")


####################
# MODEL DEFINITION #
####################
from spektral.layers import ECCConv, GlobalSumPool, GraphMasking

# Parameters
F = dataset[0].n_node_features  # Dimension of node features
S = dataset[0].n_edge_features  # Dimension of edge features
n_out = dataset[0].n_labels  # Dimension of the target


class Net(Model):
    def __init__(self,num_layers=4, num_units=128):
        super().__init__()
        self.masking = GraphMasking()
        
        self.convs=[]
        for i in range(num_layers):
           new_layer = ECCConv(num_units, activation="relu")
           self.convs.append(  new_layer  )
        
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_out)

    def call(self, inputs):
        x, a, e = inputs
        x = self.masking(x)
        for l in self.convs:
            x = l([x, a, e])
        output = self.global_pool(x)
        output = self.dense(output)

        return output


keras.backend.set_image_data_format("channels_last")

with context.device_scope():
    model = Net(hp["num_layers"], hp["num_units"])
    context.model_wrapper_before_compil(model)  # updating the graph

    # Call the Adam optimizer
    if hasattr(tf.keras.optimizers, "legacy"):  # Last TF2 version
        optimizer = tf.keras.optimizers.legacy.Adam(context.local_learning_rate)
    else:  # Older TF2 version
        optimizer = tf.keras.optimizers.Adam(context.local_learning_rate)
    optimizer = context.opt_wrapper(optimizer)

    train_steps_per_exec = len(trainDataset) // context.local_batch_size
    eval_steps_per_exec = len(testDataset) // context.local_batch_size

    # FUNCTION MODE
    # loss_fn = keras.losses.CategoricalCrossentropy(reduction="sum")
    loss_fn = keras.losses.MSE
    model.compile(
        optimizer=optimizer, loss=loss_fn, steps_per_execution=train_steps_per_exec
    )

    context.model_wrapper_after_compil(model)

    print("Start training ...")
    print("Warmup ...")
    loader_tr = BatchLoader(trainDataset, batch_size=context.local_batch_size, mask=True, epochs=epochs+1) # The +1 is a safety margin
    gen_train = loader_tr.load()
    #gen_train = tf.data.Dataset.from_tensor_slices(gen_train)

    model.fit(
        gen_train, callbacks=context.callbacks, epochs=1, verbose=VERBOSITY,
        steps_per_epoch = train_steps_per_exec
    )

    print("Launch the trainng ...")
    st = time.time()
    model.fit(
        gen_train, callbacks=context.callbacks, epochs=epochs, verbose=VERBOSITY,
        steps_per_epoch = train_steps_per_exec
    )
    training_time = time.time() - st

    model.summary()

    ##############
    # EVALUATING #
    ##############
    model.compile(
        metrics=["MAE", "MSE"],
        loss=loss_fn, # Mean squarred error
        steps_per_execution=eval_steps_per_exec,
    )
    print("Start evaluating ...")
    print("Warmup ...")
    loader_ev = BatchLoader(testDataset, batch_size=context.local_batch_size, mask=True, epochs=2+1)
    gen_ev = loader_ev.load()
    local_loss = model.evaluate(gen_ev, callbacks=context.callbacks, steps=eval_steps_per_exec, verbose=VERBOSITY)
    print("Launch the evaluation ...")
    st = time.time()
    local_loss = model.evaluate(gen_ev, callbacks=context.callbacks, steps=eval_steps_per_exec, verbose=VERBOSITY)
    prediction_time = time.time() - st

    # Display

    print(
        f"<tr>   <td>{context.num_ranks * context.num_replicas}{context.DEVICE}</td>"
        f"<td>{round(training_time, 2)}</td><td>{round(prediction_time, 2)}</td>"
        f"<td>{round(local_loss[1], 2)}</td><td> {round(local_loss[2], 2)} </td>    </tr> {model_id}"
    )


