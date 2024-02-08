import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation, Cora
from spektral.layers import GCNConv, GATConv, ARMAConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess, GCNFilter


epochs = 1000


tf.random.set_seed(seed=0)  # make weight initialization reproducible

OPTIMIZER = keras.optimizers.Adam(0.01)

from tf_running_context import TFRunningContext

try:
    device_type = sys.argv[1]
    model_id = int(sys.argv[2])
except IndexError:
    device_type = "GPU"
    model_id = 0
context = TFRunningContext(
    device_type,
    global_learning_rate=0.001,
    global_batch_size=1,
    pipelining=True,
    grad_ac=1,
)

# Load data
dataset = Citation("Pubmed", normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
graph = dataset[0]

# Display stats on the graph
print(str(graph))
print(f"n_edges={graph.n_edges}")


# We convert the binary masks to sample weights so that we can compute the
# average loss over the nodes (following original implementation by
# Kipf & Welling)
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)


####################
# MODEL DEFINITION #
####################
def get_model(model_id):
    """
    :param model_id: 0 for standard Graph Conv., 1 Attention mechanism, 2 ARMA mechanism
    :return: keras model with 2 layers
    """
    if model_id == 0:
        # SOURCE: https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py
        model = GCN(n_labels=dataset.n_labels)
    elif model_id == 1:
        # SOURCE : https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py
        channels = 8  # Number of channels in each head of the first GAT layer
        n_attn_heads = 8  # Number of attention heads in first GAT layer
        dropout = 0.5  # Dropout rate for the features and adjacency matrix
        l2_reg = 1e-4  # L2 regularization rate

        N = dataset.n_nodes  # Number of nodes in the graph
        F = dataset.n_node_features  # Original size of node features
        n_out = dataset.n_labels  # Number of classes

        # Model definition
        x_in = keras.Input(shape=(F,))
        a_in = keras.Input((N,), sparse=True)

        do_1 = keras.layers.Dropout(dropout)(x_in)
        gc_1 = GATConv(
            channels,
            attn_heads=n_attn_heads,
            concat_heads=True,
            dropout_rate=dropout,
            activation="elu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            attn_kernel_regularizer=keras.regularizers.l2(l2_reg),
            bias_regularizer=keras.regularizers.l2(l2_reg),
        )([do_1, a_in])
        do_2 = keras.layers.Dropout(dropout)(gc_1)
        gc_2 = GATConv(
            n_out,
            attn_heads=1,
            concat_heads=False,
            dropout_rate=dropout,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            attn_kernel_regularizer=keras.regularizers.l2(l2_reg),
            bias_regularizer=keras.regularizers.l2(l2_reg),
        )([do_2, a_in])
        model = keras.Model(inputs=[x_in, a_in], outputs=gc_2)
    elif model_id == 2:
        # SOURCE : https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_arma.py#L44
        channels = 16  # Number of channels in the first layer
        iterations = 1  # Number of iterations to approximate each ARMA(1)
        order = 2  # Order of the ARMA filter (number of parallel stacks)
        share_weights = True  # Share weights in each ARMA stack
        dropout_skip = 0.75  # Dropout rate for the internal skip connection of ARMA
        dropout = 0.5  # Dropout rate for the features
        l2_reg = 5e-5  # L2 regularization rate
        a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1

        N = dataset.n_nodes  # Number of nodes in the graph
        F = dataset.n_node_features  # Original size of node features
        n_out = dataset.n_labels  # Number of classes

        # Model definition
        x_in = keras.Input(shape=(F,))
        a_in = keras.Input((N,), sparse=True, dtype=a_dtype)

        gc_1 = ARMAConv(
            channels,
            iterations=iterations,
            order=order,
            share_weights=share_weights,
            dropout_rate=dropout_skip,
            activation="elu",
            gcn_activation="elu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )([x_in, a_in])
        gc_2 = keras.layers.Dropout(dropout)(gc_1)
        gc_2 = ARMAConv(
            n_out,
            iterations=1,
            order=1,
            share_weights=share_weights,
            dropout_rate=dropout_skip,
            activation="softmax",
            gcn_activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )([gc_2, a_in])
        model = keras.Model(inputs=[x_in, a_in], outputs=gc_2)
    else:
        raise ValueError(f"model_id={model_id} not expected")
    return model

keras.backend.set_image_data_format("channels_last")

with context.device_scope():
    model = get_model(model_id)

    context.model_wrapper_before_compil(model)  # updating the graph

    # FUNCTION MODE
    model.compile(
        optimizer=OPTIMIZER, loss=keras.losses.CategoricalCrossentropy(reduction="sum")
    )

    # Call the Adam optimizer
    if hasattr(tf.keras.optimizers, "legacy"):  # Last TF2 version
        optimizer = tf.keras.optimizers.legacy.Adam(context.local_learning_rate)
    else:  # Older TF2 version
        optimizer = tf.keras.optimizers.Adam(context.local_learning_rate)
    optimizer = context.opt_wrapper(optimizer)

    loader_tr = SingleLoader(dataset, sample_weights=weights_tr, epochs=1)
    xy_train=loader_tr.load()


    model.fit(
        xy_train, steps_per_epoch=loader_tr.steps_per_epoch, epochs=1, verbose=0
    )
    exit()

    loader_tr = SingleLoader(dataset, sample_weights=weights_tr, epochs=epochs)
    st = time.time()
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        epochs=epochs,
        verbose=0,
    )
    training_time = time.time() - st

    context.synch_barrier()  # synch barrier before launch

    # Evaluate model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(reduction="sum"),
        weighted_metrics=["acc"],
    )

    # warmup before evaluation
    loader_te = SingleLoader(dataset, sample_weights=weights_te)
    eval_results = model.evaluate(
        loader_te.load(), steps=loader_te.steps_per_epoch, verbose=0
    )

    st = time.time()
    loader_te = SingleLoader(dataset, sample_weights=weights_te)
    eval_results = model.evaluate(
        loader_te.load(), steps=loader_te.steps_per_epoch, verbose=0
    )
    prediction_time = time.time() - st
    local_loss, local_accuracy = eval_results
    print(
        f"<tr>   <td>{context.num_ranks * context.num_replicas}{context.DEVICE}</td>"
        f"<td>{round(training_time,2)}</td><td>{round(prediction_time,2)}</td>"
        f"<td>{round(local_loss,4)}</td><td>{round(local_accuracy,4)}</td>    </tr> {model_id}"
    )

