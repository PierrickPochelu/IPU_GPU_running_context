import tensorflow as tf
import sys
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
from typing import Callable

# Alternative implementation: https://github.com/tokusumi/keras-flops/blob/master/keras_flops/flops_calculation.py
def get_flops(model_builder:Callable):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    num_flops=-1
    num_params=-1

    with graph.as_default():
        with session.as_default():

            model=model_builder() # put the model in the default tensorflow session


            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops_info = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
            num_flops = flops_info.total_float_ops

            opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params_info = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            num_params=params_info.total_parameters

    tf.compat.v1.reset_default_graph()

    del model

    return num_flops, num_params

def create_model(MODELID, batch_size=1, input_shape=(224,224,3)):
    import keras
    if MODELID==2:
        from tensorflow.keras.applications.resnet50 import ResNet50
        input_layer = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
        x = ResNet50(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)
    elif MODELID==3:
        from tensorflow.keras.applications.vgg16 import VGG16
        input_layer = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
        x = VGG16(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)
    elif MODELID==4:
        from tensorflow.keras.applications import MobileNet
        input_layer = tf.keras.layers.Input(shape=(224, 224, 3), batch_size=1)
        x = MobileNet(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)
    elif MODELID == 5:
        from tensorflow.keras.applications import InceptionV3
        input_layer = tf.keras.layers.Input(shape=(224, 224, 3), batch_size=1)
        x = InceptionV3(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)
    elif MODELID == 6:
        from tensorflow.keras.applications import MobileNetV3Small
        input_layer = tf.keras.layers.Input(shape=(224, 224, 3), batch_size=1)
        x = MobileNetV3Small(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)

    else:
        raise ValueError("ERROR")
    # (7748073478, 23555082)
    return model

def create_graph_model(MODELID):
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

    tf.random.set_seed(seed=0)  # make weight initialization reproducible

    # Load data
    dataset = Citation("Pubmed", normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
    graph = dataset[0]

    # Display stats on the graph
    print(str(graph))
    print(f"n_edges={graph.n_edges}")


    if MODELID == 0:
        # SOURCE: https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gcn.py
        model = GCN(n_labels=dataset.n_labels)
    elif MODELID == 1:
        # SOURCE : https://github.com/danielegrattarola/spektral/blob/master/examples/node_prediction/citation_gat.py
        channels = 8  # Number of channels in each head of the first GAT layer
        n_attn_heads = 8  # Number of attention heads in first GAT layer
        dropout = 0.5  # Dropout rate for the features and adjacency matrix
        l2_reg = 1e-4  # L2 regularization rate
        a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1
        N = dataset.n_nodes  # Number of nodes in the graph
        F = dataset.n_node_features  # Original size of node features
        n_out = dataset.n_labels  # Number of classes

        # Model definition
        x_in = keras.Input(shape=(F,), batch_size=1)
        a_in = keras.Input((N,), sparse=True, dtype=a_dtype, batch_size=1)


        gc_obj=GATConv(
            channels,
            attn_heads=n_attn_heads,
            concat_heads=True,
            dropout_rate=dropout,
            activation="elu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            attn_kernel_regularizer=keras.regularizers.l2(l2_reg),
            bias_regularizer=keras.regularizers.l2(l2_reg),
        )
        gc_1 = gc_obj([x_in, a_in])
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
    elif MODELID == 2:
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
        x_in = keras.Input(shape=(F,), batch_size=1)
        a_in = keras.Input((N,), sparse=True, dtype=a_dtype, batch_size=1)


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
        raise ValueError(f"model_id={MODELID} not expected")
    return model

"""
for i in range(1,1+1):
    def F():
        m=create_graph_model(i)
        return m

    info = get_flops(F)
    with open("out.txt","a") as f:
        info_tuple=(i, info[0], info[1], float(info[0]) / info[1])
        f.write(str(info_tuple)+"\n")
"""