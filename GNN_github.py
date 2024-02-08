import numpy as np
from tensorflow import keras
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation, Cora
from spektral.layers import GCNConv, GATConv, ARMAConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess, GCNFilter
import popdist.tensorflow
from popdist import tensorflow as tf
from tensorflow.python.ipu.horovod import popdist_strategy

from tensorflow.python.ipu import (
    config,
    utils,
    ipu_compiler,
    scopes,
    loops,
    ipu_infeed_queue,
    ipu_outfeed_queue,
    ipu_strategy,
    ops
)

# INIT
popdist.init()
cfg = config.IPUConfig()
popdist_on = popdist.isPopdistEnvSet()
num_global_replicas = (
    popdist.getNumTotalReplicas() if popdist_on else 1
)
num_instances = popdist.getNumInstances() if popdist_on else 1
cfg = popdist.tensorflow.set_ipu_config(
        cfg, ipus_per_replica=popdist.getNumIpusPerReplica(), configure_device=True
)
cfg.configure_ipu_system()  # initializing IPU for Tensorflow session

# APPLICATION DESIGN

epochs=1000

# Load data
dataset = Citation("Pubmed", normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
graph = dataset[0]
print(str(graph))
print(f"n_edges={graph.n_edges}")


def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)
keras.backend.set_image_data_format("channels_last")
context_device_scope=popdist_strategy.PopDistStrategy() if popdist_on else ipu_strategy.IPUStrategy()
with context_device_scope.scope():
    model = GCN(n_labels=dataset.n_labels)

    model.set_pipelining_options(
        gradient_accumulation_steps_per_replica=1,
        pipeline_schedule=ops.pipelining_ops.PipelineSchedule.Grouped,
    )
    model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=1)

    # optimizer = keras.optimizers.legacy.Adam(0.001) # <--- uncoment if needed
    optimizer = keras.optimizers.Adam(0.001)

    model.compile(
        optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(reduction="sum")
    )

    loader_tr = SingleLoader(dataset, sample_weights=weights_tr, epochs=1)
    xy_train=loader_tr.load()

    model.fit(
        xy_train, steps_per_epoch=loader_tr.steps_per_epoch, epochs=1, verbose=0
    )

"""
[1,0]<stderr>:Traceback (most recent call last):
[1,0]<stderr>:  File "GNN_github.py", line 91, in <module>
[1,0]<stderr>:    model.fit(
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/keras/engine/base_layer.py", line 3434, in wrapper
[1,0]<stderr>:    return delegate_func(obj, *args, **kwargs)
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/keras/ipu/extensions/extensions_base.py", line 1377, in _fit_delegate
[1,0]<stderr>:    for epoch, iterator in data_handler.enumerate_epochs_with_reuse(
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/keras/ipu/extensions/data_adapter.py", line 290, in enumerate_epochs_with_reuse
[1,0]<stderr>:    data_iterator = manager.get_infeed(mode, self._dataset, infeed_kwargs)
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/keras/ipu/extensions/data_feed_manager.py", line 43, in get_infeed
[1,0]<stderr>:    self._infeeds[mode] = ipu_infeed_queue.IPUIterator(dataset, **kwargs)
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/tensorflow/python/ipu/ipu_infeed_queue.py", line 442, in __init__
[1,0]<stderr>:    self._create_iterator(dataset, **kwargs)
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/tensorflow/python/ipu/ipu_infeed_queue.py", line 447, in _create_iterator
[1,0]<stderr>:    self._infeed_queue = IPUInfeedQueue(dataset, **kwargs)
[1,0]<stderr>:  File "/home/ipuuser/manu_tests/.venv_pytorch/lib/python3.8/site-packages/tensorflow/python/ipu/ipu_infeed_queue.py", line 145, in __init__
[1,0]<stderr>:    raise ValueError("""Output shape {} is not fully defined. If using \ tf.Dataset.batch, set `drop_remainder=True`.""".format(output_shape))
[1,0]<stderr>:ValueError: Output shape <unknown> is not fully defined. If using \ tf.Dataset.batch, set `drop_remainder=True`.
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[19468,1],0]
  Exit code:    1
--------------------------------------------------------------------------
"""