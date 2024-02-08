import tensorflow as tf

def fix_seed(seed=42):
    import numpy as np
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TFRunningContext:
    def __init__(self, DEVICE: str, global_learning_rate: float, global_batch_size: int, pipelining: bool = False,
                 grad_ac: int = 1):
        """
        :param DEVICE: the list of supported devices and their code names are foundable in run()
        :param global_learning_rate: Learning rate during the gradient descent. Local batch is handle by the running context
        :param global_batch_size: Batch size during the gradient descent. Local batch size is handled by the running context
        """

        # The below global variables are set by the running context and given to the application code.
        self.DEVICE = DEVICE
        self.global_learning_rate = global_learning_rate
        self.global_batch_size = global_batch_size
        self.pipelining = pipelining
        self.grad_ac = grad_ac

        # default value
        self.local_batch_size = global_batch_size
        self.local_learning_rate = global_learning_rate

        # Only useful for Horovod
        self.hvd = None

        # Only useful for popdist
        self.popdist = None

        # In the context of Horovod rank==int(hvd.rank())
        # In the context of Popdist rank is named "instances"
        self.rank = 0  # default rank. The rank is a coarser grain than popdist replicas.

        self.num_ranks = 1  # default number of ranks (named "instances" in Graphcore). WARNING: Do not confuse "replicas" and "instances". num_ranks==int(hvd.size())
        self.num_replicas = 1 # num_replicas is always equal to 1 if you use GPUs

        self.callbacks = []  # callbacks given to the model.fit() function
        self.device_scope = None  # empty function = default behaviour

        class DefaultScope:
            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        self.device_scope = (
            DefaultScope  # Scope for assigning tensor to the right accelerator (ipu, cpu, ...)
        )
        self.opt_wrapper = (
            lambda x: x
        )  # optimizer wrapper for injecting gradient synchronization code. Default is identity function.
        self.synch_barrier = (
            lambda *x: None
        )  # synch_barrier function allows fair time comparison. Initialized with a procedure doing nothing.
        self.model_wrapper_before_compil = (
            lambda x: None
        )  # Tune the model before calling keras_model.compile(...). Default do nothing for the specified hardware.

        self.model_wrapper_after_compil = lambda x: None

    def _ipu_config(self):
        from tensorflow.python import ipu
        from tensorflow.python.ipu import (
            ipu_compiler as compiler,
        )  # do not directly use tensorflow.compiler but ipu_compiler if the below application is Tensorflow (and not Keras).

        # Below code is detailed in:
        # https://github.com/graphcore/tensorflow/blob/r2.6/sdk-release-3.2/tensorflow/python/ipu/config.py
        cfg = ipu.config.IPUConfig()  # Create the IPU hardware configure
        cfg.auto_select_ipus = 1  # Attach one IPU to the current process (or MPI rank)
        # TODO: other settings include FP32, FP16, ...
        cfg.configure_ipu_system()  # Running hardware configuration IPU

    def _cpu_config(self):
        import os

        # Masking GPUs for
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device_scope = lambda *x: tf.device("/cpu:0")

    def _default_config(self):
        pass  # Tensorflow default behaviour is to use the GPU

    def _horovod_config(self):
        import horovod.keras as hvd
        self.hvd = hvd

        hvd.init()

        # Pin GPU to be used to process local rank (one GPU per process)
        if self.DEVICE[:3] == "GPU":
            gpus = tf.config.experimental.list_physical_devices("GPU")
            print("List of visible physical GPUs : ", gpus, " local_rank=", str(hvd.local_rank()))
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
                tf.config.experimental.set_memory_growth(
                    gpus[hvd.local_rank()], True
                )  # Dynamic allocation mode

        # Broadcast initial weights from rank 0 to all other processes.
        self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Local identifier among processes
        self.rank, self.num_ranks = int(hvd.rank()), int(hvd.size())

        # Global_batch_size computing workload is shared among workers
        self.local_batch_size = int(self.global_batch_size / self.num_ranks)

        # The aggregation operator between process is the average.
        # Thus, we multiply the learning rate for incresing the gradient vector-length
        self.local_learning_rate = self.num_ranks * self.global_learning_rate

        # Uncomment below block for regular checkpointing
        # if hvd.rank() == 0:
        # callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

        # Gradient synchronization operator
        self.opt_wrapper = hvd.DistributedOptimizer

        self.synch_barrier = lambda *arg: hvd.allgather(tf.Variable([0]))

    def _popdist_config(self):
        print("POPDIST CONFIG")
        # Cancel previous Tensorflow import
        import os
        import sys
        
        # The below code seems to file when we scale
        """
        for m in ["tensorflow", "keras"]:
            if m in sys.modules:
                os.remove(sys.modules[m].__cached__)  # remove cached bytecode
                del sys.modules[m]  # remove tensorflow
       """ 

        # Import the PopDist Tensorflow
        import popdist
        self.popdist = popdist
        import popdist.tensorflow
        from popdist import tensorflow as tf
        from tensorflow.python.ipu.horovod import popdist_strategy
        # from tensorflow.python.ipu import horovod as hvd # Horovod is now obsolete when PopDist. However, horovod contains a richer API than Popdist: allreduce and rank are implemented with horovod

        # import precision # TODO: FP32, FP16, Hybrid
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

        popdist.init()

        # Below code is copy/paste from ~/pierrick_tests/tensorflow2/tests_serial/test_distributed_training.py
        cfg = config.IPUConfig()
        popdist_on = popdist.isPopdistEnvSet()
        num_global_replicas = (
            popdist.getNumTotalReplicas() if popdist_on else 1
        )
        num_instances = popdist.getNumInstances() if popdist_on else 1

        if popdist_on:
            cfg = popdist.tensorflow.set_ipu_config(
                cfg, ipus_per_replica=popdist.getNumIpusPerReplica(), configure_device=True
            )
            popdist.init()
            self.num_replicas=popdist.getNumTotalReplicas() // popdist.getNumInstances()
            # TODO: never tesk by combining data parallel + model parallel
        else:
            cfg.auto_select_ipus = num_global_replicas
            self.num_replicas = num_global_replicas
        cfg.configure_ipu_system()  # initializing IPU for Tensorflow session

        # https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/configuration.html
        self.rank, self.num_ranks = int(popdist.getInstanceIndex()), int(popdist.getNumInstances())
        self.local_learning_rate = self.num_ranks * self.global_learning_rate
        self.local_batch_size = int(self.global_batch_size // num_global_replicas)

        # Strategy object: PopDistStrategy for multinode multi-IPU, IPUStrategy for mono-node multi-IPU
        strategy = (
            popdist_strategy.PopDistStrategy() if popdist_on else ipu_strategy.IPUStrategy()
        )
        self.device_scope = strategy.scope  # Tensorflow scope allows to assign tensors on IPUs

        self.synch_barrier = lambda *arg: popdist.synchronize()

        # Improving parallelsim with pipeline-parallelism and gradient accumulation
        # https://docs.graphcore.ai/projects/tensorflow-user-guide/en/3.2.0/tensorflow/perf_training.html#pipelined-training
        def model_wrapper_before_compil(model):
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=self.grad_ac,  # TODO introducing grad. accum feature
                pipeline_schedule=ops.pipelining_ops.PipelineSchedule.Grouped,
            )  # Enable pipeline-parallelism if the model is splitted
        

        def model_wrapper_after_compil(model):
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=self.grad_ac)
        self.model_wrapper_before_compil=model_wrapper_before_compil
        self.model_wrapper_after_compil=model_wrapper_after_compil

        print(f"popdist_on={popdist_on}")
        print(f"num_global_replicas={num_global_replicas}")
        print(f"num_instances={num_instances}")
        print(f"device_scope={self.device_scope}")

    def _multiple_gpu_config(self):
        self._default_config()
        self._horovod_config()

    def _multiple_ipu_config(self):
        self._popdist_config()

    def _multiple_cpu_config(self):
        self._cpu_config()
        self._horovod_config()

    def run(self):
        """
        Function to launch for preparing the running context to the targeted accelerator.
        :return: nothing
        """
        fix_seed()

        # Call the right config option according to the DEVICE variable
        configs = {
            "CPU": self._cpu_config,
            "IPU": self._ipu_config,
            "GPU": self._default_config,
            "TPU": self._default_config,
            "GPUS": self._multiple_gpu_config,
            "IPUS": self._multiple_ipu_config,
            "CPUS": self._multiple_cpu_config,
        }
        configs[self.DEVICE]()  # update global variables

