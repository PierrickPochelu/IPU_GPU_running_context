from transformers import TFAutoModelForSequenceClassification, TFBertForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

from graphcore_util_code.data_utils.glue.load_glue_data import *


VERBOSITY = 1
NUM_EPOCHS = 2

try:
      DEVICE, GBS, GRADAC, MODELID = sys.argv[1],int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
except:
      DEVICE, GBS, GRADAC, MODELID = "GPU", 1, 1, 7


from tf_running_context import TFRunningContext


context = TFRunningContext(DEVICE, global_learning_rate=0.001, global_batch_size=GBS, pipelining=True, grad_ac=GRADAC)


context.run()

print(context.__dict__)


######################
# DATASET PROCESSING #
######################
keras.backend.set_image_data_format("channels_last")

"""
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])  # doctest: +SKIP

dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
"""

tf_dataset_train = get_generated_dataset()
tf_dataset_test = get_generated_dataset()
num_train_samples=len(tf_dataset_train)
num_test_samples=len(tf_dataset_test)
(
    train_dataset,
    eval_dataset,
    test_dataset,
    num_train_samples,
    num_eval_samples,
    num_test_samples,
    raw_datasets,
)=get_glue_data("cola",micro_batch_size=context.local_batch_size, cache_dir="./tmp/",generated_dataset=True)


######################################
##### KERAS MODEL DEFINITION #########
######################################



with context.device_scope():  # Mapping between hardware and tensors
    keras.backend.set_image_data_format("channels_last")
    #model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = TFBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    """
    
    tf_dataset_train = model.prepare_tf_dataset(
        dataset["train"], batch_size=context.local_batch_size, shuffle=True, tokenizer=tokenizer
    )
    tf_dataset_test = model.prepare_tf_dataset(
        dataset["test"], batch_size=context.local_batch_size, shuffle=True, tokenizer=tokenizer
    )

    tf_dataset_train = tf.data.Dataset.from_tensor_slices(
        tf_dataset_train
    )
    """

    import tensorflow as tf

    if hasattr(tf.keras.optimizers, "legacy"):  # Last TF2 version
        optimizer = tf.keras.optimizers.legacy.Adam(context.local_learning_rate)
    else:  # Older TF2 version
        optimizer = tf.keras.optimizers.Adam(context.local_learning_rate)
    optimizer = context.opt_wrapper(optimizer)

    train_steps_per_exec = int(num_train_samples // context.local_batch_size)
    eval_steps_per_exec = int(num_test_samples  // context.local_batch_size)




    ############
    # TRAINING #
    ############
    context.model_wrapper_before_compil(model)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  steps_per_execution=train_steps_per_exec)
    #model.summary()
    context.model_wrapper_after_compil(model)

    #warmup
    model.fit(tf_dataset_train,
              epochs=1,
              batch_size=context.local_batch_size,
              callbacks=context.callbacks,
              verbose=VERBOSITY)
    context.synch_barrier()
    st=time.time()
    model.fit(tf_dataset_train,
              epochs=NUM_EPOCHS,
              batch_size=context.local_batch_size,
              callbacks=context.callbacks,
              verbose=VERBOSITY)
    training_time = time.time()


    ##############
    # EVALUATING #
    ##############
    model.compile(
        metrics=["accuracy"],
        loss="sparse_categorical_crossentropy",
        steps_per_execution=eval_steps_per_exec,
    )

    import time


    local_accuracy = model.evaluate(tf_dataset_test, batch_size=context.local_batch_size, verbose=0)
    context.synch_barrier()
    st=time.time()
    local_accuracy = model.evaluate(tf_dataset_test, batch_size=context.local_batch_size, verbose=0)
    prediction_time = time.time()-st

    # Copy paste in a HTML table
    real_batch_size=context.global_batch_size*context.grad_ac
    print(f"<tr>   <td>{context.num_ranks*context.num_replicas}{context.DEVICE} BS={real_batch_size} {context.global_batch_size}GA{context.grad_ac}</td><td>{training_time}</td><td>{prediction_time}</td><td>{local_accuracy[0]}</td><td>{local_accuracy[1]}</td>    </tr> {MODELID}")
