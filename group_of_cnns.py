# Code written by Pierrick Pochelu

###########
# IMPORTS #
###########
import sys
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

NUM_EPOCHS = 10

# Build asynch. generator between reading batches and training computing.
# Different implentation style are possible
loading_generator_style = "TF"


from tf_running_context import TFRunningContext


global_batch_size = 1
grad_ac = 1
img_size = 224
MAX_IMG = 1000

context = TFRunningContext(sys.argv[1], global_learning_rate=0.001, global_batch_size=global_batch_size, pipelining=True, grad_ac=grad_ac)



context.run()

print(context.__dict__)

##############################
#### DATASET PROCESSING ######
###############################
keras.backend.set_image_data_format("channels_last")

print("BUILDING IMAGES ...")
# Reading raw images
(trainImages, trainLabels), (
    testImages,
    testLabels,
) = keras.datasets.cifar10.load_data()

# FIRST SPLIT
if MAX_IMG < len(trainImages):
    trainImages = trainImages[:MAX_IMG]
    trainLabels = trainLabels[:MAX_IMG]
if MAX_IMG < len(testImages):
    testImages = testImages[:MAX_IMG]
    testLabels = testLabels[:MAX_IMG]


# Preprocessing data from [0;255] to [0;1.0]
trainImages = trainImages.astype(np.float32) / 255.0
testImages = testImages.astype(np.float32) / 255.0
trainLabels = trainLabels.astype(np.int32)
testLabels = testLabels.astype(np.int32)

# Selection of images. The nunber of images should be multiple of the batch size, otherwise remainding images are ignored.
global_batch_with_grad_ac=context.global_batch_size * context.grad_ac
training_images = int(
    (len(trainImages) // global_batch_with_grad_ac) * global_batch_with_grad_ac
)  # Force all steps to be the same size
testing_images = int((len(testImages) // context.global_batch_size) * context.global_batch_size)
trainImages = trainImages[:training_images]
trainLabels = trainLabels[:training_images]
testImages = testImages[:testing_images]
testLabels = testLabels[:testing_images]

# Each rank (or "instance") is responsible for a subset of images
if context.num_ranks > 1:
    num_train_per_replica = len(trainImages) // context.num_ranks
    trainImages = trainImages[
                  context.rank * num_train_per_replica: (context.rank + 1) * num_train_per_replica
                  ]
    trainLabels = trainLabels[
                  context.rank * num_train_per_replica: (context.rank + 1) * num_train_per_replica
                  ]
    num_eval_per_replica = len(testImages) // context.num_ranks
    testImages = testImages[
                 context.rank * num_eval_per_replica: (context.rank + 1) * num_eval_per_replica
                 ]
    testLabels = testLabels[
                 context.rank * num_eval_per_replica: (context.rank + 1) * num_eval_per_replica
                 ]

"""
# Resizing
from scipy.ndimage import zoom

trainImages = zoom(trainImages, (1, 7.0, 7.0, 1), order=1)
testImages = zoom(testImages, (1, 7.0, 7.0, 1), order=1)

# Storing it for next time
np.save(f"trainImages{sn}.npy", trainImages)
np.save(f"trainLabels{sn}.npy", trainLabels)
np.save(f"testImages{sn}.npy", testImages)
np.save(f"testLabels{sn}.npy", testLabels)
"""
##########################
# BUILDING EFFICIENT I/O #
##########################

xy_train, xy_test = None, None


def tf_generator(trainImages, trainLabels, testImages, testLabels):
    # WARNING: display warning message "INVALID_ARGUMENT" . Those messages are Keras error messages .
    train_gen = tf.data.Dataset.from_tensor_slices(
        (tf.cast(trainImages, tf.float32), tf.cast(trainLabels, tf.float32))
    )

    eval_gen = tf.data.Dataset.from_tensor_slices(
        (tf.cast(testImages, tf.float32), tf.cast(testLabels, tf.float32))
    )


    train_gen = (
        train_gen.shuffle(len(trainImages))
        .batch(context.local_batch_size, drop_remainder=True)
        .map(lambda image, label: (tf.image.resize(image, (img_size, img_size)), label))
        .prefetch(tf.data.AUTOTUNE)
    )
    eval_gen = (
        eval_gen.shuffle(len(testImages))
        .batch(context.local_batch_size, drop_remainder=True)
        .map(lambda image, label: (tf.image.resize(image, (img_size, img_size)), label))
        .prefetch(tf.data.AUTOTUNE)
    )

    xy_train = (train_gen,)
    xy_test = (eval_gen,)
    return xy_train, xy_test


def keras_generator(trainImages, trainLabels, testImages, testLabels):
    # WARNING: keras generator is not compatible with 'model.fit'. We should call 'model.fit_generator instead'.
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    gen = ImageDataGenerator()
    train_gen = gen.flow(trainImages, trainLabels, batch_size=context.local_batch_size)

    gen2 = ImageDataGenerator()
    eval_gen = gen2.flow(testImages, testLabels, batch_size=context.local_batch_size)

    xy_train = (train_gen,)
    xy_test = (eval_gen,)
    return (xy_train,), (xy_test,)


def no_generator(trainImages, trainLabels, testImages, testLabels):
    # Entiere dataset is given to model.fit(). I/O may bottleneck performance.
    return (trainImages, trainLabels), (testImages, testLabels)

generators = {"TF": tf_generator, "KERAS": keras_generator, "NO": no_generator}
xy_train, xy_test = generators[loading_generator_style](
    trainImages, trainLabels, testImages, testLabels
)

######################################
##### KERAS MODEL DEFINITION #########
######################################

keras.backend.set_image_data_format("channels_last")

def TRAINING(model_class):

    with context.device_scope():  # Mapping between hardware and tensors

        input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
        x = model_class(weights=None, include_top=False, classes=10)(input_layer)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=x)

        context.model_wrapper_before_compil(model)  # updating the graph




        model.summary()

        # Call the Adam optimizer
        if hasattr(tf.keras.optimizers, "legacy"):  # Last TF2 version
            optimizer = tf.keras.optimizers.legacy.Adam(context.local_learning_rate)
        else:  # Older TF2 version
            optimizer = tf.keras.optimizers.Adam(context.local_learning_rate)
        optimizer = context.opt_wrapper(optimizer)

        # Compute the number of steps
        # Number of steps in the replica = #images in the rank // local_batch_size
        train_steps_per_exec = len(trainImages) // context.local_batch_size
        eval_steps_per_exec = len(testImages) // context.local_batch_size


        ############
        # TRAINING #
        ############
        # Keras computing graph construction. Plugs together : the model, the loss, the optimizer
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            steps_per_execution=train_steps_per_exec,
        )
        context.model_wrapper_after_compil(model)

        # warmup training
        model.fit(
            *xy_train,
            epochs=1,
            batch_size=context.local_batch_size,
            callbacks=context.callbacks,
            verbose=0)

        context.synch_barrier()  # synch barrier before launch

        st = time.time()
        model.fit(
            *xy_train,
            epochs=NUM_EPOCHS,
            batch_size=context.local_batch_size,
            callbacks=context.callbacks,
            verbose=2,
        )
        training_time = time.time() - st

        ###############
        # EVALUATING #
        ###############
        model.compile(
            metrics=["accuracy"],
            loss="sparse_categorical_crossentropy",
            steps_per_execution=eval_steps_per_exec,
        )
        (local_loss, local_accuracy) = model.evaluate(*xy_test, batch_size=context.local_batch_size, verbose=0)
        context.synch_barrier()

        st = time.time()
        (local_loss, local_accuracy) = model.evaluate(*xy_test, batch_size=context.local_batch_size, verbose=0)
        prediction_time = time.time() - st

        # If multiprocessing rank > 1, otherwise it is only identity
        """
        if hvd is not None:
            print(f"all reduce betwen {hvd.size()}")
            global_loss = hvd.allreduce(np.array(local_loss)).numpy()
            global_accuracy = hvd.allreduce(np.array(local_accuracy)).numpy()
        else:
            global_loss, global_accuracy = local_loss, local_accuracy
        """

        print("**************")

        local_loss=round(local_loss, 4)
        local_accuracy=round(local_accuracy, 4)
        training_time=round(training_time, 0)
        prediction_time=round(prediction_time, 2)


        print(f"MODELID: {MODELID}")
        print("local batch_size", context.local_batch_size)
        print(
            "local train images:", len(trainImages), "local test images:", len(testImages)
        )
        print("train_steps_per_execution", train_steps_per_exec)
        print(
            "img trained per second",
            NUM_EPOCHS * context.num_ranks * len(trainImages) / training_time,
        )
        print(
            "img prediction per second",
            context.num_ranks * len(testImages) / prediction_time,
        )
        print("gradient accumulation", context.grad_ac)
        print(
            "local loss:", local_loss
        )  # Local loss. All losses should be average for the global loss.
        print(
            "local accuracy:", local_accuracy,
        )  # Local accuracy. All accuracies should be average for the global average
        print("training_time:", training_time)
        print("prediction_time", prediction_time)

        # Copy paste in a HTML table
        real_batch_size=context.global_batch_size*context.grad_ac
        print(f"<tr>   <td>{context.num_ranks*context.num_replicas}{context.DEVICE} BS={real_batch_size} {context.global_batch_size}GA{context.grad_ac}</td><td>{training_time}</td><td>{prediction_time}</td><td>{local_loss}</td><td>{local_accuracy}</td>    </tr> {MODELID}")


from tensorflow.keras.applications.resnet50 import ResNet50

model_classes = [ResNet50]
for model_class in model_classes:
    TRAINING(model_class)