from tensorflow import keras
import tensorflow as tf
from collections import OrderedDict


def AlexNet(classes=10, weights=None, include_top=False):
    image_width, image_height, channels = 224, 224, 3
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        # layer 6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.5),

        # layer 7
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.5),

        # layer 8
        tf.keras.layers.Dense(units=classes,
                              activation=tf.keras.activations.softmax)
    ])

    return model

def LeNet5(classes=10, weights=None, include_top=False):
    image_width, image_height, channels = 32, 32, 3
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=6,
                               kernel_size=(3, 3),
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.AveragePooling2D(),

        # layer 2
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.AveragePooling2D(),

        # layer 3
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=120,
                              activation=tf.keras.activations.relu),

        # layer 4
        tf.keras.layers.Dense(units=80,
                              activation=tf.keras.activations.relu),

        # layer 8
        tf.keras.layers.Dense(units=classes,
                              activation=tf.keras.activations.softmax)
    ])

    return model

def get_models():
    import tensorflow
    MODELS = {
    "xception": (tensorflow.keras.applications.Xception,224),
    "vgg16": (tensorflow.keras.applications.VGG16,224),
    "vgg19": (tensorflow.keras.applications.VGG19,224),
    "r50": (tensorflow.keras.applications.ResNet50,224),
    "r50v2": (tensorflow.keras.applications.ResNet50V2,224),
    "r101": (tensorflow.keras.applications.ResNet101,224),
    "r101v2": (tensorflow.keras.applications.ResNet101V2,224),
    "r152": (tensorflow.keras.applications.ResNet152,224),
    "r152v2": (tensorflow.keras.applications.ResNet152V2,224),

    "inv3": (tensorflow.keras.applications.InceptionV3,224),
    "inv4": (tensorflow.keras.applications.InceptionResNetV2,224),

    "mob": (tensorflow.keras.applications.MobileNet,224),
    "mob2": (tensorflow.keras.applications.MobileNetV2,224),
    "mob3s": (tensorflow.keras.applications.MobileNetV3Small,224),
    "mob3l": (tensorflow.keras.applications.MobileNetV3Large,224),
    }

    MODELS={"dens169": (tf.keras.applications.DenseNet169,32),
            "dens201": (tf.keras.applications.DenseNet201,32),
            "nas":(tf.keras.applications.NASNetMobile,224)}



    MODELS={"nas": (tf.keras.applications.NASNetMobile, 224)}
    MODELS={"eff1": (tf.keras.applications.EfficientNetB1,240),
            "eff2": (tf.keras.applications.EfficientNetB2,260),
            "eff3": (tf.keras.applications.EfficientNetB3,300) }
    MODELS = {"lenet": (LeNet5, 32)}
    MODELS = {"dens121": (tf.keras.applications.DenseNet121,224)}
    MODELS=OrderedDict(sorted(MODELS.items()))
    return MODELS # keras_model_class_ptr, img_size

def CNN(model_name, fixed_batch_size=None):


    model_class_ptr,img_size=get_models()[model_name]
    
    if fixed_batch_size:
        input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3),batch_size=fixed_batch_size)
    else:
        input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))



    if model_name=="nas" or model_name=="alex" or model_name=="lenet":
            x = model_class_ptr(weights=None, include_top=True, classes=10)(input_layer)
            return keras.Model(inputs=input_layer, outputs=x)

    x = model_class_ptr(weights=None, include_top=False, classes=10)(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=x)

    return model

