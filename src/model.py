import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import config
    
def residual_block(x, filters, blocks, stride=1):
    shortcut = x
    for i in range(blocks):
        if i == 0:
            shortcut = Conv2D(filters, (1, 1), strides=(stride, stride))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(x)
        x = Activation('relu')(x)
        stride = 1  
    x = tf.keras.layers.add([x, shortcut])
    return x

# ResNet 18
def build_resnet():
    input_layer = Input(shape=(*config.IMAGE_SIZE, 3))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    x = residual_block(x, 64, 2)
    x = Dropout(0.5)(x)
    x = residual_block(x, 128, 2, stride=2)
    x = Dropout(0.5)(x)
    x = residual_block(x, 256, 2, stride=2)
    x = Dropout(0.5)(x)
    x = residual_block(x, 512, 2, stride=2)
    
    x = GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(len(config.CLASSES), activation='softmax', name='flower_prob')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
        steps_per_execution=8
    )
    return model
