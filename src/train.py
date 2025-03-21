import math, re, os, sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from model import build_resnet, build_resnet50
from dataloader import get_training_dataset, get_validation_dataset, count_data_items
import config

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Tensorflow version " + tf.__version__)


# os.environ['CUDA_VISIBLE_DEVICES'] = "5" 

def lrfn(epoch):
    if epoch < config.LR_RAMPUP_EPOCHS:
        lr = (config.LR_MAX - config.LR_START) / config.LR_RAMPUP_EPOCHS * epoch + config.LR_START
    elif epoch < config.LR_RAMPUP_EPOCHS + config.LR_SUSTAIN_EPOCHS:
        lr = config.LR_MAX
    else:
        lr = (config.LR_MAX - config.LR_MIN) * config.LR_EXP_DECAY**(epoch - config.LR_RAMPUP_EPOCHS - config.LR_SUSTAIN_EPOCHS) + config.LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

# Learning rate schedule for training
# rng = [i for i in range(EPOCHS)]
# y = [lrfn(x) for x in rng]
# plt.plot(rng, y)
# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


NUM_TRAINING_IMAGES = count_data_items(config.TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(config.VALIDATION_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // config.BATCH_SIZE
VALIDATION_STEPS = -(-NUM_VALIDATION_IMAGES // config.BATCH_SIZE)

NUM_VAL_IMAGES = count_data_items(config.VALIDATION_FILENAMES)


# Create the model
model = build_resnet()
# model = build_resnet50()

# tensorboard --logdir=
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)

history = model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=config.EPOCHS,
                    validation_data=get_validation_dataset(), validation_steps=VALIDATION_STEPS,
                    callbacks=[lr_callback, tensorboard_callback])


val = get_validation_dataset(ordered=True) 

print('Computing predictions...')
val_data = val.map(lambda image, idnum: image)
prob = model.predict(val_data)
preds = np.argmax(prob, axis=-1)

print('Generating submission.csv file...')
val_data = val.map(lambda image, idnum: idnum).unbatch()
val_id = next(iter(val_data.batch(NUM_VAL_IMAGES))).numpy().astype('U')

# Write the submission file
np.savetxt(
    'submission.csv',
    np.rec.fromarrays([val_id, preds]),
    fmt=['%s', '%d'],
    delimiter=',',
    header='id,label',
    comments='',
)

# main.py wrap up

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Tensorflow version " + tf.__version__)

    model = build_resnet()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_DIR, histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    history = model.fit(
        get_training_dataset(), 
        steps_per_epoch=STEPS_PER_EPOCH, 
        epochs=config.EPOCHS,
        validation_data=get_validation_dataset(), 
        validation_steps=VALIDATION_STEPS,
        callbacks=[lr_callback, tensorboard_callback]
    )

    val = get_validation_dataset(ordered=True) 
    print('Computing predictions...')
    val_data = val.map(lambda image, idnum: image)
    prob = model.predict(val_data)
    preds = np.argmax(prob, axis=-1)

    print('Generating submission.csv file...')
    val_data = val.map(lambda image, idnum: idnum).unbatch()
    val_id = next(iter(val_data.batch(NUM_VAL_IMAGES))).numpy().astype('U')

    np.savetxt(
        config.SUBMISSION_FILE,
        np.rec.fromarrays([val_id, preds]),
        fmt=['%s', '%d'],
        delimiter=',',
        header='id,label',
        comments='',
    )

if __name__ == "__main__":
    main()
