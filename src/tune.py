import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from dataloader import get_training_dataset, get_validation_dataset
from model import build_resnet_tune
import config

# Count steps
from dataloader import count_data_items
NUM_TRAINING_IMAGES = count_data_items(config.TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(config.VALIDATION_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // config.BATCH_SIZE
VALIDATION_STEPS = -(-NUM_VALIDATION_IMAGES // config.BATCH_SIZE)

def tune_model():
    tuner = kt.Hyperband(
        build_resnet_tune,  # Model-building function with hyperparams
        objective='val_sparse_categorical_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='flower_tuning'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    tuner.search(
        get_training_dataset(),
        epochs=10,
        validation_data=get_validation_dataset(),
        validation_steps=VALIDATION_STEPS,
        callbacks=[early_stopping]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    tuner.results_summary()

    return best_model
