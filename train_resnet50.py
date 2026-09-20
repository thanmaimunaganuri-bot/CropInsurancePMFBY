##########################################################
# IMPORTS
##########################################################

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

##########################################################
# Ignore TensorFlow Warnings
##########################################################

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

##########################################################
# Dataset Paths
##########################################################
TRAIN_DIR = "/content/dataset/train"
TEST_DIR = "/content/dataset/test"

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

##########################################################
# Image Parameters
##########################################################

IMG_SIZE = 224
BATCH_SIZE = 32

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15

##########################################################
# DATA AUGMENTATION
##########################################################

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.20,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest"

)

##########################################################
# TEST DATA
##########################################################

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

##########################################################
# TRAIN GENERATOR
##########################################################

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True

)

##########################################################
# TEST GENERATOR
##########################################################

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

##########################################################
# CLASS INFORMATION
##########################################################

NUM_CLASSES = len(train_generator.class_indices)

print("\nClasses Found :", NUM_CLASSES)

print(train_generator.class_indices)

##########################################################
# SAVE CLASS NAMES
##########################################################

with open(
    "models/classes.json",
    "w"
) as f:

    json.dump(
        train_generator.class_indices,
        f,
        indent=4
    )

print("\nclasses.json Saved Successfully")

##########################################################
# LOAD RESNET50
##########################################################

print("\nLoading ResNet50...\n")

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_tensor=Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

)

##########################################################
# FREEZE RESNET50
##########################################################

base_model.trainable = False

##########################################################
# CUSTOM CLASSIFIER
##########################################################

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(

    512,

    activation="relu"

)(x)

x = Dropout(0.50)(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(0.30)(x)

outputs = Dense(

    NUM_CLASSES,

    activation="softmax"

)(x)

##########################################################
# FINAL MODEL
##########################################################

model = Model(

    inputs=base_model.input,

    outputs=outputs

)

##########################################################
# COMPILE
##########################################################

model.compile(

    optimizer=Adam(

        learning_rate=1e-4

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

##########################################################
# MODEL SUMMARY
##########################################################

model.summary()
##########################################################
# CALLBACKS
##########################################################

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

checkpoint = ModelCheckpoint(

    filepath="models/cnn.h5",

    monitor="val_accuracy",

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=1e-7,

    verbose=1

)

##########################################################
# TRAIN ONLY CLASSIFICATION HEAD
##########################################################

print("\n==========================================")
print("PHASE 1 : TRAINING CLASSIFICATION HEAD")
print("==========================================\n")

history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=PHASE1_EPOCHS,

    callbacks=[

        early_stopping,

        checkpoint,

        reduce_lr

    ],

    verbose=1

)

##########################################################
# SAVE PHASE 1 MODEL
##########################################################

model.save(

    "models/resnet50_phase1.h5"

)

print("\nPhase 1 Model Saved Successfully.")

##########################################################
# PHASE 1 EVALUATION
##########################################################

loss, accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print("\nPhase 1 Validation Accuracy : {:.2f}%".format(accuracy*100))

print("Phase 1 Validation Loss     : {:.4f}".format(loss))
##########################################################
# PHASE 2 : FINE-TUNING
##########################################################

print("\n==========================================")
print("PHASE 2 : FINE-TUNING RESNET50")
print("==========================================\n")

##########################################################
# Unfreeze Last 50 Layers
##########################################################

base_model.trainable = True

for layer in base_model.layers[:-50]:

    layer.trainable = False

for layer in base_model.layers[-50:]:

    layer.trainable = True

##########################################################
# Show Trainable Layers
##########################################################

trainable_layers = 0

for layer in model.layers:

    if layer.trainable:

        trainable_layers += 1

print("\nTrainable Layers :", trainable_layers)

##########################################################
# Recompile Model
##########################################################

model.compile(

    optimizer=Adam(

        learning_rate=1e-5

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

##########################################################
# Fine-Tune Model
##########################################################

history_finetune = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=PHASE2_EPOCHS,

    callbacks=[

        early_stopping,

        checkpoint,

        reduce_lr

    ],

    verbose=1

)

##########################################################
# Save Fine-Tuned Model
##########################################################

model.save(

    "models/resnet50_finetuned.h5"

)

print("\nFine-Tuned Model Saved Successfully.")

##########################################################
# Load Best Model
##########################################################

print("\nLoading Best Model...\n")

model.load_weights(

    "models/cnn.h5"

)

print("Best Model Loaded Successfully.")

##########################################################
# Evaluate Fine-Tuned Model
##########################################################

loss, accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print("\nFine-Tuned Validation Accuracy : {:.2f}%".format(accuracy * 100))

print("Fine-Tuned Validation Loss     : {:.4f}".format(loss))
##########################################################
# EVALUATE BEST MODEL
##########################################################

print("\n==========================================")
print("FINAL MODEL EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print("\nFinal Test Accuracy : {:.2f}%".format(test_accuracy * 100))
print("Final Test Loss     : {:.4f}".format(test_loss))

##########################################################
# PREDICTIONS
##########################################################

print("\nGenerating Predictions...\n")

predictions = model.predict(

    test_generator,

    verbose=1

)

y_pred = np.argmax(

    predictions,

    axis=1

)

y_true = test_generator.classes

##########################################################
# CLASSIFICATION REPORT
##########################################################

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=list(

            train_generator.class_indices.keys()

        )

    )

)

##########################################################
# CONFUSION MATRIX
##########################################################

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")

cm = confusion_matrix(

    y_true,

    y_pred

)

print(cm)

##########################################################
# ACCURACY GRAPH
##########################################################

train_acc = (
    history.history["accuracy"] +
    history_finetune.history["accuracy"]
)

val_acc = (
    history.history["val_accuracy"] +
    history_finetune.history["val_accuracy"]
)

plt.figure(figsize=(8,5))

plt.plot(

    train_acc,

    label="Training Accuracy",

    linewidth=2

)

plt.plot(

    val_acc,

    label="Validation Accuracy",

    linewidth=2

)

plt.title("ResNet50 Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    "models/accuracy_graph.png",

    dpi=300

)

plt.show()

##########################################################
# LOSS GRAPH
##########################################################

train_loss = (
    history.history["loss"] +
    history_finetune.history["loss"]
)

val_loss = (
    history.history["val_loss"] +
    history_finetune.history["val_loss"]
)

plt.figure(figsize=(8,5))

plt.plot(

    train_loss,

    label="Training Loss",

    linewidth=2

)

plt.plot(

    val_loss,

    label="Validation Loss",

    linewidth=2

)

plt.title("ResNet50 Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    "models/loss_graph.png",

    dpi=300

)

plt.show()

##########################################################
# SAVE FINAL MODEL
##########################################################

model.save(

    "models/cnn.h5"

)

print("\ncnn.h5 Saved Successfully.")

##########################################################
# SAVE CLASS NAMES
##########################################################

with open(

    "models/classes.json",

    "w"

) as f:

    json.dump(

        train_generator.class_indices,

        f,

        indent=4

    )

print("classes.json Saved Successfully.")

##########################################################
# FINISHED
##########################################################

print("\n==========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nGenerated Files:")

print("✔ models/cnn.h5")
print("✔ models/resnet50_phase1.h5")
print("✔ models/resnet50_finetuned.h5")
print("✔ models/classes.json")
print("✔ models/accuracy_graph.png")
print("✔ models/loss_graph.png")

print("\nYour ResNet50 model is now ready for Flask deployment.")