import os
import json
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#########################################
# Parameters
#########################################

IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 1

train_path = "dataset/train"
test_path = "dataset/test"

#########################################
# Create folders
#########################################

os.makedirs("models", exist_ok=True)

os.makedirs("static", exist_ok=True)

#########################################
# Data Generators
#########################################

train_gen = ImageDataGenerator(

    rescale=1./255

)

test_gen = ImageDataGenerator(

    rescale=1./255

)

#########################################
# Load Dataset
#########################################

train_data = train_gen.flow_from_directory(

    train_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'

)

test_data = test_gen.flow_from_directory(

    test_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'

)

#########################################
# Save Classes
#########################################

classes = train_data.class_indices

with open(

    'models/classes.json',

    'w'

) as f:

    json.dump(

        classes,

        f

    )

print(

    "classes.json Saved Successfully"

)

print(

    classes

)

num_classes = len(classes)

#########################################
# Disease Distribution Graph
#########################################

labels = list(classes.keys())

counts = []

for label in labels:

    folder = os.path.join(

        train_path,

        label

    )

    counts.append(

        len(

            os.listdir(folder)

        )

    )

plt.figure(

    figsize=(12,5)

)

plt.bar(

    labels,

    counts

)

plt.xticks(

    rotation=90

)

plt.title(

    "Disease Distribution"

)

plt.tight_layout()

plt.savefig(

    "static/disease_distribution.png"

)

plt.close()

#########################################
# Train/Test Pie Chart
#########################################

train_count = len(

    train_data.filenames

)

test_count = len(

    test_data.filenames

)

plt.figure(

    figsize=(6,6)

)

plt.pie(

    [

        train_count,

        test_count

    ],

    labels=[

        'Train',

        'Test'

    ],

    autopct='%1.1f%%'

)

plt.title(

    "Train Test Distribution"

)

plt.savefig(

    "static/train_test.png"

)

plt.close()

#########################################
# CNN Model
#########################################

model = Sequential()

model.add(

    Conv2D(

        16,

        (3,3),

        activation='relu',

        input_shape=(64,64,3)

    )

)

model.add(

    MaxPooling2D()

)

model.add(

    Conv2D(

        32,

        (3,3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D()

)

model.add(

    Flatten()

)

model.add(

    Dense(

        64,

        activation='relu'

    )

)

model.add(

    Dense(

        num_classes,

        activation='softmax'

    )

)

#########################################
# Compile
#########################################

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

#########################################
# Training
#########################################

history = model.fit(

    train_data,

    validation_data=test_data,

    epochs=EPOCHS

)

#########################################
# Accuracy Graph
#########################################

plt.figure()

plt.plot(

    history.history['accuracy']

)

plt.plot(

    history.history['val_accuracy']

)

plt.title(

    "Accuracy Curve"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Accuracy"

)

plt.legend(

    [

        'Train',

        'Validation'

    ]

)

plt.savefig(

    "static/accuracy.png"

)

plt.close()

#########################################
# Loss Graph
#########################################

plt.figure()

plt.plot(

    history.history['loss']

)

plt.plot(

    history.history['val_loss']

)

plt.title(

    "Loss Curve"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.legend(

    [

        'Train',

        'Validation'

    ]

)

plt.savefig(

    "static/loss.png"

)

plt.close()

#########################################
# Save CNN Model
#########################################

model.save(

    "models/cnn.h5"

)

print(

    "cnn.h5 Saved Successfully"

)