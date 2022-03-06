import matplotlib.pyplot as plt
import app
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

import numpy

from sklearn.metrics import classification_report, confusion_matrix

data_generator=ImageDataGenerator(rescale=1.0/255, vertical_flip=True, zoom_range=0.2, rotation_range=30, width_shift_range=0.05, height_shift_range=0.05)
train_iterator=data_generator.flow_from_directory('augmented-data/train', class_mode='categorical', color_mode='grayscale', batch_size=8, target_size=(256,256))
test_iterator=data_generator.flow_from_directory('augmented-data/test', class_mode='categorical', color_mode='grayscale', batch_size=8, target_size=(256,256))

model=Sequential()
model.add(keras.Input(shape=(256, 256, 1)))
model.add(layers.Conv2D(8, 3, strides=2, padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(8, 3, strides=2, padding='valid', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.SGD(learning_rate=0.05), metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.AUC()])

history = model.fit(train_iterator, steps_per_epoch=train_iterator.samples/8, epochs=10, validation_data=test_iterator, validation_steps=test_iterator.samples/8)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
fig.tight_layout()
 
fig.savefig('static/images/my_plots.png')

test_steps_per_epoch = numpy.math.ceil(test_iterator.samples / test_iterator.batch_size)
predictions = model.predict(test_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = numpy.math.ceil(test_iterator.samples / test_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = test_iterator.classes
class_labels = list(test_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)
print(cm)


# Do Matplotlib extension below

# use this savefig call at the end of your graph instead of using plt.show()
# plt.savefig('static/images/my_plots.png')

