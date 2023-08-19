# Import TensorFlow and Keras from TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
# ... (previous code remains the same) ...
from sklearn.metrics import confusion_matrix, classification_report

# Defining the paths to the dataset
train_data = 'train1'
test_data = 'val'

# Initializing the CNN
np.random.seed(1337)

# Defining the number of classes and other parameters
num_classes = 6
input_shape = (64, 64, 3)
batch_size = 16
epochs = 200

# Normalization
train_data_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess and augment the training images
train_generator = train_data_gen.flow_from_directory(
    train_data,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and preprocess the testing dataset
test_generator = test_data_gen.flow_from_directory(
    test_data,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN classifier architecture with L2 regularization
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu', kernel_regularizer='l2'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compile the classifier
classifier.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Add EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-8)

# Train the classifier
history = classifier.fit(train_generator,
                    steps_per_epoch=train_generator.n // batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // batch_size,
                    callbacks=[early_stopping, reduce_lr])

# Save the classifier
classifier.save('tomato_cnn_classifier.h5')

# Evaluate the classifier on the testing dataset
test_loss, test_accuracy = classifier.evaluate(test_generator)

# Compute the predicted labels for the testing dataset
predicted_labels = classifier.predict(test_generator)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Get the true labels for the testing dataset
true_labels = test_generator.classes

# Compute the accuracy
accuracy = np.sum(predicted_classes == true_labels) / len(true_labels)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
print('Computed Accuracy:', accuracy)

# Compute confusion matrix
confusion = confusion_matrix(true_labels, predicted_classes)

# Print confusion matrix in a good format
print("Confusion Matrix:")
print(confusion)

# Compute and print accuracy per class
class_accuracy = confusion.diagonal() / confusion.sum(axis=1)
class_names = list(test_generator.class_indices.keys())

print("\nAccuracy per class:")
for class_name, acc in zip(class_names, class_accuracy):
    print(f"{class_name}: {acc:.4f}")

# Plot training and validation accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'bo-', label='Train')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()