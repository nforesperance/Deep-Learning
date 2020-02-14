import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# Creating a model and flattenig our data
# We are going to have 128 hidden layers
# and 10 tables
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax") #softmax => sum of output of 1
	])
# Training the model 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# Testing our model 
test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('\nTest accuracy:', test_acc)

# Making a prediction
predictions = model.predict(test_images)
# print(class_names[np.armax(predictions[0])])

# Now we will display the first 5 images and their predictions using matplotlib 
plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual" +class_names[test_labels[i]])
    plt.title("Prediction"+class_names[np.argmax(predictions[i])])
    plt.show()