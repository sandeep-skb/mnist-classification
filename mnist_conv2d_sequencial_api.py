# This code classifies mnist dataset with convolutions in Sequential API.

import tensorflow as tf


#Callback function to stop training when required accuracy is reached.
class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc') > 0.99:
      print("\nTraining accuracy reached 99%. Stopping Training.")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#rescaling and reshaping before feeding into conv layers. convolution expects 3 dimensions, so reshaping to include channels.
training_images = training_images.reshape(60000, 28,28,1)
training_images = training_images/255.0
test_images = test_images.reshape(10000,28, 28, 1)
test_images = test_images/255.0

#Model construction
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28,28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=100, epochs=5)

model.evaluate(test_images, test_labels)
