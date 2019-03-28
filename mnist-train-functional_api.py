import tensorflow as tf

#Callback function to stop training when the accuracy reaches 99%
class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc') > 0.99:
      print("\nTraining accuracy reached 99%. Stopping Training.")
      self.model.stop_training = True



mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


training_images = training_images.reshape(60000,28,28,1) / 255.0
test_images = test_images.reshape(10000,28,28,1) / 255.0


X_input = tf.keras.Input(shape=(28,28,1))
X = tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu)(X_input)
X = tf.keras.layers.MaxPooling2D(2,2)(X)
X = tf.keras.layers.Flatten()(X)
X = tf.keras.layers.Dense(128, activation=tf.nn.relu)(X)
X = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(X)
model = tf.keras.models.Model(inputs=X_input, outputs=X)


model.summary()

callback = myCallbacks()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callback])
