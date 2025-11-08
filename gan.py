import keras

latentDim = 100

def buildGenerator():
  return keras.Sequential([
    keras.layers.Input(shape=(latentDim,)),
    keras.layers.Dense(8 * 8 * 256, activation="relu"),
    keras.layers.Reshape((8, 8, 256)),
    keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
    keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid")
  ])

def buildDiscriminator():
  return keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Conv2D(64, 4, strides=2, padding="same"), #, activation="relu"),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, 4, strides=2, padding="same"), #, activation="relu"),
    keras.layers.LeakyReLU(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
  ])
