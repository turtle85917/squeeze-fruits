import keras
import numpy
import gan
import matplotlib.pyplot as plt
from utils import getNoise
from datetime import datetime
from tensorflow import function as tfuction, GradientTape, ones_like, zeros_like

# Environment
epochs = 500
batchSize = 64

# Load datset
trainDs = keras.utils.image_dataset_from_directory(
  "dataset",
  image_size=(64, 64), # 100x100 -> 64x64
  batch_size=batchSize,
  shuffle=True
).map(lambda x, y: (x / 255., y))

loss = keras.losses.BinaryCrossentropy(from_logits=True)
doptimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
goptimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

generator = gan.buildGenerator()
discriminator = gan.buildDiscriminator()
generator.compile(loss=loss, optimizer=goptimizer, metrics=["accuracy"])
discriminator.compile(loss=loss, optimizer=doptimizer, metrics=["accuracy"])

@tfuction
def trainStep(realImages):
  noise = getNoise(batchSize, gan.latentDim)
  with GradientTape() as g, GradientTape() as d:
    genImage = generator(noise, training=True)
    fakeOutput = discriminator(genImage, training=True)
    realOutput = discriminator(realImages, training=True)
    dLossReal = loss(ones_like(realOutput), realOutput)
    dLossFake = loss(zeros_like(fakeOutput), fakeOutput)
    dLoss = (dLossReal + dLossFake) * 0.5
    gLoss = loss(ones_like(fakeOutput), fakeOutput)
  dgradients = d.gradient(dLoss, discriminator.trainable_variables)
  doptimizer.apply_gradients(zip(dgradients, discriminator.trainable_variables))
  ggradients = g.gradient(gLoss, generator.trainable_variables)
  goptimizer.apply_gradients(zip(ggradients, generator.trainable_variables))
  return dLoss, gLoss

startTime = datetime.now()

for epoch in range(epochs):
  for realImages, _ in trainDs:
    d_loss, g_loss = trainStep(realImages)
  print(f"Epoch {epoch + 1}/{epochs} | D Loss={d_loss:.4f} | G Loss={g_loss:.4f}")

endTime = datetime.now()
print(f"최종 학습 소요 시간 : {endTime - startTime}")

generator.save("generator.keras")
discriminator.save("discriminator.keras")

noise = numpy.random.normal(0, 1, (1, gan.latentDim))
generated = generator(noise, training=False)
plt.imshow((generated[0] * 255.).numpy().astype("uint8"))
plt.title("Generated Image")
plt.tight_layout()
plt.show()
