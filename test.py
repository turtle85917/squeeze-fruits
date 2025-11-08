import keras
import matplotlib.pyplot as plt
from gan import latentDim
from utils import getNoise

model = keras.models.load_model("generator.keras")

fig, axes = plt.subplots(5, 5)

for i in range(5):
  for j in range(5):
    z = getNoise(1, latentDim)
    g = model(z)
    axes[i, j].imshow((g[0] * 255.).numpy().astype("uint8"))

plt.show()
