# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv("./datasets/dataset1.csv")
print(df.head())
# %%
runs, volt, ip, ton, toff, ra = (
    df["run"],
    df["volt"],
    df["ip"],
    df["ton"],
    df["toff"],
    df["ra"],
)
plt.plot([1, 2, 3], [4, 5, 7])
plt.show()
# %%
imagesample = np.array(
    [
        [
            [120, 100, 254],
            [100, 245, 200],
            [100, 245, 200],
            [100, 245, 200],
            [100, 245, 200],
            [254, 120, 254],
        ],
        [
            [120, 100, 100],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 100, 120],
        ],
        [
            [120, 100, 100],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 100, 120],
        ],
        [
            [120, 100, 100],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254],
            [254, 100, 120],
        ],
        [
            [120, 100, 254],
            [100, 245, 200],
            [100, 245, 200],
            [100, 245, 200],
            [100, 245, 200],
            [254, 120, 254],
        ],
    ]
)


plt.imshow(imagesample)
plt.axis("off")
plt.show()
plt.imsave("image.png", imagesample.astype(np.uint8))
# %%
print(imagesample.ndim)
print(imagesample.shape)
# %%

h, w, c = imagesample.shape

target_h, target_w = 1920, 1080

scaled = np.zeros((target_h, target_w, c), dtype=int)

for i in range(target_h):
    for j in range(target_w):
        scaled[i, j] = imagesample[int(i * h / target_h)][int(j * w / target_w)]

print(scaled.shape)

plt.imshow(scaled)
plt.axis("off")
plt.show()
plt.imsave("image.png", scaled.astype(np.uint8))
# %%

h, w = 20, 20
n_colors = np.random.randint(10, 100)

shape = np.random.randint(0, n_colors, (h, w))

for _ in range(5):
    new_shape = shape.copy()
    for i in range(h):
        for j in range(w):
            neighbors = shape[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2]
            new_shape[i, j] = np.bincount(neighbors.flatten()).argmax()
    shape = new_shape

palette = np.random.default_rng().integers(0, 256, (n_colors, 3))
image = palette[shape]

target_h, target_w = 1920, 1080

scaled = np.zeros((target_h, target_w, 3), dtype=np.uint8)

for i in range(target_h):
    for j in range(target_w):
        scaled[i, j] = image[int(i * h / target_h)][int(j * w / target_w)]

plt.imshow(scaled)
plt.axis("off")
plt.show()
plt.imsave("image1.png", scaled.astype(np.uint8))
