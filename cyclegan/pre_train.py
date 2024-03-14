import matplotlib.pyplot as plt

from build_model import *
from deal_data import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from loss import *
import time

domain_a, domain_b = load_image()


# model = load_model("generator_domain_a")
#
generator = build_generator()
optimizer = tf.keras.optimizers.Adam(1e-4)
generator.compile(optimizer=optimizer, loss="mse")
generator.fit(x=domain_a,y=domain_b, verbose=1, epochs=50, batch_size=20)
generator.save("generator_domain_a")

# model = load_model("generator_domain_a")
#
# plt.imshow(nature,cmap="gray")
# plt.show()
# nature = nature.reshape(1,128,128,1)
# pred = model(nature)
# pred = tf.reshape(pred,[128,128])
# plt.imshow(pred, cmap="gray")
# plt.show()