from prepare import *
from build_model import *
from total_model import *

class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(2, 4, figsize=(15, 4))
        for i, img in enumerate(test_horses.take(8)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            plt.subplot(2,8,i+1)
            plt.title("Domain A")
            plt.axis("off")
            plt.imshow(img)
            plt.subplot(2, 8, i + 9)
            plt.title("Domain B")
            plt.axis("off")
            plt.imshow(prediction)
            # ax[i, 0].imshow(img)
            # ax[i, 1].imshow(prediction)
            # ax[i, 0].set_title("Domain A")
            # ax[i, 1].set_title("Domain B")
            # ax[i, 0].axis("off")
            # ax[i, 1].axis("off")

            prediction = tf.keras.preprocessing.image.array_to_img(prediction)
            prediction.save("result_image/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 71))
            prediction.close()
        plt.savefig(f"result_image/sample{epoch+71}")
        plt.close()

# Loss function for evaluating adversarial loss
adv_loss_fn = tf.keras.losses.MeanSquaredError()
# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,disc_loss_fn=discriminator_loss_fn,)
# Callbacks
plotter = GANMonitor()
checkpoint_filepath = "model_weight/model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)


from os import path
weight_file = "model_weight/model_checkpoints/cyclegan_checkpoints.020"
# if not path.exists(weight_file+'.index'):
    # Here we will train the model for just one epoch as each epoch takes around
    # 7 minutes on a single P100 backed machine.
cycle_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")
history = cycle_gan_model.fit(tf.data.Dataset.zip((train_horses, train_zebras)), epochs=20, callbacks=[plotter,model_checkpoint_callback])
# else:
#     # Load the checkpoints
#     cycle_gan_model.load_weights(weight_file).expect_partial()
#     print("Weights loaded successfully")

print(history.history["G_adv_loss"])
G_adv_loss = history.history['G_adv_loss']
F_adv_loss = history.history['F_adv_loss']
D_X_loss = history.history['D_X_loss']
D_Y_loss = history.history['D_Y_loss']
Cycle_loss_G = history.history['Cycle_loss_G']
Cycle_loss_F = history.history['Cycle_loss_F']
id_loss_G = history.history['id_loss_G']
id_loss_F = history.history['id_loss_F']

plt.plot(G_adv_loss)
plt.plot(F_adv_loss)
plt.plot(D_X_loss)
plt.plot(D_Y_loss)
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(["G_adv_loss", "F_adv_loss", "D_X_loss", "D_Y_loss"], loc="upper right")
plt.title("the adv loss")
plt.savefig("result_image/the adv loss")
plt.close()

plt.plot(Cycle_loss_G)
plt.plot(Cycle_loss_F)
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(["Cycle_loss_G", "Cycle_loss_F"], loc="upper right")
plt.title("the cycle loss")
plt.savefig("result_image/the cycle loss")
plt.close()

plt.plot(id_loss_G)
plt.plot(id_loss_F)
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(["id_loss_G", "id_loss_F"], loc="upper right")
plt.title("the id loss")
plt.savefig("result_image/the id loss")
plt.close()





_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, img in enumerate(test_horses.take(4)):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    prediction = tf.keras.preprocessing.image.array_to_img(prediction)
    prediction.save("result_image/predicted_img_{i}.png".format(i=i))
plt.tight_layout()
plt.close()