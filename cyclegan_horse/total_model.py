from prepare import *
from build_model import *

class CycleGan(tf.keras.Model):
    def __init__(self,generator_G,generator_F,discriminator_X,discriminator_Y,lambda_cycle=10.0,lambda_identity=0.5,):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self,gen_G_optimizer,gen_F_optimizer,disc_X_optimizer,disc_Y_optimizer,gen_loss_fn,disc_loss_fn,):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = self.identity_loss_fn(real_y, same_y)* self.lambda_cycle* self.lambda_identity
            id_loss_F = self.identity_loss_fn(real_x, same_x)* self.lambda_cycle* self.lambda_identity

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F
            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

            # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(zip(grads_G, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(grads_F, self.gen_F.trainable_variables))

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(zip(disc_X_grads, self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "G_adv_loss": gen_G_loss,
            "F_adv_loss": gen_F_loss,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
            "Cycle_loss_G": cycle_loss_G,
            "Cycle_loss_F": cycle_loss_F,
            "id_loss_G": id_loss_G,
            "id_loss_F": id_loss_F
        }


# {
#             "G_loss": total_loss_G,
#             "F_loss": total_loss_F,
#             "D_X_loss": disc_X_loss,
#             "D_Y_loss": disc_Y_loss,
        # }

# {"G_adv_loss": gen_G_loss, "F_adv_loss": gen_F, "D_X_loss": disc_X_loss, "D_Y_loss": disc_Y_loss, "Cycle_loss_G": cycle_loss_G,
#                 "Cycle_loss_F": cycle_loss_F, "id_loss_G": id_loss_G, "id_loss_F": id_loss_F}