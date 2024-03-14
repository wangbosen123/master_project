from train import *


model = load_model("pretrain_fer.h5")
weights = model.get_weights()
print(weights)
model.summary()


input = Input((224,224,3))
x = input
for layer in model.layers[1:19]:
    x = layer(x)

model = Model(input, x)
model.summary()

# model.save("change_input_shape224_224_encoder.h5")

model = load_model("change_input_shape224_224_encoder.h5")
model.summary()
weights = model.get_weights()
print(weights)


x = Flatten()(model.output)
x = Dense(128,activation=LeakyReLU(0.3))(x)
x = Dense(64,activation=LeakyReLU(0.3))(x)
x = Dense(7,activation=LeakyReLU(0.3))(x)
overall_model = Model(model.input,x)
overall_model.summary()