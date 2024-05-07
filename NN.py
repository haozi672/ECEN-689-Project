import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# NN
def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation='tanh', input_shape=(2,)),
        layers.Dense(64, activation='tanh'),
        layers.Dense(1)  # temperature output
    ])
    return model


def loss_fn(data, model):
    z, tau = data
    inputs = tf.concat([z, tau], axis=-1)
    inputs = tf.expand_dims(inputs, axis=-1)

    predictions = model(inputs)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        u = model(inputs)
        du_dz, du_dtau = tape.gradient(u, inputs)
    du_dz_z, du_dtau_tau = tape.gradient([du_dz, du_dtau], inputs)
    physics_loss = tf.reduce_mean(tf.square(du_dtau - 0.5 * sigma ** 2 * (T - tau) * du_dz_z
                                            - r * u + mu * (T - tau) * du_dz))

    return physics_loss


def generate_data(n_samples):
    z = np.random.uniform(z_min, z_max, size=(n_samples,))
    tau = np.random.uniform(tau_min, tau_max, size=(n_samples,))
    return [z, tau]

z_min = 0
z_max = 5000
tau_min = 0
tau_max = 1

sigma = 0.1
T = 1
r = 0.05
mu = 0.02


model = build_model()


optimizer = keras.optimizers.Adam(learning_rate=0.001)
n_epochs = 1000
batch_size = 32

for epoch in range(n_epochs):
    training_data = generate_data(batch_size)
    z_train, tau_train = training_data
    with tf.GradientTape() as tape:
        loss = loss_fn(training_data, model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

z_test = np.array([z_value])
tau_test = np.array([tau_value])
option_value = model.predict(np.expand_dims([z_test, tau_test], axis=0))[0, 0]
print("Predicted option value:", option_value)

