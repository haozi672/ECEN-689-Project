import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

# Load the data
file_path = '/home/thenaz/Documents/ECEN 689/Project/DEMAND_DATA_SET_TEXAS.xlsx'
df = pd.read_excel(file_path)

# Split data into features and targets
features = df.iloc[:, 9:62]  # Adjust indices as per your dataset
targets = df.iloc[:, 1:9]

# Split the dataset into training, validation, and test sets
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=42)
train_features, val_features, train_targets, val_targets = train_test_split(train_features, train_targets, test_size=0.1, random_state=42)

# Normalize features and targets with explicit data type setting
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_features = scaler.fit_transform(train_features).astype(np.float32)
scaled_test_features = scaler.transform(test_features).astype(np.float32)
scaled_val_features = scaler.transform(val_features).astype(np.float32)

# Normalize targets
scaler_targets = MinMaxScaler(feature_range=(0, 1))
scaled_train_targets = scaler_targets.fit_transform(train_targets).astype(np.float32)
scaled_test_targets = scaler_targets.transform(test_targets).astype(np.float32)
scaled_val_targets = scaler_targets.transform(val_targets).astype(np.float32)

batch_size = 32  # Define batch size globally
train_dataset = tf.data.Dataset.from_tensor_slices((scaled_train_features, scaled_train_targets)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((scaled_val_features, scaled_val_targets)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((scaled_test_features, scaled_test_targets)).batch(batch_size)

class EnergyDemandPINN(tf.keras.Model):
    def __init__(self):
        super(EnergyDemandPINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(units=8, activation=None)  

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)
    

def thermal_loss(y_pred, T, T_heat, T_cool, lambda_heat, lambda_cool):
    T = tf.cast(T, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)  
    heating_demand = tf.maximum(0., T_heat - T)
    cooling_demand = tf.maximum(0., T - T_cool)
    return lambda_heat * tf.reduce_mean(tf.square(heating_demand)) + lambda_cool * tf.reduce_mean(tf.square(cooling_demand))

def socioeconomic_loss(y_pred, GDP, Population, alpha, beta):
   
    GDP = tf.expand_dims(GDP, -1)
    Population = tf.expand_dims(Population, -1)  
    

    GDP = tf.broadcast_to(GDP, y_pred.shap)
    Population = tf.broadcast_to(Population, y_pred.shape)  
    
    expected_demand = alpha * GDP + beta * Population
    return tf.reduce_mean(tf.square(y_pred - expected_demand))

def capacity_loss(y_pred, CAP_max):
    y_pred = tf.cast(y_pred, tf.float32)  
    return tf.reduce_mean(tf.maximum(0., y_pred - CAP_max))

optimizer = tf.optimizers.Adam(learning_rate=1e-3)


idx_temperature = 0  
idx_gdp = 1          
idx_population = 2   

# Instantiate the model
model = EnergyDemandPINN()
model.compile(optimizer='adam', loss='mse')

# Use fit to train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Initialising the hyperparameters
T_heat = 18          # Heating threshold temperature
T_cool = 24          # Cooling threshold temperature
lambda_heat = 0.33    # Weight for heating demand in the loss function
lambda_cool = 0.33   # Weight for cooling demand in the loss function
alpha = 0.9          # Weight for GDP in the loss function
beta =  1.0          # Weight for population in the loss function
CAP_max = 100000     # Maximum capacity for energy demand

# Update the train_step definition if necessary to ensure it accepts all necessary parameters
@tf.function
def train_step(model, features, targets, idx_temperature, idx_gdp, idx_population,
               T_heat, T_cool, lambda_heat, lambda_cool, alpha, beta, CAP_max):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss_data = tf.reduce_mean(tf.square(targets - predictions))
        
        # Extract specific features for the physics-informed loss
        T = features[:, idx_temperature]
        GDP = features[:, idx_gdp]
        Population = features[:, idx_population]
        
        # Debugging
        tf.print("Shapes:", tf.shape(predictions), tf.shape(GDP), tf.shape(Population))

        # Compute physics-informed loss components
        loss_thermal = thermal_loss(predictions, T, T_heat, T_cool, lambda_heat, lambda_cool)
        loss_socioeconomic = socioeconomic_loss(predictions, GDP, Population, alpha, beta)
        loss_capacity = capacity_loss(predictions, CAP_max)
        
        # Combine all losses
        loss_total = loss_data + loss_thermal + loss_socioeconomic + loss_capacity

    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_total

# Prepare for training
train_dataset = tf.data.Dataset.from_tensor_slices((scaled_train_features, scaled_train_targets)).batch(32)

# Run the training loop
num_batches = len(scaled_train_features) // batch_size
for epoch in range(100):
    for step in range(num_batches):  
        x_batch_train, y_batch_train = next(iter(train_dataset))
        loss_value = train_step(model, x_batch_train, y_batch_train, idx_temperature, idx_gdp, idx_population,
                                T_heat, T_cool, lambda_heat, lambda_cool, alpha, beta, CAP_max)
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss_value.numpy()}")
            
val_loss = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss}")


predicted_targets = model.predict(test_dataset)
actual_targets = scaler_targets.inverse_transform(scaled_test_targets)
predicted_targets = scaler_targets.inverse_transform(predicted_targets)

# Plotting multiple targets
num_targets = actual_targets.shape[1] 

plt.figure(figsize=(18, 4 * num_targets))

for i in range(num_targets):
    plt.subplot(num_targets, 1, i + 1)
    plt.plot(actual_targets[:, i], label='Actual', linewidth=2) 
    plt.plot(predicted_targets[:, i], label='Predicted', linestyle='--', linewidth=2)  
    plt.title(f'Target {i + 1}', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)  
    plt.ylabel('Value', fontsize=12)  
    plt.legend(loc='upper right')  

plt.tight_layout()  
plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0  
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Calculate and print the MAPE for each target column in the training set
train_mape = []
print("Training Set MAPE:")
for i in range(actual_targets.shape[1]):
    mape = mean_absolute_percentage_error(scaled_train_targets[:, i], model.predict(train_dataset)[:, i])
    train_mape.append(mape)
    print(f"Target Column {i+1} - MAPE: {mape:.4f}%")

# Calculate and print the MAPE for each target column in the validation set
val_mape = []
print("\nValidation Set MAPE:")
for i in range(val_targets.shape[1]):
    mape = mean_absolute_percentage_error(scaled_val_targets[:, i], model.predict(val_dataset)[:, i])
    val_mape.append(mape)
    print(f"Target Column {i+1} - MAPE: {mape:.4f}%")

# Calculate and print the MAPE for each target column in the test set
test_mape = []
print("\nTest Set MAPE:")
for i in range(test_targets.shape[1]):
    mape = mean_absolute_percentage_error(scaled_test_targets[:, i], model.predict(test_dataset)[:, i])
    test_mape.append(mape)
    print(f"Target Column {i+1} - MAPE: {mape:.4f}%")
