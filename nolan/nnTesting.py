### GOAL OF THIS FUNC IS TO TEST NN MODELING AND TRAINING ###

import matplotlib.pyplot as plt
import numpy as np

# Parameters for projectile motion
g = 9.81  # gravity, m/s^2
v0 = 20  # initial velocity, m/s
angle = 45  # launch angle in degrees
t_total = 5  # total time, seconds

# Convert angle to radians
angle_rad = np.radians(angle)

# Initial velocity components
vx0 = v0 * np.cos(angle_rad)
vy0 = v0 * np.sin(angle_rad)

# Time points
t = np.linspace(0, t_total, num=10000)

# Position equations
x = vx0 * t
y = vy0 * t - 0.5 * g * t**2

# Combine x and y positions as features and label y as target
data = np.column_stack((x, t))  # features: [x, t]
target = y  # target: y position

# Add noise to the target
noise = np.random.normal(0, 1, len(target))
target_noisy = target + noise

# Plot the truth
plt.figure(figsize=(10, 6))
plt.plot(
    data[:, 0],
    target,
    label='True Projectile Motion Data',
    linestyle='--',
    color='black',
)
plt.scatter(data[:, 0], target_noisy, label='Noisy Projectile Motion Data')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Projectile Motion Data with Noise')
plt.legend()
plt.grid(True)
# plt.show()


# ######### Will use tensorflow, keras #########
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Define the neural network
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define the neural network
model = Sequential(
    [
        Dense(
            64, activation='relu', input_shape=(2,)
        ),  # input layer with 2 features: x and t
        Dense(64, activation='relu'),  # hidden layer
        Dense(1),  # output layer: y (vertical position)
    ]
)

# Compile the model with MSE loss and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

# Train the model for 100 epochs
history = model.fit(
    x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1
)

# Save the trained model
model.save('projectile_motion_model.h5')

# Evaluate the model on the test data
test_loss = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")

# Make predictions for new data and plot them with a big red X
predictions = model.predict(x_test)

# plt.figure(figsize=(10, 6))
# plt.scatter(x_test[:, 0], y_test, label='True Projectile Motion Data')
plt.scatter(
    x_test[:, 0], predictions, label='Predicted Projectile Motion Data', color='red'
)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Projectile Motion Data with Predictions')
plt.legend()
plt.show()


#### NOW, TEST PULLING A TRAINED MODEL FROM FILE ####

# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('./nolan/projectile_motion_model.h5')

# Now, plot a ton of predictions
predictions = model.predict(data)
# plt.figure(figsize=(10, 6))
# plt.scatter(data[:, 0], target, label='True Projectile Motion Data')
plt.scatter(
    data[:, 0], predictions, label='Predicted Projectile Motion Data', color='red'
)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Projectile Motion Data with Predictions')
plt.legend()
plt.show()
