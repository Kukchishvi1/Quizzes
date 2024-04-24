import ipaddress
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

df = pd.read_csv("Darknet.csv")
df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)
df = df.dropna()

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

# Save the preprocessed dataset with encoded labels to a CSV file
df.to_csv("preprocessed_encoded.csv", index=False)

# Remove rows containing problematic values
problematic_indices = np.any(np.isnan(df.values) | np.isinf(df.values), axis=1)
df = df[~problematic_indices]

# Clip extreme values to a specified range
clipped_df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(clipped_df.iloc[:, :-1])  # Exclude the target column 'Label1'

# Combine scaled features and labels into a DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=clipped_df.columns[:-1])  # Exclude the target column 'Label1'
df_scaled['Label1'] = clipped_df['Label1']

# Save the preprocessed dataset to a CSV file
df_scaled.to_csv("scaled3.csv", index=False)




#
#Now We Evaluate the Model
#

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Read the preprocessed dataset
df = pd.read_csv("scaled3.csv")

# Drop unnecessary columns
features = df.drop(['Label1'], axis=1)
label = df['Label1']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Number of classes in the label
num_classes = len(label_encoder.classes_)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use num_classes for output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=20, batch_size=500, validation_data=(X_test, y_test_encoded))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

