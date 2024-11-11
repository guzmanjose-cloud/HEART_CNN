import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib

# Define the image size (height, width, channels)
IMG_SIZE = (224, 224)  # Updated size for your images
NUM_CLASSES = 4  # Number of heart disease types (based on 4 folders)

# Function to load images from folders
def load_images_from_folder(folder_path):
    images = []
    labels = []
    label = 0
    for disease_folder in os.listdir(folder_path):
        disease_folder_path = os.path.join(folder_path, disease_folder)
        if os.path.isdir(disease_folder_path):  # Skip non-directory files like .DS_Store
            print(f"Disease folder: {disease_folder} - Assigned Label: {label}")  # Print folder name and assigned label
            for filename in os.listdir(disease_folder_path):
                img_path = os.path.join(disease_folder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)  # Resize all images
                    images.append(np.array(img))
                    labels.append(label)  # Assign a numeric label for each disease type
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            label += 1
    return np.array(images), np.array(labels)

# Load the dataset
train_images, train_labels = load_images_from_folder('/Users/joseguzman/Desktop/heart_data/training')

# Load the validation dataset
val_images, val_labels = load_images_from_folder('/Users/joseguzman/Desktop/heart_data/validation')

# Normalize the images to range [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# One-hot encode the target labels for training and validation
train_labels_one_hot = to_categorical(train_labels, num_classes=NUM_CLASSES)
val_labels_one_hot = to_categorical(val_labels, num_classes=NUM_CLASSES)

# Flatten each image for KMeans clustering
train_images_flattened = train_images.reshape(train_images.shape[0], -1)

# Apply KMeans to the flattened images
kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=40)
kmeans.fit(train_images_flattened)

# Save the KMeans model
joblib.dump(kmeans, '/Users/joseguzman/Desktop/heart_data/kmeans_model.pkl')

# Get cluster labels for each image
cluster_labels = kmeans.labels_

# One-hot encode the cluster labels
cluster_labels_one_hot = to_categorical(cluster_labels, num_classes=NUM_CLASSES)

# Expand the one-hot encoded cluster labels to match the image dimensions
cluster_labels_one_hot_expanded = np.tile(cluster_labels_one_hot[:, np.newaxis, np.newaxis, :], (1, 224, 224, 1))

# Concatenate the original images with the expanded cluster labels
train_images_with_clusters = np.concatenate([train_images, cluster_labels_one_hot_expanded], axis=-1)

# Validate images
val_images_flattened = val_images.reshape(val_images.shape[0], -1)
val_cluster_labels = kmeans.predict(val_images_flattened)
val_cluster_labels_one_hot = to_categorical(val_cluster_labels, num_classes=NUM_CLASSES)
val_cluster_labels_one_hot_expanded = np.tile(val_cluster_labels_one_hot[:, np.newaxis, np.newaxis, :], (1, 224, 224, 1))

# Concatenate validation images with the expanded cluster labels
val_images_with_clusters = np.concatenate([val_images, val_cluster_labels_one_hot_expanded], axis=-1)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce the data to 2D using PCA for visualization
pca = PCA(n_components=2)
train_images_2d = pca.fit_transform(train_images_flattened)

# Plot the clusters in 2D
plt.figure(figsize=(10, 7))
plt.scatter(train_images_2d[:, 0], train_images_2d[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.colorbar(label='Cluster Label')
plt.title('KMeans Clusters in 2D after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Apply PCA to reduce validation images to 2D for visualization
val_images_2d = pca.transform(val_images_flattened)  # Use the same PCA instance for consistency

# Get cluster labels for validation data
val_cluster_labels = kmeans.predict(val_images_flattened)

# Plot the validation clusters in 2D
plt.figure(figsize=(10, 7))
plt.scatter(val_images_2d[:, 0], val_images_2d[:, 1], c=val_cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.colorbar(label='Cluster Label')
plt.title('KMeans Clusters for Validation Data in 2D after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import mode

# Get the predicted cluster labels for the validation set
val_cluster_labels = kmeans.predict(val_images_flattened)

# Map each cluster to the most frequent true label within it
cluster_to_label = {}
for cluster in range(NUM_CLASSES):
    mask = (val_cluster_labels == cluster)
    most_common_label = mode(val_labels[mask])[0][0]  # Find the most common true label for this cluster
    cluster_to_label[cluster] = most_common_label

# Convert cluster predictions to label predictions based on the mapping
predicted_labels = np.array([cluster_to_label[cluster] for cluster in val_cluster_labels])

# Calculate accuracy
accuracy = accuracy_score(val_labels, predicted_labels)
print("Validation Accuracy:", accuracy)




# Define your CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 7)))  # Reduced filters
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # Reduced filters
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  # Reduced units
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(NUM_CLASSES, activation='softmax'))  # Output layer for categorical classification

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Display the model architecture
    model.summary()
 
    return model

# Create and train the model on the dataset with cluster labels
model = create_cnn_model()  # Define your CNN architecture
history = model.fit(
    train_images_with_clusters,
    train_labels_one_hot,
    epochs=10,
    batch_size=32,
    validation_data=(val_images_with_clusters, val_labels_one_hot)
)

# Save the model in TensorFlow SavedModel format
#model.save('/Users/joseguzman/Desktop/heart_data/heart_disease_model')

# After defining and training the model
#model.save('/Users/joseguzman/Desktop/heart_data/heart_disease_model.h5')

model.save('/Users/joseguzman/Desktop/heart_disease_model.keras')
