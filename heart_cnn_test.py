import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.utils import to_categorical
import joblib  # Ensure you have joblib installed

# Define constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 4

# Load the saved model
model = tf.keras.models.load_model('/Users/joseguzman/Desktop/heart_data/heart_disease_model.h5')

# Load the KMeans model
kmeans = joblib.load('/Users/joseguzman/Desktop/heart_data/kmeans_model.pkl')

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    labels = []
    label = 0
    for disease_folder in os.listdir(folder_path):
        disease_folder_path = os.path.join(folder_path, disease_folder)
        if os.path.isdir(disease_folder_path):
            print(f"Disease folder: {disease_folder} - Assigned Label: {label}")  # Print folder name and assigned label
            for filename in os.listdir(disease_folder_path):
                img_path = os.path.join(disease_folder_path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            label += 1
    return np.array(images), np.array(labels)

# Load the testing dataset
test_images, test_labels = load_images_from_folder('/Users/joseguzman/Desktop/heart_data/testing')

# Normalize the test images
test_images = test_images / 255.0

# One-hot encode the test labels
test_labels_one_hot = to_categorical(test_labels, num_classes=NUM_CLASSES)

# Flatten the test images for KMeans clustering
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Predict KMeans cluster labels for the test images
test_cluster_labels = kmeans.predict(test_images_flattened)

# One-hot encode the cluster labels for the test set
test_cluster_labels_one_hot = to_categorical(test_cluster_labels, num_classes=NUM_CLASSES)

# Expand the one-hot encoded cluster labels to match the image dimensions
test_cluster_labels_one_hot_expanded = np.tile(test_cluster_labels_one_hot[:, np.newaxis, np.newaxis, :], (1, IMG_SIZE[0], IMG_SIZE[1], 1))

# Concatenate the test images with the expanded cluster labels (resulting in 7-channel images)
test_images_with_clusters = np.concatenate([test_images, test_cluster_labels_one_hot_expanded], axis=-1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images_with_clusters, test_labels_one_hot, verbose=1)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
test_images_reduced = pca.fit_transform(test_images_flattened)

# Plot the clustered data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(test_images_reduced[:, 0], test_images_reduced[:, 1], c=test_cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title("KMeans Clustering of Testing Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Perform t-SNE to reduce dimensions to 2 for better visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
test_images_tsne = tsne.fit_transform(test_images_flattened)

# Plot the clustered data using t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(test_images_tsne[:, 0], test_images_tsne[:, 1], c=test_cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title("t-SNE Visualization of KMeans Clustering on Testing Data")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

