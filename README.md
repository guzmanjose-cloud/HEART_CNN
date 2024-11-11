Heart Disease Detection with CNN
Overview

This project aims to classify heart disease types using a Convolutional Neural Network (CNN) model. The model is trained on a dataset of medical heart images to predict the presence of specific heart conditions, including coronary artery disease, angina, and other heart-related ailments. The project includes data preprocessing, feature extraction, model training, and deployment in a web application.
Project Structure

    src/: Contains the main code files.
        heart_cnn_app.py: Main application file for serving the model in a web app.
        heart_cnn_test.py: Script for testing the CNN model and visualizing clustering results.
        heart_cnn_directory_creator.py: Utility script for organizing data directories.
        heart_disease_model.h5: Pre-trained CNN model.
        kmeans_model.pkl: KMeans model used for clustering image data.
    templates/: HTML templates for the web app interface.
    requirements.txt: Lists the dependencies required to run the project.

Dataset

The dataset used in this project contains heart images divided into categories based on heart disease types:

    coronary artery disease
    angina disease
    cardio disease
    hypotension disease

Images are in JPEG, JPG, and PNG formats. The project includes methods for handling these formats, especially optimizing CNN performance for PNG files.
Models

    Convolutional Neural Network (CNN): The primary model used for image classification.
    KMeans Clustering: Used as a preprocessing step to cluster images, providing labels that improve CNN training.

Features

    Image Preprocessing: Handles various image formats and preprocesses them for input into the CNN model.
    Dimensionality Reduction: Utilizes PCA and t-SNE for visualizing image clusters.
    Web Application: A Flask/Render-based web app to allow users to upload images and receive heart disease predictions.
    Deployment: The application is deployed using Render with Gunicorn as the server.

Setup and Installation

    1. Clone the repository:

    2. git clone https://github.com/your-username/Heart_CNN.git

    3. Navigate to the project directory:

    4. cd Heart_CNN

    5. Install the required packages:

    6. pip install -r requirements.txt

    7. Run the web application:

    8. gunicorn src.heart_cnn_app:app

Usage

    Open the deployed web application.
    Upload a heart image.
    The model will predict the type of heart disease based on the uploaded image and display the result.

Results

    The CNN model is trained to classify images into specific heart disease types, achieving high accuracy on JPEG files and progressively improved accuracy on PNG files.
    Clustering visualizations are generated using PCA and t-SNE to show distinct clusters in the data.

Future Work

    Improve the model's performance on PNG images.
    Integrate additional heart disease types and expand the dataset.
    Optimize the web application interface for better user experience.
