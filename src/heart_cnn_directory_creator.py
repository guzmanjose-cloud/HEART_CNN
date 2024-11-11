import os
from sklearn.model_selection import train_test_split
import shutil

# Define the dataset path
dataset_path = '/Users/joseguzman/Desktop/heart_data'

# Define output directories for training, testing, and validation
output_dirs_path = {
    "train": os.path.join(dataset_path, "training"),
    "test": os.path.join(dataset_path, "testing"),
    "val": os.path.join(dataset_path, "validation")
}

# Create output directories if they don't exist
for path in output_dirs_path.values():
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Directory path {path} already exists.")

# Define the heart disease categories
heart_disease_categories = ["angina_disease", "cardio_vascular_disease_", "coronary_artery_disease", "hypotension_disease_"]
train_test_val_path = ["training", "testing", "validation"]

# Create category subdirectories for train, test, and validation
for train_test_val in train_test_val_path:
    train_test_val_dir = os.path.join(dataset_path, train_test_val)

    for category in heart_disease_categories:
        category_folder_path = os.path.join(train_test_val_dir, category)
        if not os.path.exists(category_folder_path):
            os.makedirs(category_folder_path)
        else:
            print(f"Directory path {category_folder_path} already exists.")

# Split and move files for each category
for category in heart_disease_categories:
    category_dataset_path = os.path.join(dataset_path, category)
    images = []

    # Debug: print the category dataset path being processed
    print(f"Looking for images in: {category_dataset_path}")

    # Check if the category directory exists and contains images
    if os.path.exists(category_dataset_path):
        for img in os.listdir(category_dataset_path):
            if img.lower().endswith(('jpeg', 'jpg', 'png',"jpe")):
                images.append(os.path.join(category_dataset_path, img))
    else:
        print(f"Category directory {category_dataset_path} does not exist.")
        continue

    # Debug: show the images found
    if images:
        print(f"Found {len(images)} images in {category_dataset_path}.")
    else:
        print(f"No images found in {category_dataset_path}.")
        continue

    # Split data into train, test, and validation sets
    train_val, test = train_test_split(images, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    # Copy files to the appropriate directories
    for file in train:
        shutil.copy(file, os.path.join(output_dirs_path['train'], category))
    for file in val:
        shutil.copy(file, os.path.join(output_dirs_path['val'], category))
    for file in test:
        shutil.copy(file, os.path.join(output_dirs_path['test'], category))

# Final debug: show that the images were processed
print("Image processing complete.")
