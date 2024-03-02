import cv2
import numpy as np
import os

# Assuming the labels are color-coded RGB images
def create_class_mask(image_path, color):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Define color lower and upper bounds
    lower = np.array(color, dtype = "uint8")
    upper = np.array(color, dtype = "uint8")
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    return mask

# Folder path for your labeled images
labeled_folder = '/home/azuo/LRHPerception/datasets/kitti/data_semantics/training/semantic_rgb'
# Output folder path for the created masks
output_folder = '/home/azuo/LRHPerception/datasets/kitti/data_semantics/training/road_gt'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the color for the class of interest. 
# For example, if cars are represented by red, you'd use [255, 0, 0].
color_of_interest = [128, 64, 128]

for file in os.listdir(labeled_folder):
    if file.endswith(".png"):  # or ".jpg" or whatever your image format is
        image_path = os.path.join(labeled_folder, file)
        mask = create_class_mask(image_path, color_of_interest)
        print(mask)
        # Save the mask image
        cv2.imwrite(os.path.join(output_folder, file), mask)
