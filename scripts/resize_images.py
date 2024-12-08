import numpy as np
from torchvision.transforms import Resize, ToTensor, ToPILImage
from PIL import Image
import os
import cv2
import glob



folder = "/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform/camera_thunder_wrist_resized"
files = glob.glob(os.path.join(folder, "*.npy"))
files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
for file in files:
    image = np.load(os.path.join(folder, file))
    print(image.shape)

cv2.destroyAllWindows()



# def resize_npy_images(input_folder, output_folder, size=256):
#     """
#     Resize all .npy images in the input folder to the specified size and save them in the output folder with the same names.

#     Args:
#         input_folder (str): Path to the folder containing original .npy images.
#         output_folder (str): Path to the folder where resized .npy images will be saved.
#         size (int): The size to resize the smaller side of the images.
#     """
#     # Create output folder if it does not exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Define the resize transformation
#     resize_transform = Resize(size)
    
#     # Iterate through all .npy files in the input folder
#     for file_name in os.listdir(input_folder):
#         input_file_path = os.path.join(input_folder, file_name)
        
#         # Check if the file is a .npy file
#         if file_name.lower().endswith('.npy'):
#             try:
#                 # Load the .npy file
#                 npy_array = np.load(input_file_path)

#                 # Ensure the image is in (H, W, C) format (if not, modify accordingly)
#                 if len(npy_array.shape) == 2:  # Grayscale image (H, W)
#                     npy_array = np.expand_dims(npy_array, axis=-1)  # Add channel dimension
#                 elif len(npy_array.shape) == 3 and npy_array.shape[0] in [1, 3]:  # (C, H, W) format
#                     npy_array = np.transpose(npy_array, (1, 2, 0))  # Convert to (H, W, C)

#                 # Convert to PIL Image for resizing
#                 img = Image.fromarray(npy_array.astype(np.uint8))
#                 resized_img = resize_transform(img)

#                 # Convert back to NumPy array
#                 resized_npy_array = np.array(resized_img)

#                 # Save the resized array as a .npy file in the output folder
#                 output_file_path = os.path.join(output_folder, file_name)
#                 np.save(output_file_path, resized_npy_array)
#                 print(f"Resized and saved: {file_name}")
#             except Exception as e:
#                 print(f"Failed to process {file_name}: {e}")

# # Folder paths
# input_folder = "/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform/camera_thunder_wrist"  # Path to the folder containing original .npy files
# output_folder = "/home/rpmdt05/Code/the-real-bartender/dataset/transformed_data/initial_transform/camera_thunder_wrist_resized"  # Path to save resized .npy files

# # Resize all .npy files in the input folder and save to the output folder
# resize_npy_images(input_folder, output_folder, size=256)


