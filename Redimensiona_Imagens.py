import os
import cv2

def resize_images(source_dir, dest_dir, target_size=(1024, 1024)):
    """
    Resizes all images in source_dir to target_size and saves them in dest_dir.
    Maintains original file extensions and names.
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Supported image extensions
    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # Process each file in source directory
    for filename in os.listdir(source_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in supported_ext:
            try:
                # Read the image
                img_path = os.path.join(source_dir, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize the image
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    
                    # Save to destination
                    dest_path = os.path.join(dest_dir, filename)
                    cv2.imwrite(dest_path, resized_img)
                    
                    print(f"Resized {filename} to {target_size[0]}x{target_size[1]}")
                else:
                    print(f"Could not read image: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
source_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Fundos/01Backgrounds'
destination_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Fundos/SquareBackgrounds'

resize_images(source_directory, destination_directory)