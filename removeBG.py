import cv2
import numpy as np
import os
from pathlib import Path

def remove_background_and_save(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.png'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(img)

        final_image = cv2.merge([b, g, r, alpha])
        cv2.imwrite(output_path, final_image)
        
 
# Exemplo de uso:
input_directory =  "/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/brancoTeste"
output_directory = "/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_teste"

remove_background_and_save(input_directory, output_directory)
