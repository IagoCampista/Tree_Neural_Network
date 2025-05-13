import cv2
import numpy as np
import os
from pathlib import Path

def remove_background_and_save(input_dir, output_dir, threshold=150):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.png'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img.shape[2] == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        
        # Método aprimorado (considera pixels onde TODOS os canais são ≥ threshold)
        rgb = img_rgba[:, :, :3]
        white_mask = np.all(rgb >= threshold, axis=-1)
        
        # Aplica transparência + remove pequenos resíduos com morfologia
        img_rgba[:, :, 3] = np.where(white_mask, 0, img_rgba[:, :, 3])
        
        # Opcional: remove pixels isolados (descomente se necessário)
        from skimage import morphology
        alpha_channel = img_rgba[:, :, 3]
        alpha_channel = morphology.remove_small_objects(alpha_channel.astype(bool), min_size=50)
        img_rgba[:, :, 3] = alpha_channel.astype(np.uint8) * 255
        
        cv2.imwrite(output_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
        print(f"Processed: {filename} (Threshold={threshold})")
# Exemplo de uso:
input_directory =  "/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG"
output_directory = "/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG_Transparente"

remove_background_and_save(input_directory, output_directory)
