import cv2
import numpy as np

def count_white_pixels(image_path):
    # Carrega a imagem (em formato BGR)
    img = cv2.imread(image_path)
    
    if img is None:
        print("Erro: Imagem não encontrada ou formato inválido.")
        return 0
    
    # Define o valor "branco puro" em BGR (OpenCV usa BGR em vez de RGB)
    white_bgr = [255, 255, 255]
    
    # Cria uma máscara onde pixels brancos = 1 (True), outros = 0 (False)
    white_pixels_mask = np.all(img == white_bgr, axis=-1)
    
    # Conta quantos pixels são brancos
    white_pixel_count = np.sum(white_pixels_mask)
    
    return white_pixel_count

# Exemplo de uso:
image_path = "/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG_Transparente/0_arvores28.png"  # Substitua pelo caminho da sua imagem
total_white = count_white_pixels(image_path)

print(f"A imagem tem {total_white} pixels brancos (RGB 255,255,255)")