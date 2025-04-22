import cv2
import numpy as np
import os
from pathlib import Path

def remove_background_and_save(input_dir, output_dir):
    # Garante que o diretório de saída exista
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Percorre todas as imagens no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Carrega a imagem (com canal alpha se existir)
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            # Se a imagem já tem transparência (4 canais: BGRA), converte para RGBA
            if img.shape[2] == 4:
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                # Se não tem alpha, converte para RGBA (adicionando canal alpha)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)

            # Define o fundo branco (RGB 255,255,255) ou quadriculado (padrão cinza claro)
            lower_white = np.array([240, 240, 240], dtype=np.uint8)  # Limiar para fundo branco/claro
            upper_white = np.array([255, 255, 255], dtype=np.uint8)

            # Cria uma máscara para o fundo branco
            rgb = img_rgba[:, :, :3]
            white_mask = cv2.inRange(rgb, lower_white, upper_white)

            # Remove o fundo (define alpha=0 onde a máscara é branca)
            img_rgba[:, :, 3] = np.where(white_mask == 255, 0, img_rgba[:, :, 3])

            # Salva a imagem com transparência
            cv2.imwrite(output_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
            print(f"Processed: {filename}")

# Exemplo de uso:
input_directory = "/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/Individuais_PNG"
output_directory = "/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/Individuais_Transparente"

remove_background_and_save(input_directory, output_directory)