# import os
# import cv2

# def convert_images_to_png(source_directory, destination_directory):
#     # Verifica se o diretório de destino existe, se não, cria-o
#     if not os.path.exists(destination_directory):
#         os.makedirs(destination_directory)
    
#     # Itera sobre todos os arquivos no diretório de origem
#     for filename in os.listdir(source_directory):
#         # Obtém a extensão do arquivo
#         file_ext = os.path.splitext(filename)[1].lower()
        
#         # Verifica se o arquivo é uma imagem e não é PNG
#         if file_ext in ['.jpg', '.jpeg', '.bmp', '.gif', '.tiff'] and file_ext != '.png':
#             # Lê a imagem usando OpenCV
#             img = cv2.imread(os.path.join(source_directory, filename))
            
#             # Converte a imagem para PNG
#             png_filename = os.path.splitext(filename)[0] + '.png'
#             cv2.imwrite(os.path.join(destination_directory, png_filename), img)
            
#             print(f"Converted {filename} to {png_filename}")

# # Especifica os diretórios de origem e destino
# source_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/ArvoresIndividuais'
# destination_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/Individuais_PNG'

# # Converte as imagens para PNG
# convert_images_to_png(source_directory, destination_directory)

import os
import cv2
import shutil

def convert_jpg_to_jpeg(source_directory, destination_directory):
    # Verifica se o diretório de destino existe, se não, cria-o
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Itera sobre todos os arquivos no diretório de origem
    for filename in os.listdir(source_directory):
        # Obtém a extensão do arquivo
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Verifica se o arquivo é JPG ou JPEG
        if file_ext in ['.jpg', '.jpeg']:
            # Se for JPG, converte para JPEG
            if file_ext == '.jpg':
                # Lê a imagem usando OpenCV
                img = cv2.imread(os.path.join(source_directory, filename))
                
                # Converte a imagem para JPEG
                jpeg_filename = os.path.splitext(filename)[0] + '.jpeg'
                cv2.imwrite(os.path.join(destination_directory, jpeg_filename), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                print(f"Converted {filename} to {jpeg_filename}")
            # Se já for JPEG, apenas copia para o diretório de destino
            elif file_ext == '.jpeg':
                shutil.copy2(
                    os.path.join(source_directory, filename),
                    os.path.join(destination_directory, filename)
                )
                print(f"Copied {filename} to destination directory")

# Especifica os diretórios de origem e destino
source_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/Backgrounds'
destination_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/01Backgrounds'

# Executa a conversão
convert_jpg_to_jpeg(source_directory, destination_directory)
