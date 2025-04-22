import os
import random
import cv2

def get_random_images(directory, num_images, used_images):
    all_images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    available_images = list(set(all_images) - set(used_images))
    selected_images = random.sample(available_images, num_images)
    return selected_images

def paste_random_trees(base_image_path, random_tree_paths, output_path):
    # Lê a imagem base
    base_img = cv2.imread(base_image_path)
    base_height, base_width = base_img.shape[:2]

    for tree_path in random_tree_paths:
        # Lê a imagem aleatória
        tree = cv2.imread(tree_path)
        tree_height, tree_width = tree.shape[:2]

        # Verifica se a imagem aleatória é maior que a imagem base
        if tree_height > base_height/2 or tree_width > base_width/2:
            # Calcula o novo tamanho mantendo a proporção
            scale_factor = 1/7
            new_height = int(base_height * scale_factor)
            new_width = int((tree_width / tree_height) * new_height)
            tree = cv2.resize(tree, (new_width, new_height))
            tree_height, tree_width = tree.shape[:2]

        # Gera posições aleatórias para colar a imagem
        x_offset = random.randint(0, base_width - tree_width)
        y_offset = random.randint(0, base_height - tree_height)

        # Cola a imagem aleatória na imagem base
        base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width] = tree

    # Salva a imagem resultante
    cv2.imwrite(output_path, base_img)


def process_images(source_directory, tree_directory, destination_directory):
    # Arquivo para armazenar imagens usadas
    used_tree_list = 'used_random_trees.txt'
    
    # Cria um vetor contendo o nome de todas as imagens de arvores já usadas
    if os.path.exists(used_tree_list):
        with open(used_tree_list, 'r') as f:
            used_images = f.read().splitlines()
    else:
        used_images = []

    # Verifica se o diretório de destino existe, se não, cria-o
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Itera sobre todos os arquivos no diretório de origem
    for background_image in os.listdir(source_directory):
        file_ext = os.path.splitext(background_image)[1].lower()
        
        # Verifica se o arquivo é uma imagem
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            source_path = os.path.join(source_directory, background_image)
            destination_path = os.path.join(destination_directory, background_image)
            
            # Obtém 3 arvores aleatórias do diretório de arvores
            random_tree_paths = [os.path.join(tree_directory, img) for img in get_random_images(tree_directory, 3, used_images)]
            
            # Cola as imagens de arvore aleatórias na imagem base e salva no diretório de destino
            paste_random_trees(source_path, random_tree_paths, destination_path)
            
            # Marca as imagens aleatórias das arvores como usadas
            used_images.extend(random_tree_paths)
            
            # Salva a lista de imagens usadas no arquivo
            with open(used_tree_list, 'w') as f:
                for item in used_images:
                    f.write("%s\n" % item)
            
            print(f"Processed {background_image}")

# Especifica os diretórios de origem e destino
source_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/FittedBackgroundsTeste'
tree_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/Individuais_Transparente'
destination_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/Dataset'

# Processa as imagens para colar imagens aleatórias e salvar o resultado
process_images(source_directory, tree_directory, destination_directory)
