import os
import random
import cv2
import numpy as np

def get_random_images(directory, num_images, used_images):
    all_images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    available_images = list(set(all_images) - set(used_images))
    selected_images = random.sample(available_images, min(num_images, len(available_images)))
    return selected_images

def paste_random_trees(base_image_path, random_tree_paths, output_path):
    # Lê a imagem base (sempre em 3 canais)
    base_img = cv2.imread(base_image_path)
    if base_img is None:
        print(f"Could not read base image: {base_image_path}")
        return
    
    # Converte para RGBA (adiciona canal alfa opaco)
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)
    
    base_height, base_width = base_img.shape[:2]

    # Cria o caminho para a subpasta de labels
    labels_dir = os.path.join(os.path.dirname(output_path), "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # Cria o nome do arquivo de texto na subpasta labels
    txt_filename = os.path.splitext(os.path.basename(output_path))[0] + '.txt'
    txt_path = os.path.join(labels_dir, txt_filename)

    with open(txt_path, 'w') as txt_file:
        for tree_path in random_tree_paths:
            # Lê a imagem da árvore com transparência (4 canais)
            tree = cv2.imread(tree_path, cv2.IMREAD_UNCHANGED)
            if tree is None:
                print(f"Could not read tree image: {tree_path}")
                continue
                
            # Se a imagem tem apenas 3 canais, converte para 4 (adiciona canal alfa opaco)
            if tree.shape[2] == 3:
                tree = cv2.cvtColor(tree, cv2.COLOR_BGR2BGRA)
            
            tree_height, tree_width = tree.shape[:2]

            # Redimensiona se necessário (mantendo proporção)
            if tree_height > base_height/2 or tree_width > base_width/2:
                scale_factor = 1/7
                new_height = int(base_height * scale_factor)
                new_width = int((tree_width / tree_height) * new_height)
                tree = cv2.resize(tree, (new_width, new_height), interpolation=cv2.INTER_AREA)
                tree_height, tree_width = tree.shape[:2]

            # Gera posições aleatórias
            x_offset = random.randint(0, base_width - tree_width)
            y_offset = random.randint(0, base_height - tree_height)

            # Calcula as informações da árvore
            center_x = x_offset + tree_width // 2
            center_y = y_offset + tree_height // 2
            width = tree_width
            height = tree_height
            
            # Escreve as informações no arquivo de texto
            txt_file.write(f"0 {center_x} {center_y} {width} {height}\n")

            # Extrai o canal alfa e cria máscara
            alpha_channel = tree[:, :, 3] / 255.0
            inverse_alpha = 1.0 - alpha_channel
            
            # Para cada canal de cor (BGR)
            for c in range(0, 3):
                base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, c] = (
                    alpha_channel * tree[:, :, c] + 
                    inverse_alpha * base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, c]
                )

    # Remove o canal alfa antes de salvar (se quiser manter como JPG)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)
    
    # Salva a imagem resultante
    cv2.imwrite(output_path, base_img)

def process_images(source_directory, tree_directory, destination_directory):
    used_tree_list = os.path.join(destination_directory, 'used_random_trees.txt')
    
    if os.path.exists(used_tree_list):
        with open(used_tree_list, 'r') as f:
            used_images = f.read().splitlines()
    else:
        used_images = []

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    for background_image in os.listdir(source_directory):
        file_ext = os.path.splitext(background_image)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            source_path = os.path.join(source_directory, background_image)

            # Create three variations for each background image
            for i in range(1, 4):
                destination_path = os.path.join(destination_directory, 
                                             f"{os.path.splitext(background_image)[0]}_{i}{file_ext}")
                
                random_trees = get_random_images(tree_directory, 3, used_images)
                random_tree_paths = [os.path.join(tree_directory, img) for img in random_trees]
                
                paste_random_trees(source_path, random_tree_paths, destination_path)

                # Armazena apenas os nomes dos arquivos, não os caminhos completos
                used_images.extend(random_trees)
                
                with open(used_tree_list, 'w') as f:
                    for item in used_images:
                        f.write("%s\n" % item)
            
            print(f"Processed {background_image}")

# Diretórios
source_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Fundos/SquareBackgrounds'
tree_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_Transparente'
destination_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Dataset1'

process_images(source_directory, tree_directory, destination_directory)