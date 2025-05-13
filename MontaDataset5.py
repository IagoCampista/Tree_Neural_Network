import os
import random
import cv2
import numpy as np
import math 

def get_random_images(tree_directory, num_images):

    #Pega todas as imagens de arvores validas dentro da pasta. 
    all_tree_images = [f for f in os.listdir(tree_directory)
                             if os.path.isfile(os.path.join(tree_directory, f))
                             and os.path.splitext(f)[1].lower() in ['.png']]
    
    #Confere se depois da selecao das imagens validas, a lista nao ficou vazia. Indicando que nenhuma imagem valida passou pelo criterio de selecao
    if not all_tree_images:
        print(f"Nenhuma imagem de arvore valida encontrada na {tree_directory}")
        return
    
    #Embaralha as imagens de arvore
    random.shuffle(all_tree_images)

    selected_images = random.sample(all_tree_images, num_images)
    return selected_images

def paste_random_trees(base_image_path, random_tree_paths, output_image_path, labels_dir):
    # Lê a imagem base (sempre em 3 canais para iniciar)
    base_img = cv2.imread(base_image_path)
    if base_img is None:
        print(f"Não foi possivel ler a imagem: {base_image_path}")
        return

    # Converte para BGRA para facilitar a colagem com alfa
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)

    base_height, base_width = base_img.shape[:2]

    # Cria o nome do arquivo de texto na subpasta labels
    txt_filename = os.path.splitext(os.path.basename(output_image_path))[0] + '.txt'
    txt_path = os.path.join(labels_dir, txt_filename)

    with open(txt_path, 'w') as txt_file:
        for tree_path in random_tree_paths:
            # Lê a imagem da árvore com transparência (4 canais)
            # cv2.IMREAD_UNCHANGED -> imagem será lida com todos os seus canais originais, inclusive o canal alfa (transparência), se ele existir.
            tree = cv2.imread(tree_path, cv2.IMREAD_UNCHANGED)
            if tree is None:
                print(f"Imagem não conseguiu ser lida: {tree_path}")
                continue

            tree_height, tree_width = tree.shape[:2]

            # Redimensiona se necessário (mantendo proporção)
            max_dim_ratio = 1/2 # Maximo tamanho da arvore, neste caso 50% do tamanho da imagem de fundo
            
            if tree_height > base_height * max_dim_ratio or tree_width > base_width * max_dim_ratio:
                scale_factor = 1/7
                new_height = int(base_height * scale_factor)
                new_width = int((tree_width / tree_height) * new_height)

                tree = cv2.resize(tree, (new_width, new_height), interpolation=cv2.INTER_AREA)
                tree_height, tree_width = tree.shape[:2]

            x_offset = random.randint(0, base_width - tree_width)
            y_offset = random.randint(0, base_height - tree_height)

            # Extrai o canal alfa e cria máscara
            # percorre a imagem e seleciona apenas o canal alfa e normaliza entre 0 e 1 (0 transparente e 1 opaco)
            # depois cria o inverso do alfa (a opacidade)
            alpha_channel = tree[:, :, 3] / 255.0
            inverse_alpha = 1.0 - alpha_channel

            # Aplica a colagem usando o canal alfa
            for current_channel in range(0, 3): # Para cada canal de cor (BGR)
                # Calcula a região de interesse na imagem base
                roi = base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, current_channel]

                # Aplica a fórmula de blending: foreground * alpha + background * (1 - alpha)
                base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, current_channel] = (alpha_channel * tree[:, :, current_channel] + inverse_alpha * roi)

            # Calcula as informações da árvore para o formato YOLO
            # Coordenadas do centro normalizadas (0 a 1)
            center_x = (x_offset + tree_width / 2) / base_width
            center_y = (y_offset + tree_height / 2) / base_height
            # Largura e altura normalizadas (0 a 1)
            width = tree_width / base_width
            height = tree_height / base_height

            # Escreve as informações no arquivo de texto (formato YOLO: class_id center_x center_y width height)
            # Assumindo class_id 0 para "tree"
            txt_file.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n") # Usando .6f para 6 casas decimais

    # Remove o canal alfa antes de salvar (para salvar como JPEG)
    final_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)

    # Salva a imagem resultante
    cv2.imwrite(output_image_path, final_img)

def process_images(source_directory, tree_directory, destination_directory):

    #Pega todas as imagens de fundo validas dentro da pasta. / Previne que arquivos como .DSSTORE sejam levados em consideracao e quebre o código
    all_background_images = [f for f in os.listdir(source_directory)
                             if os.path.isfile(os.path.join(source_directory, f))
                             and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    
    #Confere se depois da selecao das imagens validas, a lista nao ficou vazia. Indicando que nenhuma imagem valida passou pelo criterio de selecao
    if not all_background_images:
        print(f"Nenhuma imagem de fundo valida encontrada na {source_directory}")
        return

    # Define as pastas de saída
    train_images_dir = os.path.join(destination_directory, 'train', 'images')
    train_labels_dir = os.path.join(destination_directory, 'train', 'labels')
    val_images_dir = os.path.join(destination_directory, 'val', 'images')
    val_labels_dir = os.path.join(destination_directory, 'val', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # --- Imagens de Treinamento ---
    print("\n--- Montando Imagens de Treinamento ---")
    for i, background_image in enumerate(all_background_images):
        print(f"Montando imagem: {background_image} -- {i+1}/{len(all_background_images)}")

        #Pega o caminho da imagem de fundo atual, para saber referenciala e pega-la no sistema de arquivo
        background_path = os.path.join(source_directory, background_image)

        #Separa o nome da imagem da sua extensao
        base_name = os.path.splitext(background_image)[0]
        file_ext = os.path.splitext(background_image)[1].lower()
        
        # Cria 3 variações para cada imagem de fundo
        for j in range(1, 4):
            output_filename = f"{base_name}_{j}{file_ext}"
            destination_path = os.path.join(train_images_dir, output_filename)

            random_trees = get_random_images(tree_directory, 3)
            random_tree_paths = [os.path.join(tree_directory, img) for img in random_trees]

            paste_random_trees(background_path, random_tree_paths, destination_path, train_labels_dir)
    
    #Pega todas as imagens validas geradas acima. / Previne que arquivos como .DSSTORE sejam levados em consideracao e quebre o código
    print("train_images_dir:  "+train_images_dir)
    all_training_images = [f for f in os.listdir(train_images_dir)
                             if os.path.isfile(os.path.join(train_images_dir, f))
                             and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    
    #Confere se depois da selecao das imagens validas, a lista nao ficou vazia. Indicando que nenhuma imagem valida passou pelo criterio de selecao
    if not all_training_images:
        print(f"Nenhuma imagem de fundo valida encontrada na {train_images_dir}")
        return
    
    #Embaralha as imagens 
    random.shuffle(all_training_images)
 
    #Separa as imagens em dois grupos: treinamento e validação, na proporção 75/25%
    num_val_images = math.floor(len(all_training_images) * 0.25)

    split_point = len(all_training_images) - num_val_images

    train_images = all_training_images[:split_point]
    val_images = all_training_images[split_point:]

    print(f"Total background images found: {len(all_training_images)}")
    print(f"Training background images: {len(train_images)}")
    print(f"Validation background images: {len(val_images)}")


    print("\n--- Separando as Imagens de Validação ---")
    for i, val_image in enumerate(val_images):
        print(f"Movendo imagem: {val_image} -- {i+1}/{len(val_images)}")

        # --- Caminho do arquivo de imagem na pasta de treino (origem) ---
        source_image_path = os.path.join(train_images_dir, val_image)
        print("Origem imagem: " + source_image_path)

        # --- Caminho do arquivo de imagem na pasta de validação (destino) ---
        destination_image_path = os.path.join(val_images_dir, val_image)
        print("Destino imagem: " + destination_image_path)

        # --- Caminho do arquivo de label correspondente na pasta de treino (origem) ---
        label_filename = os.path.splitext(val_image)[0] + '.txt'
        source_label_path = os.path.join(train_labels_dir, label_filename)
        print("Origem label: " + source_label_path)

        # --- Caminho do arquivo de label correspondente na pasta de validação (destino) ---
        destination_label_path = os.path.join(val_labels_dir, label_filename)
        print("Destino label: " + destination_label_path)

        # Originalmente, os.rename() serve para renomear arquivos ou diretórios. 
        # No entanto, se o novo caminho especificado (dst) incluir um diretório diferente do original (src), 
        # o efeito é que o arquivo (ou diretório) é movido para esse novo local.

        # Mover a imagem usando os.rename()
        os.rename(source_image_path, destination_image_path)
        # Mover o label correspondente usando os.rename()
        os.rename(source_label_path, destination_label_path)



# Diretórios
source_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Fundos/FittedBackgrounds'
tree_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG_Transparente'
destination_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/TreeDataset'

# Chame a função principal para iniciar o processo
process_images(source_directory, tree_directory, destination_directory)