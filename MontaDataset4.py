import os
import random
import cv2
import numpy as np
import math # Import math for floor calculation

def get_random_images(directory, num_images, used_images):
    """
    Gets a list of random image filenames from a directory, excluding those already used.

    Args:
        directory (str): The path to the directory containing images.
        num_images (int): The number of random images to select.
        used_images (list): A list of filenames that have already been used.

    Returns:
        list: A list of selected image filenames.
    """
    all_images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    available_images = list(set(all_images) - set(used_images))
    # Ensure we don't request more images than available
    selected_images = random.sample(available_images, min(num_images, len(available_images)))
    return selected_images

def paste_random_trees(base_image_path, random_tree_paths, output_image_path):
    """
    Pastes random tree images onto a base image and generates YOLO format labels.

    Args:
        base_image_path (str): The path to the background image.
        random_tree_paths (list): A list of paths to the tree images to paste.
        output_image_path (str): The desired path to save the resulting image.
    """
    # Lê a imagem base (sempre em 3 canais para iniciar)
    base_img = cv2.imread(base_image_path)
    if base_img is None:
        print(f"Could not read base image: {base_image_path}")
        return

    # Converte para BGRA para facilitar a colagem com alfa
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)

    base_height, base_width = base_img.shape[:2]

    # Cria o caminho para a subpasta de labels
    # A subpasta labels deve estar no mesmo nível do diretório 'images'
    # Ex: se output_image_path é /path/to/dataset/train/images/img.jpg,
    # labels_dir será /path/to/dataset/train/labels
    output_dir = os.path.dirname(output_image_path) # e.g., /path/to/dataset/train/images
    parent_dir = os.path.dirname(output_dir)      # e.g., /path/to/dataset/train
    labels_dir = os.path.join(parent_dir, "labels") # e.g., /path/to/dataset/train/labels

    os.makedirs(labels_dir, exist_ok=True)

    # Cria o nome do arquivo de texto na subpasta labels
    txt_filename = os.path.splitext(os.path.basename(output_image_path))[0] + '.txt'
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
                # Adiciona canal alfa opaco (todos os pixels opacos)
                b, g, r = cv2.split(tree)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                tree = cv2.merge((b, g, r, alpha))

            tree_height, tree_width = tree.shape[:2]

            # Redimensiona se necessário (mantendo proporção)
            max_dim_ratio = 1/2 # Max tree size is 50% of base image dimension
            if tree_height > base_height * max_dim_ratio or tree_width > base_width * max_dim_ratio:
                 # Calculate scale factors based on max dimension
                height_scale = (base_height * max_dim_ratio) / tree_height
                width_scale = (base_width * max_dim_ratio) / tree_width
                # Use the smaller scale factor to maintain aspect ratio and fit within limits
                scale_factor = min(height_scale, width_scale)

                new_height = int(tree_height * scale_factor)
                new_width = int(tree_width * scale_factor)

                # Ensure dimensions are at least 1 pixel
                new_height = max(1, new_height)
                new_width = max(1, new_width)

                tree = cv2.resize(tree, (new_width, new_height), interpolation=cv2.INTER_AREA)
                tree_height, tree_width = tree.shape[:2]

            # Gera posições aleatórias que garantam que a árvore caiba na imagem base
            if base_width - tree_width <= 0 or base_height - tree_height <= 0:
                 print(f"Tree {tree_path} is too large to fit in base image {base_image_path} after scaling. Skipping.")
                 continue # Skip this tree if it still doesn't fit

            x_offset = random.randint(0, base_width - tree_width)
            y_offset = random.randint(0, base_height - tree_height)

            # Extrai o canal alfa e cria máscara
            alpha_channel = tree[:, :, 3] / 255.0
            inverse_alpha = 1.0 - alpha_channel

            # Aplica a colagem usando o canal alfa
            for c in range(0, 3): # Para cada canal de cor (BGR)
                # Calcula a região de interesse na imagem base
                roi = base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, c]
                # Aplica a fórmula de blending: foreground * alpha + background * (1 - alpha)
                base_img[y_offset:y_offset+tree_height, x_offset:x_offset+tree_width, c] = (
                    alpha_channel * tree[:, :, c] +
                    inverse_alpha * roi
                )

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

    # Remove o canal alfa antes de salvar (para salvar como JPG ou PNG sem alfa)
    # Se quiser salvar com transparência, use cv2.imwrite com canal alfa
    final_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)

    # Salva a imagem resultante
    cv2.imwrite(output_image_path, final_img)


def process_images(source_directory, tree_directory, destination_directory, val_split_ratio=0.25):
    """
    Processes background images, pastes random trees, generates labels,
    and splits the output into train/val directories.

    Args:
        source_directory (str): Path to the directory with base background images.
        tree_directory (str): Path to the directory with transparent tree images.
        destination_directory (str): Path where the 'train' and 'val' directories will be created.
        val_split_ratio (float): The ratio of data to be used for validation (e.g., 0.25 for 25%).
    """
    used_tree_list_path = os.path.join(destination_directory, 'used_random_trees.txt')

    if os.path.exists(used_tree_list_path):
        with open(used_tree_list_path, 'r') as f:
            used_images = [line.strip() for line in f if line.strip()] # Read and clean up lines
    else:
        used_images = []

    # Get all valid background images
    all_background_images = [f for f in os.listdir(source_directory)
                             if os.path.isfile(os.path.join(source_directory, f))
                             and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if not all_background_images:
        print(f"No valid background images found in {source_directory}")
        return

    # Shuffle background images
    random.shuffle(all_background_images)

    # Calculate split point
    num_val_images = math.floor(len(all_background_images) * val_split_ratio)
    if num_val_images == 0 and len(all_background_images) > 0:
         num_val_images = 1 # Ensure at least one image for validation if possible

    split_point = len(all_background_images) - num_val_images

    train_backgrounds = all_background_images[:split_point]
    val_backgrounds = all_background_images[split_point:]

    print(f"Total background images found: {len(all_background_images)}")
    print(f"Training background images: {len(train_backgrounds)}")
    print(f"Validation background images: {len(val_backgrounds)}")

    # Define output directories
    train_images_dir = os.path.join(destination_directory, 'train', 'images')
    train_labels_dir = os.path.join(destination_directory, 'train', 'labels')
    val_images_dir = os.path.join(destination_directory, 'val', 'images')
    val_labels_dir = os.path.join(destination_directory, 'val', 'labels')

    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # --- Process Training Images ---
    print("\n--- Processing Training Images ---")
    for i, background_image in enumerate(train_backgrounds):
        print(f"Processing train background {i+1}/{len(train_backgrounds)}: {background_image}")
        source_path = os.path.join(source_directory, background_image)
        file_ext = os.path.splitext(background_image)[1].lower()
        base_name = os.path.splitext(background_image)[0]

        # Create three variations for each background image
        for j in range(1, 4):
            output_filename = f"{base_name}_{j}{file_ext}"
            destination_path = os.path.join(train_images_dir, output_filename)

            random_trees = get_random_images(tree_directory, 3, used_images)
            random_tree_paths = [os.path.join(tree_directory, img) for img in random_trees]

            paste_random_trees(source_path, random_tree_paths, destination_path)

            # Armazena apenas os nomes dos arquivos das árvores usadas
            used_images.extend(random_trees)

    # --- Process Validation Images ---
    print("\n--- Processing Validation Images ---")
    for i, background_image in enumerate(val_backgrounds):
        print(f"Processing val background {i+1}/{len(val_backgrounds)}: {background_image}")
        source_path = os.path.join(source_directory, background_image)
        file_ext = os.path.splitext(background_image)[1].lower()
        base_name = os.path.splitext(background_image)[0]

        # Create three variations for each background image
        for j in range(1, 4):
            output_filename = f"{base_name}_{j}{file_ext}"
            destination_path = os.path.join(val_images_dir, output_filename)

            # Note: We are using the same 'used_images' list for both train and val
            # to ensure unique tree images are picked across the entire dataset generation run.
            random_trees = get_random_images(tree_directory, 3, used_images)
            random_tree_paths = [os.path.join(tree_directory, img) for img in random_trees]

            paste_random_trees(source_path, random_tree_paths, destination_path)

            # Armazena apenas os nomes dos arquivos das árvores usadas
            used_images.extend(random_trees)

    # Save the updated list of used tree images
    # Convert to set and then list to remove duplicates before saving
    used_images_unique = list(set(used_images))
    with open(used_tree_list_path, 'w') as f:
        for item in used_images_unique:
            f.write(f"{item}\n")

    print("\n--- Processing Complete ---")
    print(f"Generated training data in: {train_images_dir} and {train_labels_dir}")
    print(f"Generated validation data in: {val_images_dir} and {val_labels_dir}")
    print(f"Used tree list saved to: {used_tree_list_path}")


# Diretórios
source_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Fundos/FittedBackgrounds'
tree_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG_Transparente'
destination_directory = '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Dataset2' # This will contain 'train' and 'val' subfolders

# Chame a função principal para iniciar o processo
process_images(source_directory, tree_directory, destination_directory, val_split_ratio=0.25)