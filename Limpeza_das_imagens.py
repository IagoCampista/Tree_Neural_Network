import os
import cv2

def browse_and_manage_images(directory):
    # Arquivo para armazenar imagens analisadas
    analyzed_file = 'analyzed_images.txt'
    
    # Carrega imagens analisadas do arquivo, se existir
    if os.path.exists(analyzed_file):
        with open(analyzed_file, 'r') as f:
            analyzed_images = f.read().splitlines()
    else:
        analyzed_images = []

    # Obtém a lista de arquivos no diretório que ainda não foram analisados
    files = [f for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f)) 
            and f.lower().endswith('.png') 
            and f not in analyzed_images]

    index = 0
    print('qtd de imagens restantes', len(files))

    while index < len(files):
        filename = files[index]
        filepath = os.path.join(directory, filename)
        
        # Lê a imagem usando OpenCV
        img = cv2.imread(filepath)
        
        # Exibe a imagem
        cv2.imshow('Image Viewer', img)
        
        # Espera por uma tecla ser pressionada
        key = cv2.waitKey(0)
        
        if key == 27:  # Tecla ESC para sair
            break
        elif key == 100:  # Letra D (apaga a imagem)
            os.remove(filepath)
            print(f"Deleted {filename}")
        else:  # Qualquer outra tecla (mantém a imagem)
            print(f"Kept {filename}")
        
        # Marca a imagem como analisada
        analyzed_images.append(filename)
        
        # Salva a lista de imagens analisadas no arquivo
        with open(analyzed_file, 'w') as f:
            for item in analyzed_images:
                f.write("%s\n" % item)
        
        # Passa para a próxima imagem
        index += 1
    
    cv2.destroyAllWindows()

# Especifica o diretório contendo as imagens
image_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/ImagensArvores/Individuais_Transparente'

# Navega e gerencia as imagens
browse_and_manage_images(image_directory)