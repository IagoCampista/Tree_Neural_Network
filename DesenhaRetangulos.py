import os
import cv2
import random

def draw_bounding_boxes(image_dir, output_dir):
    # Verifica se o diretório de destino existe, se não, cria-o
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Cores diferentes para cada árvore (no formato BGR)
    colors = [
        (0, 0, 255),    # Vermelho
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 255, 255),  # Amarelo
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Ciano
    ]
    
    # Percorre todas as imagens no diretório
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Caminhos dos arquivos
            image_path = os.path.join(image_dir, filename)
            txt_path = os.path.join(image_dir, os.path.splitext(filename)[0] + '.txt')
            
            # Verifica se o arquivo de anotação existe
            if not os.path.exists(txt_path):
                print(f"Arquivo de anotação não encontrado para {filename}")
                continue
            
            # Carrega a imagem
            image = cv2.imread(image_path)
            if image is None:
                print(f"Não foi possível carregar a imagem {filename}")
                continue
            
            # Lê as anotações do arquivo txt
            with open(txt_path, 'r') as f:
                annotations = f.readlines()
            
            # Desenha os retângulos para cada árvore
            for i, annotation in enumerate(annotations):
                try:
                    center_x, center_y, width, height = map(int, annotation.strip().split())
                    
                    # Calcula as coordenadas do retângulo
                    x1 = center_x - width // 2
                    y1 = center_y - height // 2
                    x2 = center_x + width // 2
                    y2 = center_y + height // 2
                    
                    # Escolhe uma cor (cíclica se tiver mais árvores que cores)
                    color = colors[i % len(colors)]
                    
                    # Desenha o retângulo
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Adiciona um texto com o número da árvore
                    cv2.putText(image, f"Tree {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                except ValueError:
                    print(f"Formato inválido no arquivo {txt_path}, linha {i+1}")
                    continue
            
            # Salva a imagem com os retângulos
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            print(f"Processada {filename} com {len(annotations)} árvores marcadas")

# Diretórios de entrada e saída
image_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/Dataset'
output_directory = '/Users/iagocampista/Documents/Projects/Image Neural Network Train/Dataset_With_Boxes'

# Executa a função
draw_bounding_boxes(image_directory, output_directory)