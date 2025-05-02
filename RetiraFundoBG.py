from rembg import remove
import cv2

input_image = cv2.imread('/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_teste/0_arvores16.png')
output_image = remove(input_image)  # Remove fundo automaticamente
cv2.imwrite("/Users/iagocampista/Documents/Projects/Tree_Neural_Network/ImagensArvores/Individuais_PNG_Transparente/0_arvores7.png", output_image)