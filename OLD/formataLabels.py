import os

def processar_arquivos(pasta_origem, pasta_destino):
    # Verifica se a pasta de destino existe, se não, cria
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    # Percorre todos os arquivos na pasta de origem
    for nome_arquivo in os.listdir(pasta_origem):
        if nome_arquivo.endswith('.txt'):
            caminho_origem = os.path.join(pasta_origem, nome_arquivo)
            caminho_destino = os.path.join(pasta_destino, nome_arquivo)
            
            # Lê o arquivo original e processa cada linha
            with open(caminho_origem, 'r') as arquivo_origem:
                linhas = arquivo_origem.readlines()
            
            # Adiciona '0 ' no início de cada linha
            linhas_processadas = []
            for linha in linhas:
                linha_processada = '0 ' + linha
                linhas_processadas.append(linha_processada)
            
            # Salva o arquivo modificado na pasta de destino
            with open(caminho_destino, 'w') as arquivo_destino:
                arquivo_destino.writelines(linhas_processadas)
            
            print(f'Arquivo processado: {nome_arquivo}')

# Exemplo de uso
pasta_origem =  '/Users/iagocampista/Documents/Projects/Tree_Neural_Network/Dataset'  # Substitua pelo caminho real
pasta_destino = '/Users/iagocampista/Documents/Projects/Tre_Neural_Network/Labels'  # Substitua pelo caminho real

processar_arquivos(pasta_origem, pasta_destino)