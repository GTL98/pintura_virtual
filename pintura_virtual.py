################################################################################################
#                                                                                              #
#  Toda a lógica mais detalhada está presente no arquivo "Contador de Dedos.ipynb"             #
#                                                                                              #
#  Em caso de dúvidas, consultar a documentação:                                               #
#      - "Aula 1 - Rastramento de mão (Introdução).ipynb" no link abaixo.                      #
#                                                                                              #
#  GitHub: https://github.com/GTL98/curso-completo-de-visao-computacional-avancada-com-python  #
#                                                                                              #
################################################################################################

# Importar as bibliotecas
import os
import cv2
import time
import numpy as np
import rastreamento_mao as rm

# Definir o tamanho da tela
largura_tela = 1280
altura_tela = 720

# Configurar a cor inicial
cor = (255, 0, 255)

# Configurar a espesurra
espessura_pincel = 15
espessura_borracha = 50

# Configurar a posição inicial da linha OpenCV
x_anterior = 0
y_anterior = 0

# Imagem do menu de cores
caminho = 'imagens_prontas'
lista_imagens = os.listdir(caminho)
lista_fotos = []
for caminho_imagem in lista_imagens:
    foto = cv2.imread(f'{caminho}/{caminho_imagem}')
    lista_fotos.append(foto)
cabecalho = lista_fotos[0]

# Módulo DetectorMao
detector = rm.DetectorMao(max_maos=1, deteccao_confianca=0.85, rastreamento_confianca=0.85)

# Tela de desenho
tela_desenho = np.zeros((720, 1280, 3), np.uint8)

# Tela de captura
cap = cv2.VideoCapture(0)
cap.set(3, largura_tela)
cap.set(4, altura_tela)

while True:
    sucesso, imagem = cap.read()
    # Inverter a imagem (1 para horizontal)
    imagem = cv2.flip(imagem, 1)
    imagem = detector.encontrar_maos(imagem)
    lista_landmark = detector.encontrar_posicao(imagem, desenho=False)
    
    # Pegar as posições das landmarks que usaremos
    if lista_landmark:
        # Pegar a posição de X e Y da ponta do indicador
        x1, y1, = lista_landmark[8][1:]
        
        # Pegar a posição de X e Y da ponta do dedo médio
        x2, y2 = lista_landmark[12][1:]
        
        # Checar se os dedos estão levantados
        dedos = detector.dedos_levantados()
        
        # Modo de seleção: indicador e médio levantados
        if dedos[1] and dedos[2]:
            x_anterior, y_anterior = 0, 0
            # Verificar se o dedo está indo para o topo da tela
            if y1 < 130:
                if 40 < x1 < 200:
                    cabecalho = lista_fotos[0]  # magenta
                    cor = (255, 0, 255)
                elif 300 < x1 < 520:
                    cabecalho = lista_fotos[1]  # azul
                    cor = (255, 0, 0)
                elif 570 < x1 < 785:
                    cabecalho = lista_fotos[2]  # verde
                    cor = (0, 255, 0)
                elif 860 < x1 < 990:
                    cabecalho = lista_fotos[3]  # amarelo
                    cor = (0, 255, 255)
                elif 1060 < x1 < 1220:
                    cabecalho = lista_fotos[4]  # borracha
                    cor = (0, 0, 0)
            
            cv2.rectangle(imagem, (x1, y1-25), (x2, y2+25), cor, cv2.FILLED)
            
        # Modo de desenho: indicador levantado e médio abaixado
        if dedos[1] and dedos[2] == False:
            cv2.circle(imagem, (x1, y1), 15, cor, cv2.FILLED)
            if x_anterior == 0 and y_anterior == 0:
                x_anterior, y_anterior = x1, y1
            
            if cor == (0, 0, 0):
                cv2.line(imagem, (x_anterior, y_anterior), (x1, y1), cor, espessura_borracha)
                cv2.line(tela_desenho, (x_anterior, y_anterior), (x1, y1), cor, espessura_borracha)
                
            else:
                cv2.line(imagem, (x_anterior, y_anterior), (x1, y1), cor, espessura_pincel)
                cv2.line(tela_desenho, (x_anterior, y_anterior), (x1, y1), cor, espessura_pincel)
                
            x_anterior, y_anterior = x1, y1
            
    # Juntar a tela de captura com a de desenho
    imagem_cinza = cv2.cvtColor(tela_desenho, cv2.COLOR_RGB2GRAY)
    _, imagem_invertida = cv2.threshold(imagem_cinza, 50, 255, cv2.THRESH_BINARY_INV)
    imagem_invertida = cv2.cvtColor(imagem_invertida, cv2.COLOR_GRAY2BGR)
    imagem = cv2.bitwise_and(imagem, imagem_invertida)
    imagem = cv2.bitwise_or(imagem, tela_desenho)
    
    # Colocar a foto no topo da tela
    imagem[0: 130, 0: 1280] = cabecalho
    
    # Mostrar a imagem na tela
    cv2.imshow('Pintura Virtual', imagem)
    
    # Terminar o loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
        
# Fechar a janela de captura
cap.release()
cv2.destroyAllWindows()
