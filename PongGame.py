import pygame
import random
import numpy as np
from IA import RedeNeural
# from pylab import plot,show
from pygame.locals import *
from sys import exit

geracao = 0
pygame.init()

nn = RedeNeural(2, 6, 4, 2)


def main():

    lisdy = ['up', 'down']

    global geracao
    geracao += 1

    anguloBol = 8

    larguraTela = 1280
    alturaTela = 480

    gameScreenWidth = 640
    gameScreenHeight = 480

    infoScreenWidth = larguraTela + gameScreenWidth
    infoScreenHeight = alturaTela

    # Gráfico
    eixox = []
    eixoy = []
    cont = 0

    contagemPerdas = 0

    x = 10
    y = (alturaTela/2 -30)

    # Raquete
    posRaqx = 10
    posRaqy = (gameScreenHeight / 2) - 30

    # Bolinha
    posBolx = gameScreenWidth / 2
    posBoly = random.randint(20, 460)
    dx = 'right'
    dy = lisdy[random.randint(0, 1)]

    tela = pygame.display.set_mode((larguraTela, alturaTela))
    pygame.display.set_caption('Pong')
    relogio = pygame.time.Clock()

    fonte = pygame.font.SysFont('arial', 30, True, False)
    pontos = 0

    target = []


    while True:
        relogio.tick(60)
        tela.fill((0, 0, 0))


        texto_formatado = fonte.render(f"Pontos: {pontos}", True, (255, 255, 255))
        tela.blit(texto_formatado, (400, 20))

        texto_formatado = fonte.render(f"Geração: {geracao}", True, (255, 255, 255))
        tela.blit(texto_formatado, (20, 20))

        texto_formatado = fonte.render(f"{contagemPerdas}", True, (255, 0, 0))
        tela.blit(texto_formatado, (300, 20))


        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        raquete = pygame.draw.rect(tela, (255, 255, 255), (posRaqx, posRaqy, 10, 60))
        bolinha = pygame.draw.circle(tela, (255, 255, 255), (posBolx, posBoly), 8)

        #infos da nn
        pygame.draw.rect(tela, (255, 255, 204), (gameScreenWidth, 0, larguraTela-gameScreenWidth, gameScreenHeight))


        # Movendo Bolinha eixo x
        # Chegando do lado direito
        if posBolx >= gameScreenWidth - 10:
            dx = 'left'

        # Chegando no lado esquerdo
        if posBolx < 10:
            dx = 'right'
            pontos = 0
            contagemPerdas += 1

            geracao += 1

            erro = (posRaqy+30) - posBoly
            if erro < 0:
                erro = erro * (-1)

            if (posRaqy + 30) - posBoly > 0:
                target = [[1, 0]]
            else:
                target = [[0, 1]]

            nn.train([[(posRaqy + 30) - posBoly, (posBolx + 10) - posBolx]], target)

            # if contagemPerdas >= 6:

            #     erro = 600
            #     pontos = 0
            #     contagemPerdas = 0

            #     nn.train([[(posRaqy + 30) - posBoly, (posBolx + 10) - posBolx]], target)


            # Bolinha
            posBolx = gameScreenWidth / 2
            posBoly = random.randint(20, 460)
            dx = 'right'
            dy = lisdy[random.randint(0, 1)]

        if dx == 'right':
            posBolx = posBolx + anguloBol
        else:
            posBolx = posBolx - anguloBol

        # Movendo Bolinha eixo y
        if posBoly >= gameScreenHeight - 10:
            dy = 'up'
            anguloBol = random.randint(6, 10)
        if posBoly <= 10:
            dy = 'down'
            anguloBol = random.randint(6, 10)

        if dy == 'up':
            posBoly = posBoly - anguloBol
        else:
            posBoly = posBoly + anguloBol

        # Checando colisão na raquete
        if dx == 'left':
            if posBolx <= posRaqx + 24:
                if posBoly >= posRaqy and posBoly <= posRaqy +60:
                    dx = 'right'
                    pontos = pontos + 1
                    erro = posRaqy + 30 - posBoly

                    eixoy.append(pontos)
                    eixox.append(geracao)


                    contagemPerdas = 0
                    if erro < 0:
                        erro = erro * (-1)

        move = nn.predict([[(posRaqy + 30) - posBoly, (posBolx + 10) - posBolx]])

        print(move)

        # Movendo a Raquete
        if move[0][0] > 0.5:
            if posRaqy != 0:
                posRaqy = posRaqy - 15
        if move[0][1] > 0.5:
            if posRaqy != gameScreenHeight - 60:
                posRaqy = posRaqy + 15

        target = []


        pygame.display.update()


if __name__ == '__main__':
    main()
