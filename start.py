from game import Game
import pygame
import sys

def main() -> None:
    pygame.init()
    screen=pygame.display.set_mode((600,401))
    pygame.display.set_caption("Co Caro")
    background = pygame.image.load("img/start.png")
    while True:
        screen.blit(background, (0, 0))
        check_events()
        pygame.display.flip()

def check_events():
    for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                 pygame.quit() 
                 exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN  :
                pos = pygame.mouse.get_pos()
                
                if pos[0] >= 243 and pos[0] <= 354 and pos[1] >= 280 and pos[1] <= 315 :
                    game = Game(1)
                    game.play()     
                if pos[0] >= 243 and pos[0] <= 354 and pos[1] >= 320 and pos[1] <= 355 :
                    game = Game(2)
                    game.play() 


if __name__ == "__main__":
    main()
