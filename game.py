from typing import Tuple
from board import Board
from ab_pruning import ab_pruning
import pygame
class Game():

    def __init__(self, ai_player) -> None:
        pygame.init()
        self.screen=pygame.display.set_mode((600,401))
        pygame.display.set_caption("Co Caro")
        self.background = pygame.image.load("img/background.png")
        self.SIZE = 15
        self.MARGIN = 1
        self.HEIGHT = 23
        self.WIDTH = 23
        self.BG = 21
        self.ending = False
        self._board = Board(ai_player)
        self._turn = 0
        self._ai_player = ai_player
        # Tải lịch sử các nước đi từ CSV
        self.move_history = self._board.analyze_move_history('move_history.csv')
        
    def end(self,pictureURL):
        self.background = pygame.image.load(pictureURL)
        while self.ending == True:
            self.screen.blit(self.background, (0, 0))
            self.check_events_ends()
            pygame.display.flip()
            
    def check_events_ends(self):
        for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit() 
                    exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN  :
                    pos = pygame.mouse.get_pos()
                    
                    if pos[0] >= 244 and pos[0] <= 356 and pos[1] >= 310 and pos[1] <= 340 :
                        self.reset()
                        self._turn = -1
                        self.ending = False

    def play(self) -> None:
        while True:
            self.run_game()
            self.screen.blit(self.background, (0, 0))

            while not self._board.victory():  # Trong khi chưa có ai thắng
                self.run_game()

                # Kiểm tra lượt chơi của AI
                if (self._turn % 2) + 1 == self._ai_player:
                    self._board, _ = ab_pruning(self._board, 2, -2 ** 32, 2 ** 32, True)
                    self.run_game()
                else:
                    self.run_game()
                    pos = [-1, -1]

                    while True:
                        pos = self.get_mouse_events()
                        if pos != [-1, -1]:  # Nếu có nước đi hợp lệ
                            self._board.place_stone(pos)
                            break

                self.run_game()

                # Kiểm tra kết thúc trò chơi
                tmp = self._board.victory()
                if tmp:
                    print("Lịch sử nước đi:", self._board.history)  # In lịch sử nước đi
                    self._board.save_history_to_csv('move_history.csv')  # Lưu lịch sử vào CSV
                    if tmp == 1 and self._ai_player == 1:
                        self.ending = True
                        self.end("img/lose.png")  # Người chơi thắng
                    elif tmp == 2 and self._ai_player == 2:
                        self.ending = True
                        self.end("img/win.png")  # AI thắng
                    elif tmp == 1 and self._ai_player == 2:
                        self.ending = True
                        self.end("img/win.png")  # AI thắng
                    elif tmp == 2 and self._ai_player == 1:
                        self.ending = True
                        self.end("img/lose.png")  # Người chơi thắng

                self._turn += 1  # Chuyển lượt chơi

    def reset(self):
        self._board._board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        self.background = pygame.image.load("img/background.png")
        self.screen.blit(self.background, (0, 0))
        self._board = Board(self._ai_player)
        self.draw_board()
        
    def run_game(self):
        pygame.display.flip() 
        self.draw_board()
        self.get_mouse_events()
            
    def get_mouse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                 pygame.quit() 
                 exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN :
                pos = pygame.mouse.get_pos()
                if pos[0] >= 465 and pos[0] <= 530 and pos[1] >= 320 and pos[1] <= 390 :
                    self.reset()
                    self._turn = 0
                    self.play()                    
                elif pos[0] >= 21 and pos[0] <= 380 and pos[1] >= 21 and pos[1] <= 380:
                    column = (pos[0] - self.BG) // (self.WIDTH + self.MARGIN)
                    row = (pos[1] - self.BG) // (self.HEIGHT + self.MARGIN)
                    if self._board._board[row][column] == 0:
                        return [column,row]
        return [-1,-1] 
   
                           
    def draw_board(self):
        imgO = pygame.image.load("img/O.png")
        imgX = pygame.image.load("img/X.png")
        imgO = pygame.transform.scale(imgO, (self.WIDTH,self.HEIGHT))
        imgX = pygame.transform.scale(imgX, (self.WIDTH,self.HEIGHT))
                   
        for row in range(self.SIZE):
            for column in range(self.SIZE):
                color = "#fffced"
                pygame.draw.rect(self.screen,
                            color, [self.BG+ (self.MARGIN + self.WIDTH) * column + self.MARGIN, self.BG + (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                            self.WIDTH, self.HEIGHT])
                if self._board._board[row][column] == 1:
                    self.screen.blit( imgX,  (self.BG +(self.MARGIN + self.WIDTH) * column + self.MARGIN, self.BG + (self.MARGIN + self.HEIGHT) * row + self.MARGIN) )
                if self._board._board[row][column] == 2:
                    self.screen.blit( imgO, (self.BG +(self.MARGIN + self.WIDTH) * column + self.MARGIN, self.BG + (self.MARGIN + self.HEIGHT) * row + self.MARGIN) )
        
        
        if self._ai_player == 2:
            com_turn = imgO
            player_turn = imgX
        else:
            player_turn = imgO
            com_turn = imgX
        self.screen.blit(player_turn, (502, 272))
        self.screen.blit(com_turn, (502, 245))                  
    