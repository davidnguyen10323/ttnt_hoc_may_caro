import csv
import re
import os
import copy
from typing import Any, Iterable, List, Tuple


SIZE = 15
P1_VICTORY_PATTERN = re.compile(r"11111")
P2_VICTORY_PATTERN = re.compile(r"22222")

PATTERN_1 = re.compile(r"211110|011112")
PATTERN_2 = re.compile(r"011110")
PATTERN_3 = re.compile(r"01110")
PATTERN_4 = re.compile(r"2011100|0011102")
PATTERN_5 = re.compile(r"010110|011010")
PATTERN_6 = re.compile(r"0110|01010")

P2_PATTERN_1 = re.compile(r"122220|022221")
P2_PATTERN_2 = re.compile(r"022220")
P2_PATTERN_3 = re.compile(r"02220")
P2_PATTERN_4 = re.compile(r"1022200|0022201")
P2_PATTERN_5 = re.compile(r"020220|022020")
P2_PATTERN_6 = re.compile(r"0220|02020")


def spiral(n: int) -> List[Tuple[int, int]]:
   
    dx, dy = 1, 0  # Starting increments
    x, y = 0, 0    # Starting location
    matrix = [[-1]*n for _ in range(n)]
    for i in range(n**2):
        matrix[x][y] = i
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and matrix[nx][ny] == -1:
            x, y = nx, ny
        else:
            dx, dy = -dy, dx
            x, y = x + dx, y + dy
    output = [(0, 0) for _ in range(n**2)]
    for i in range(n):
        for j in range(n):
            output[matrix[i][j]] = (i, j)
    return output

SPIRAL_ORDER = spiral(SIZE)[::-1]


def stringfy(matrix: List[List[int]]) -> str:
    string = ""
    for line in matrix:
        string += "".join(map(str, line)) + "\n"
    return string


class Board():

    def __init__(self, ai_player: int) -> None:
        self._board = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
        self._actual_player = 1
        self._ai_player = ai_player
        self.history = []  # Thêm thuộc tính này

    def place_stone(self, position: Tuple[int, int]) -> None:
        x_coord, y_coord = position
        self._board[y_coord][x_coord] = self._actual_player
        self.history.append(position)  # Lưu nước đi vào lịch sử
        self._actual_player = 1 if self._actual_player == 2 else 2

    def analyze_history(self) -> dict:
        """Phân tích lịch sử để tìm ra các nước đi và tần suất của chúng."""
        move_count = {}
        for move in self.history:
            move_count[move] = move_count.get(move, 0) + 1
        return move_count

    def analyze_move_history(self, history_file: str) -> dict:
        """Phân tích lịch sử từ file CSV để tìm ra tần suất các nước đi thắng."""
        move_count = {}
        with open(history_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua dòng tiêu đề
            for row in reader:
                move = eval(row[1])  # Chuyển chuỗi thành tuple
                move_count[tuple(move)] = move_count.get(tuple(move), 0) + 1
        return move_count

    def save_history_to_csv(self, filename: str) -> None:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Move Number', 'Position'])  # Tiêu đề cột
            for i, move in enumerate(self.history):
                writer.writerow([i + 1, move])  # Lưu từng nước đi

    def load_history_from_csv(self, filename: str) -> list:
        history = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua dòng tiêu đề
            for row in reader:
                move = eval(row[1])  # Chuyển chuỗi thành tuple
                history.append(move)
        return history

    def is_empty(self, position: Tuple[int, int]) -> bool:
        x_coord, y_coord = position
        return self._board[y_coord][x_coord] == 0

    def _diagonals(self) -> List[List[int]]:
        return [[self._board[SIZE - p + q - 1][q]
                 for q in range(max(p - SIZE + 1, 0), min(p + 1, SIZE))]
                for p in range(SIZE + SIZE - 1)]

    def _antidiagonals(self) -> List[List[int]]:
        return [[self._board[p - q][q]
                 for q in range(max(p - SIZE + 1, 0), min(p + 1, SIZE))]
                for p in range(SIZE + SIZE - 1)]

    def _columns(self) -> List[List[int]]:
        return [[self._board[i][j]
                 for i in range(SIZE)]
                for j in range(SIZE)]

    def victory(self) -> bool:
        whole_board = "\n".join(
            map(stringfy,
                [self._board,
                 self._diagonals(),
                 self._antidiagonals(),
                 self._columns()]))
        if P1_VICTORY_PATTERN.search(whole_board):
            return 1 
        if P2_VICTORY_PATTERN.search(whole_board):
            return 2
        return False
        

    def evaluate(self, move_history: dict = None) -> int:
        whole_board = "\n".join(map(stringfy,[self._board,self._diagonals(),
                                            self._antidiagonals(),self._columns()]))
        p1_value = 0
        p2_value = 0
        if P1_VICTORY_PATTERN.search(whole_board):
            p1_value += 2**25
        elif P2_VICTORY_PATTERN.search(whole_board):
            p2_value += 2**25

        # Tăng điểm cho các nước đi theo các mẫu
        p1_value += 37 * 56 * len(PATTERN_2.findall(whole_board))
        p1_value += 56 * len(PATTERN_1.findall(whole_board))
        p1_value += 56 * len(PATTERN_3.findall(whole_board))
        p1_value += 56 * len(PATTERN_4.findall(whole_board))
        p1_value += 56 * len(PATTERN_5.findall(whole_board))
        p1_value += len(PATTERN_6.findall(whole_board))

        p2_value += 37 * 56 * len(P2_PATTERN_2.findall(whole_board))
        p2_value += 56 * len(P2_PATTERN_1.findall(whole_board))
        p2_value += 56 * len(P2_PATTERN_3.findall(whole_board))
        p2_value += 56 * len(P2_PATTERN_4.findall(whole_board))
        p2_value += 56 * len(P2_PATTERN_5.findall(whole_board))
        p2_value += len(P2_PATTERN_6.findall(whole_board))

        # Thêm giá trị từ lịch sử nước đi (nếu có)
        if move_history:
            for move in self.history:
                if move in move_history:
                    # Cộng thêm điểm cho các nước đi thắng trong lịch sử
                    if self._actual_player == 1:
                        p1_value += 10 * move_history[move]  # Tăng giá trị cho người chơi 1
                    else:
                        p2_value += 10 * move_history[move]  # Tăng giá trị cho người chơi 2

        return p1_value - p2_value \
            if self._ai_player == 1 \
            else p2_value - p1_value

    def adjacents(self) -> Iterable[Any]:
        actual_board = copy.deepcopy(self)
        for i, j in SPIRAL_ORDER:
            if actual_board.is_empty((i, j)):
                actual_board.place_stone((i, j))
                yield actual_board
                actual_board._actual_player = \
                    1 if actual_board._actual_player == 2 else 2
                actual_board._board[j][i] = 0



