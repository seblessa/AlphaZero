import contextlib
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    import pygame
from boards import draw_board_go as draw_board
import numpy as np
import copy
import sys

# Colors

BLACK = (0, 0, 0)

WHITE = (255, 255, 255)


class Go:
    """
    This class describes the game of Go.
    Player1 is represented by 1 in the board
    player2 is represented by -1 in the board
    Board is represented by a 2D matrix of size n x m
    Passes is a dictionary that keeps track if a player has passed or not in last turn
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.name = 'G'
        self.model_name = f"{self.name}{n}x{m}"
        self.player_turn = 1
        self.score = {1: 0, -1: 0}
        self.board = [[0 for _ in range(m)] for _ in range(n)]
        self.game_over = False
        self.winner = 0
        self.last_action = {1: (None, None), -1: (None, None)}
        self.last_board = None
        self.passes = {1: False, -1: False}
        self.komi = 5.5

    def clone(self):
        newGame = Go(self.n, self.m)
        newGame.player_turn = copy.deepcopy(self.player_turn)
        newGame.score = copy.deepcopy(self.score)
        newGame.board = copy.deepcopy(self.board)
        newGame.game_over = copy.deepcopy(self.game_over)
        newGame.winner = self.winner
        newGame.last_action = copy.deepcopy(self.last_action)
        newGame.last_board = copy.deepcopy(self.last_board)
        newGame.passes = copy.deepcopy(self.passes)
        return newGame

    def get_score(self):
        return self.score[-self.player_turn]

    def apply_action(self, action, verbose=False):
        """
        Input:
            action: action taken by current player

        Logic:
            If action is pass, then change turn and return
            Elif actions is valid, then update the board and change turn
        """
        if self.game_over:
            raise RuntimeError("Game over is trying to apply action")

        if action == self.getActionSize() - 1:
            self.passes[self.player_turn] = True
            self.change_turn()
            self.last_action[self.player_turn] = (None, None)
            self.update_score(verbose)
            self.is_terminal()
            return
        else:
            self.passes = {1: False, -1: False}

        i, j = self.decode(action)

        # if not pass
        if self.is_valid(i, j):
            self.board[i][j] = self.player_turn
            self.last_board = copy.deepcopy(self.board)
            self.last_action[self.player_turn] = (i, j)
            self.transform_surroundings(self.player_turn)
            self.change_turn()
            self.update_score(verbose)
            self.is_terminal()

    def is_terminal(self):
        """
        Check if the game is over (win, loss, or draw)
        """
        if self.passes[1] and self.passes[-1]:
            self.game_over = True
            if self.score[1] > self.score[-1]:
                self.winner = 1
            elif self.score[1] < self.score[-1]:
                self.winner = -1
            else:
                self.winner = 0
        return self.game_over

    def update_score(self, verbose=False):
        def influence_score(x, y):
            score = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.m:
                    score += self.board[nx][ny]
            return score

        def explore_territory(x, y):
            if (x, y) in visited or not (0 <= x < self.n and 0 <= y < self.m):
                return
            visited.add((x, y))

            if self.board[x][y] == 0:
                score = influence_score(x, y)
                if score > 0:
                    self.score[1] += 1
                elif score < 0:
                    self.score[-1] += 1

        """
        Update the score of each player
        """
        self.score = {1: 0, -1: self.komi}
        # Count the number of pieces of each player
        for i in range(self.n):
            for j in range(self.m):
                if self.board[i][j] != 0:
                    self.score[self.board[i][j]] += 1

        # Count the number of conquered territories of each player
        visited = set()

        for i in range(self.n):
            for j in range(self.m):
                if self.board[i][j] == 0 and (i, j) not in visited:
                    explore_territory(i, j)

        if verbose:
            print("\nblack score: ", self.score[1])
            print(f"white score: {self.score[-1]}\n")

    def encode(self, move):
        return move[0] * self.m + move[1]

    def decode(self, index):
        dest_m = index % self.m
        index //= self.m
        dest_n = index % self.n
        return dest_n, dest_m

    def change_turn(self):
        self.player_turn *= -1

    def getBoardSize(self):
        return self.n, self.m

    def getActionSize(self):
        """Returns: number of all possible actions"""
        return (self.n * self.m) + 1

    def get_encoded_actions(self):
        """
        Returns: list of all encoded possible actions from the current board state (legal moves + pass).
        Valid moves are encoded as 1, invalid moves as 0.
        """
        legal_actions = self.get_legal_actions()
        encoded_legal_actions_indexes = [self.encode(action) for action in legal_actions]
        encoded_legal_actions = [0] * self.getActionSize()
        for index in encoded_legal_actions_indexes:
            encoded_legal_actions[index] = 1

        return encoded_legal_actions

    def get_encoded_board(self, state=None):

        if state is None:
            state = self.board
        layer_1 = np.where(np.array(state) == -1, 1, 0).astype(np.float32)
        layer_2 = np.where(np.array(state) == 0, 1, 0).astype(np.float32)
        layer_3 = np.where(np.array(state) == 1, 1, 0).astype(np.float32)
        return np.stack([layer_1, layer_2, layer_3]).astype(np.float32)

    def does_capture(self, i, j):
        goClone = self.clone()
        goClone.board[i][j] = self.player_turn
        return goClone.transform_surroundings(self.player_turn)

    def is_valid(self, i, j):
        return self.has_liberties([(i, j)]) or (self.does_capture(i, j) and self.last_board[i][j] != self.player_turn)

    def get_legal_actions(self):
        """Returns: list of tuples of all possible actions from the current board state (legal moves + pass)"""
        legal_moves = []
        for i in range(self.n):
            for j in range(self.m):
                if self.last_action[self.player_turn] == (i, j):
                    continue
                if self.board[i][j] == 0:
                    if self.is_valid(i, j):
                        legal_moves.append((i, j))
        legal_moves.append((self.n - 1, self.m))
        return legal_moves

    def is_empty(self, i, j):
        return (0 <= i < self.n and 0 <= j < self.m) and self.board[i][j] == 0

    def has_liberties(self, group):
        for position in group:
            i, j = position
            if self.is_empty(i + 1, j) or self.is_empty(i - 1, j) or self.is_empty(i, j + 1) or self.is_empty(i, j - 1):
                return True
        return False

    def transform_surroundings(self, color):
        for i in range(self.n):
            for j in range(self.m):
                if self.board[i][j] == -color:
                    visited = [[False for _ in range(self.m)] for _ in range(self.n)]
                    group = []
                    self.dfs(i, j, -color, visited, group)

                    if not self.has_liberties(group):
                        self.remove_group(group)
                        return True
        return False

    def dfs(self, i, j, color, visited, group):
        if i < 0 or i >= self.n or j < 0 or j >= self.m or visited[i][j] or self.board[i][j] != color:
            return

        visited[i][j] = True
        group.append((i, j))

        self.dfs(i + 1, j, color, visited, group)
        self.dfs(i - 1, j, color, visited, group)
        self.dfs(i, j + 1, color, visited, group)
        self.dfs(i, j - 1, color, visited, group)

    def remove_group(self, group):
        for position in group:
            i, j = position
            self.board[i][j] = 0

    def play_game(self, player1, player2, verbose=False):
        """
        Runs a full game of Go between two players.
        """
        screen = None
        offset = None
        clock = None
        pass_button_rect = None
        screen_width = None
        screen_height = None

        if self.n == 7:
            size_cell = 100
        else:
            size_cell = 70

        if verbose:
            pygame.init()
            screen_width = self.m * size_cell  # columns * 100
            screen_height = self.n * size_cell  # rows * 100
            screen = pygame.display.set_mode((screen_width, screen_height + size_cell))
            pygame.display.set_caption("Ataxx" if self.name == "A" else "Go")
            clock = pygame.time.Clock()
            offset = (screen.get_width() - self.n * size_cell) // 2

            # Calcula a posição para colocar o botão "Pass" no centro inferior
            pass_button_width = 80
            pass_button_height = 30
            pass_button_x = (screen.get_width() - pass_button_width) - 50
            pass_button_y = screen.get_height() - pass_button_height - 20
            # Desenha o botão "Pass"
            pass_button_rect = pygame.Rect(pass_button_x, pass_button_y, pass_button_width, pass_button_height)

        while self.winner == 0:
            if self.player_turn == 1:
                nextPlayer = player1
            else:
                nextPlayer = player2

            if not verbose:
                action = nextPlayer.get_action(self)
                self.apply_action(action)

            # pygame loop
            else:

                draw_board(screen, self)

                if nextPlayer == "human":
                    legal_moves = self.get_legal_actions()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = pygame.mouse.get_pos()
                            col = (mouse_pos[0] - offset) // size_cell
                            row = (mouse_pos[1] - offset) // size_cell
                            if pass_button_rect.collidepoint(mouse_pos):
                                self.passes[self.player_turn] = True
                                self.change_turn()
                                self.is_terminal()
                            elif 0 <= row < self.n and 0 <= col < self.m:
                                if (row, col) in legal_moves:
                                    self.apply_action(self.encode((row, col)), verbose=True)
                else:
                    pygame.time.wait(500)
                    action = nextPlayer.get_action(self)
                    self.apply_action(action)

                pygame.display.flip()
                clock.tick(60)

        if verbose:
            draw_board(screen, self)
            font = pygame.font.Font(None, 50)
            text = font.render(f"Game Over, {'Black' if self.winner == 1 else 'White'} wins", True, (178, 190, 181))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.display.quit()
            pygame.quit()
            print("winner: ", self.winner)
            print()
