import contextlib
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    import pygame
from boards import draw_board_ataxx as draw_board
import numpy as np
import copy
import sys


class Ataxx:
    """
    This class describes the game of Ataxx.
    Player1 is represented by 1 in the board
    player2 is represented by -1 in the board
    Board is represented by a 2D matrix of size n x m
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.name = 'A'
        self.model_name = f"{self.name}{n}x{m}"
        self.player_turn = 1
        self.score = {1: 0, -1: 0}
        self.board = [[0 for _ in range(m)] for _ in range(n)]
        self.game_over = False
        self.winner = 0
        # Put the initial pieces in the corners
        self.board[0][0] = self.board[self.n - 1][self.m - 1] = -1
        self.board[self.n - 1][0] = self.board[0][self.m - 1] = 1

    def clone(self):
        newGame = Ataxx(self.n, self.m)
        newGame.player_turn = copy.deepcopy(self.player_turn)
        newGame.score = copy.deepcopy(self.score)
        newGame.board = copy.deepcopy(self.board)
        newGame.game_over = copy.deepcopy(self.game_over)
        newGame.winner = copy.deepcopy(self.winner)
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

        piece, (i, j) = self.decode(action)

        jumpers, walkers = self.get_legal_actions()
        walkers = [walker[1] for walker in walkers if piece == walker[0]]
        jumpers = [jumper[1] for jumper in jumpers if piece == jumper[0]]
        if (i, j) in walkers:
            self.board[i][j] = self.player_turn
            self.transform_surroundings(i, j)
            self.update_score(verbose)
            self.is_terminal()
            self.change_turn()

        elif (i, j) in jumpers:
            self.board[i][j] = self.player_turn
            self.board[piece[0]][piece[1]] = 0
            self.transform_surroundings(i, j)
            self.update_score(verbose)
            self.is_terminal()
            self.change_turn()

    def is_terminal(self):
        def fill_board(color):
            for row in range(self.n):
                for col in range(self.m):
                    if self.board[row][col] == 0:
                        self.board[row][col] = color
                        self.score[color] += 1
        """
        Check if the game is over (win, loss, or draw)
        """
        if len(self.get_player_pieces(1)) == 0:
            self.game_over = True
            self.winner = -1
        elif len(self.get_player_pieces(-1)) == 0:
            self.game_over = True
            self.winner = 1
        elif len(self.get_player_pieces(1)) + len(self.get_player_pieces(-1)) == self.n * self.m:
            self.game_over = True
            self.winner = 1 if self.score[1] > self.score[-1] else -1
        elif self.get_legal_actions(1) == ([], []):
            self.game_over = True
            fill_board(-1)
            self.winner = 1 if self.score[1] > self.score[-1] else -1
        elif self.get_legal_actions(-1) == ([], []):
            self.game_over = True
            fill_board(1)
            self.winner = 1 if self.score[1] > self.score[-1] else -1
        return self.game_over

    def update_score(self, verbose=False):
        """
        Update the score of each player
        """
        self.score = {1: 0, -1: 0}
        for row in range(self.n):
            for col in range(self.m):
                if self.board[row][col] == 1:
                    self.score[1] += 1
                elif self.board[row][col] == -1:
                    self.score[-1] += 1
        if verbose:
            print("\nRed score: ", self.score[1])
            print(f"Yellow score: {self.score[-1]}\n")

    def encode(self, piece, move):
        return piece[0] * self.m * self.n * self.m + piece[1] * self.n * self.m + move[0] * self.m + move[1]

    def decode(self, index):
        dest_m = index % self.m
        index //= self.m
        dest_n = index % self.n
        index //= self.n
        src_m = index % self.m
        index //= self.m
        src_n = index % self.n
        return (src_n, src_m), (dest_n, dest_m)

    def change_turn(self):
        self.player_turn *= -1

    def getBoardSize(self):
        return self.n, self.m

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.n * self.m * self.n * self.m

    def get_encoded_actions(self):
        legal_moves1, legal_moves2 = self.get_legal_actions()
        encoded_legal_actions = [0] * self.getActionSize()

        for piece, move in legal_moves1:
            value = self.encode(piece, move)
            encoded_legal_actions[value] = 1
        for piece, move in legal_moves2:
            value = self.encode(piece, move)
            encoded_legal_actions[value] = 1
        return encoded_legal_actions

    def get_encoded_board(self, state=None):
        if state is None:
            state = self.board
        layer_1 = np.where(np.array(state) == -1, 1, 0).astype(np.float32)
        layer_2 = np.where(np.array(state) == 0, 1, 0).astype(np.float32)
        layer_3 = np.where(np.array(state) == 1, 1, 0).astype(np.float32)

        result = np.stack([layer_1, layer_2, layer_3]).astype(np.float32)

        return result

    def is_valid(self, i, j):
        if 0 <= i < self.n and 0 <= j < self.m:
            return self.board[i][j] == self.player_turn
        return False

    def get_player_pieces(self, color):
        pieces = []
        for row in range(self.n):
            for column in range(self.m):
                if self.board[row][column] == color:
                    pieces.append((row, column))
        return pieces

    def transform_surroundings(self, i, j):
        rows, cols = self.n, self.m
        for n in range(-1, 2):
            for m in range(-1, 2):
                if 0 <= i + n < rows and 0 <= j + m < cols:
                    if self.board[i + n][j + m] == -self.player_turn:
                        self.board[i + n][j + m] = self.player_turn

    def jump(self, i, j):
        successors = []
        for n in range(-2, 3):
            for m in range(-2, 3):
                if n != 0 or m != 0 or n != 1 or m != 1 or n != -1 or m != -1:
                    if 0 <= i + n < self.n and 0 <= j + m < self.m:
                        if self.board[i + n][j + m] != 1 and self.board[i + n][j + m] != -1:
                            if abs(n) > 1 or abs(m) > 1:
                                successors.append((i + n, j + m))
        return successors

    def walk(self, i, j):
        successors = []
        for n in range(-1, 2):
            for m in range(-1, 2):
                if n != 0 or m != 0:
                    if 0 <= i + n < self.n and 0 <= j + m < self.m:
                        if self.board[i + n][j + m] != 1 and self.board[i + n][j + m] != -1:
                            successors.append((i + n, j + m))
        return successors

    def get_legal_actions(self, color=None):
        """Returns: list of all possible actions from the current board state (legal moves + pass)"""
        if color is None:
            color = self.player_turn
        walkers = []
        jumpers = []
        for piece in self.get_player_pieces(color):
            jumpers.extend((piece, successor) for successor in self.jump(*piece))
            walkers.extend((piece, successor) for successor in self.walk(*piece))
        return jumpers, walkers

    def play_game(self, player1, player2, verbose=False):
        """
        Runs a full game of Go between two players.
        """
        screen = None
        offset = None
        clock = None
        screen_width = None
        screen_height = None
        selected_piece = None
        if verbose:
            pygame.init()
            screen_width = self.m * 100  # columns * 100
            screen_height = self.n * 100  # rows * 100
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Ataxx" if self.name == "A" else "Go")
            clock = pygame.time.Clock()
            offset = (screen.get_width() - self.n * 100) // 2

        while self.winner == 0:
            if self.player_turn == 1:
                nextPlayer = player1
            else:
                nextPlayer = player2
            # pygame loop
            if not verbose:
                action = nextPlayer.get_action(self)
                self.apply_action(action)
            else:
                draw_board(screen, self, selected_piece)
                if nextPlayer == "human":
                    legal_actions = self.get_legal_actions()
                    if len(legal_actions) == 0:
                        pygame.time.wait(500)
                        self.change_turn()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = pygame.mouse.get_pos()
                            col = (mouse_pos[0] - offset) // 100
                            row = (mouse_pos[1] - offset) // 100
                            if 0 <= row < self.n and 0 <= col < self.m:
                                if selected_piece is None:
                                    if self.is_valid(row, col):
                                        selected_piece = (row, col)
                                elif selected_piece == (row, col):  # Deselect the piece
                                    selected_piece = None
                                else:
                                    jumpers, walkers = legal_actions
                                    moves = [pair[1] for pair in jumpers if selected_piece == pair[0]]
                                    moves.extend([pair[1] for pair in walkers if selected_piece == pair[0]])
                                    for move in moves:
                                        if move == (row, col):
                                            self.apply_action(self.encode(selected_piece, (row, col)), verbose=True)
                                            selected_piece = None
                                            break
                else:
                    pygame.time.wait(1000)
                    action = nextPlayer.get_action(self)
                    self.apply_action(action)

                pygame.display.flip()
                clock.tick(60)

        if verbose:
            draw_board(screen, self)
            font = pygame.font.Font(None, 50)
            text = font.render(f"Game Over, {'Red' if self.winner == 1 else 'Yellow'} wins", True, (0, 0, 0))
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.display.quit()
            pygame.quit()
            print("winner: ", self.winner)
