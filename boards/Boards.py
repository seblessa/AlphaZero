import contextlib
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    import pygame
import logging

log = logging.getLogger(__name__)

# Colors

BLACK = (0, 0, 0)

WHITE = (255, 255, 255)

DARKER_BLUE = (50, 50, 150)
DARK_BLUE = (15, 15, 120)
BLUE = (135, 206, 250)

RED = (255, 0, 0)
PRESSED_RED = (153, 0, 0)
BG_RED = (255, 95, 95)

YELLOW = (249, 166, 3)
PRESSED_YELLOW = (238, 118, 0)
BG_YELLOW = (241, 235, 156)

BROWN = (139, 69, 19)


def draw_board_ataxx(screen, game, selected_piece=None):
    rows, cols = game.n, game.m
    offset = (screen.get_width() - cols * 100) // 2

    screen.fill(BG_RED if game.player_turn == 1 else BG_YELLOW)

    for row in range(rows):
        for col in range(cols):
            x = col * 100 + 50 + offset
            y = row * 100 + 50 + offset

            # Draw the board circles
            pygame.draw.circle(screen, BLUE, (x, y), 40)

            # Draw pieces
            if game.board[row][col] == 1:
                pygame.draw.circle(screen, RED, (x, y), 40)
            elif game.board[row][col] == -1:
                pygame.draw.circle(screen, YELLOW, (x, y), 40)

            # Highlight selected piece
            if (row, col) == selected_piece:
                if game.player_turn == 1:
                    pygame.draw.circle(screen, PRESSED_RED, (x, y), 40)
                else:
                    pygame.draw.circle(screen, PRESSED_YELLOW, (x, y), 40)

            # Highlight possible moves for walkers and jumpers of the selected piece
            if selected_piece is not None:
                walkers, jumpers = game.get_legal_actions()

                # Darker shade for walkers
                if (selected_piece, (row, col)) in walkers:
                    pygame.draw.circle(screen, DARK_BLUE, (x, y), 40)

                # Even darker shade for jumpers
                if (selected_piece, (row, col)) in jumpers:
                    pygame.draw.circle(screen, DARKER_BLUE, (x, y), 40)

    pygame.display.flip()


def draw_board_go(screen, game):
    if game.n == 7:
        cell_size = 100
        tam = 40
    else:
        cell_size = 70
        tam = 28
    rows, cols = game.n, game.m
    offset = (screen.get_width() - cols * cell_size) // 2


    screen.fill(BROWN)

    for row in range(rows + 1):
        for col in range(cols + 1):
            x = col * cell_size + offset - (cell_size/2)
            y = row * cell_size + offset - (cell_size/2)
            pygame.draw.circle(screen, BLACK, (x, y), 7)

    # Desenhar linhas do tabuleiro passando pelos pontos possíveis de jogada
    for row in range(rows + 1):
        pygame.draw.line(screen, BLACK, (offset, row * cell_size + offset - (cell_size/2)),
                         (cols * cell_size + offset, row * cell_size + offset - (cell_size/2)), 1)
    for col in range(cols + 1):
        pygame.draw.line(screen, BLACK, (col * cell_size + offset - (cell_size/2), offset),
                         (col * cell_size + offset - (cell_size/2), rows * cell_size + offset), 1)

    for row in range(rows):
        for col in range(cols):
            x = col * cell_size + (cell_size/2) + offset
            y = row * cell_size + (cell_size/2) + offset

            # Draw pieces
            if game.board[row][col] == 1:
                pygame.draw.circle(screen, BLACK, (x, y), tam)
            elif game.board[row][col] == -1:
                pygame.draw.circle(screen, WHITE, (x, y), tam)

    # Calcula a posição para colocar o botão "Pass" no centro inferior
    pass_button_width = 80
    pass_button_height = 30
    pass_button_x = (screen.get_width() - pass_button_width) - 50
    pass_button_y = screen.get_height() - pass_button_height - 20

    # Desenha o botão "Pass"
    pass_button_rect = pygame.Rect(pass_button_x, pass_button_y, pass_button_width, pass_button_height)
    pygame.draw.rect(screen, BLACK, pass_button_rect, 2)

    font = pygame.font.Font(None, 24)
    pass_text = font.render("Pass", True, BLACK)
    text_rect = pass_text.get_rect(center=pass_button_rect.center)
    screen.blit(pass_text, text_rect)

    if game.player_turn == 1:
        color = BLACK
        num = 1
    else:
        color = WHITE
        num = 2
    # Adicione esta parte para exibir o jogador atual
    font = pygame.font.Font(None, int(24))
    player_text = font.render(f"Player {num}", True, color)
    text_rect = player_text.get_rect(bottomleft=(int(20), screen.get_height() - int(20)))
    screen.blit(player_text, text_rect)

    pygame.display.flip()

