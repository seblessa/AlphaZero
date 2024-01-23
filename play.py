import contextlib
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    import pygame
from AlphaZero import AlphaZeroPlayer
from Games import Ataxx, Go
from AlphaZero import CNNET
from training import args
from boards import Button
import sys
import os


best_model_dir = "./best_model/"
media_dir = "./boards/images/"

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")

BG = pygame.image.load(media_dir + "Background.png")


def get_font(size):
    return pygame.font.Font(media_dir + "font.ttf", size)


def model_selection(game_name, n, m):
    model_dir = f"{best_model_dir}{game_name}{n}x{m}/"
    try:
        os.listdir(model_dir)
    except FileNotFoundError:
        print(f"The {game_name}{n}x{m} model isn't available.\n")
        exit(0)

    model_path = f'{model_dir}model.tar'
    return model_path


def play_game(game_name, versus_ai, n, m):
    """
    Initiates and plays the selected game.

    Parameters:
    - game_name (str): Either 'A' for Ataxx or 'G' for Go.
    - versus_ai (bool): True if playing against AI, False for player vs. player.
    - n (int): Number of rows in the game board.
    - m (int): Number of columns in the game board.
    """
    if game_name == "A":
        game = Ataxx(n, m)
    elif game_name == "G":
        game = Go(n, m)
    else:
        raise NotImplementedError
    player1 = "human"
    if versus_ai:
        model = CNNET(game, args)
        model.load_model(model_selection(game_name, n, m))
        player2 = AlphaZeroPlayer(model)
    else:
        player2 = "human"
    game.play_game(player1, player2, verbose=True)

    exit(0)


def boardSize(game_name, versus_ai):
    if game_name == "G":
        pygame.display.set_caption("Menu")
        while True:
            MENU_MOUSE_POS = pygame.mouse.get_pos()

            SCREEN.blit(BG, (0, 0))

            x77_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 250),
                                text_input="7x7", font=get_font(50), base_color="#b68f40", hovering_color="White")

            x99_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 400),
                                text_input="9x9", font=get_font(50), base_color="#b68f40", hovering_color="White")

            BACK_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 550),
                                 text_input="BACK", font=get_font(50), base_color="#8B0000", hovering_color="White")

            MENU_TEXT = get_font(50).render(f"SELECT THE BOARDSIZE TO PLAY {'AGAINST ALPHAZERO' if versus_ai else 'PVP'}", True, "#d7fcd4")
            MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

            SCREEN.blit(MENU_TEXT, MENU_RECT)

            for button in [x77_BUTTON, x99_BUTTON, BACK_BUTTON]:
                button.changeColor(MENU_MOUSE_POS)
                button.update(SCREEN)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x77_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_game(game_name, versus_ai, 7, 7)
                    if x99_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_game(game_name, versus_ai, 9, 9)
                    if BACK_BUTTON.checkForInput(MENU_MOUSE_POS):
                        gameMode(game_name)

            pygame.display.update()

    elif game_name == "A":
        pygame.display.set_caption("Menu")
        while True:
            MENU_MOUSE_POS = pygame.mouse.get_pos()

            SCREEN.blit(BG, (0, 0))

            x44_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(240, 325),
                                text_input="4X4", font=get_font(50), base_color="#b68f40", hovering_color="White")

            x55_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 325),
                                text_input="5x5", font=get_font(50), base_color="#b68f40", hovering_color="White")

            x66_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(1040, 325),
                                text_input="6x6", font=get_font(50), base_color="#b68f40", hovering_color="White")

            BACK_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 550),
                                 text_input="BACK", font=get_font(50), base_color="#8B0000", hovering_color="White")

            MENU_TEXT = get_font(50).render(f"SELECT THE BOARDSIZE TO PLAY {'AGAINST ALPHAZERO' if versus_ai else 'PVP'}", True, "#d7fcd4")
            MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

            SCREEN.blit(MENU_TEXT, MENU_RECT)

            for button in [x44_BUTTON, x55_BUTTON, x66_BUTTON, BACK_BUTTON]:
                button.changeColor(MENU_MOUSE_POS)
                button.update(SCREEN)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if x44_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_game(game_name, versus_ai, 4, 4)
                    if x55_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_game(game_name, versus_ai, 5, 5)
                    if x66_BUTTON.checkForInput(MENU_MOUSE_POS):
                        play_game(game_name, versus_ai, 6, 6)
                    if BACK_BUTTON.checkForInput(MENU_MOUSE_POS):
                        gameMode(game_name)
            pygame.display.update()


def gameMode(game_name):
    pygame.display.set_caption("Menu")
    while True:
        MENU_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.blit(BG, (0, 0))

        PvP_BUTTON = Button(image=pygame.image.load(media_dir + "Options Rect.png"), pos=(640, 250),
                            text_input="PLAYER VS PLAYER", font=get_font(33), base_color="#b68f40",
                            hovering_color="White")

        PvA_BUTTON = Button(image=pygame.image.load(media_dir + "Options Rect.png"), pos=(640, 400),
                            text_input="PLAYER vs ALPHAZERO", font=get_font(29), base_color="#b68f40",
                            hovering_color="White")

        BACK_BUTTON = Button(image=pygame.image.load(media_dir + "Options Rect.png"), pos=(640, 550),
                             text_input="BACK", font=get_font(50), base_color="#8B0000", hovering_color="White")

        MENU_TEXT = get_font(50).render(f"CHOOSE THE GAME MODE FOR {'ATAXX' if game_name=='A' else 'GO'}", True, "#d7fcd4")
        MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PvP_BUTTON, PvA_BUTTON, BACK_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PvP_BUTTON.checkForInput(MENU_MOUSE_POS):
                    boardSize(game_name, False)
                if PvA_BUTTON.checkForInput(MENU_MOUSE_POS):
                    boardSize(game_name, True)
                if BACK_BUTTON.checkForInput(MENU_MOUSE_POS):
                    gameName()

        pygame.display.update()


def gameName():
    pygame.display.set_caption("Menu")
    while True:
        MENU_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.blit(BG, (0, 0))

        ATAXX_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 250),
                              text_input="ATAXX", font=get_font(50), base_color="#b68f40", hovering_color="White")

        GO_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 400),
                           text_input="GO", font=get_font(50), base_color="#b68f40", hovering_color="White")

        QUIT_BUTTON = Button(image=pygame.image.load(media_dir + "Play Rect.png"), pos=(640, 550),
                             text_input="QUIT", font=get_font(50), base_color="#8B0000", hovering_color="White")

        MENU_TEXT = get_font(50).render("SELECT THE GAME", True, "#d7fcd4")
        MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [ATAXX_BUTTON, GO_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if ATAXX_BUTTON.checkForInput(MENU_MOUSE_POS):
                    gameMode("A")
                if GO_BUTTON.checkForInput(MENU_MOUSE_POS):
                    gameMode("G")
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()


if __name__ == '__main__':
    gameName()
