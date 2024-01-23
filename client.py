from AlphaZero import AlphaZeroPlayer, CNNET
from Games import Go, Ataxx
from training import args
import socket
import time


def connect_to_server(host='172.17.30.148', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    response = client_socket.recv(1024).decode()
    print(f"Server ResponseINIT: {response}")

    g = response[-4:]  # A4x4

    game_name = g.split("x")[0][0]
    n = int(g.split("x")[1])
    if game_name == "A":
        Game = Ataxx(n, n)
    else:
        Game = Go(n, n)

    print(f"Playing: {'Ataxx' if game_name == 'A' else 'Go'} with a board {n}x{n}")
    if "1" in response:
        my_turn = True
    else:
        my_turn = False

    model = CNNET(Game, args)
    model.load_model(f"./best_model/{Game.model_name}/model.tar")
    player = AlphaZeroPlayer(model)

    while True:
        print("My turn:", my_turn)
        if my_turn:
            time.sleep(1)
            action = player.get_action(Game)
            if Game.name == "G":
                if action == Game.getActionSize() - 1:
                    message = "PASS"
                    move = Game.getActionSize() - 1
                else:
                    move = Game.decode(action)
                    message = "MOVE " + str(move[0]) + " " + str(move[1])
            else:
                move = Game.decode(action)
                message = "MOVE " + str(move[0][0]) + " " + str(move[0][1]) + " " + str(move[1][0]) + " " + str(
                    move[1][1])

            client_socket.send(message.encode())
            print("Send:", message)

            # Wait for server response
            response = client_socket.recv(1024).decode()
            print(f"Server Response1: {response}")
            if "END" in response:
                print("Game ended!")
                break
            if response == "VALID".replace(" ", ""):
                if Game.name == "G":
                    if message == "PASS":
                        Game.apply_action(move)
                    else:
                        move = int(move[0]), int(move[1])
                        Game.apply_action(Game.encode(move))
                else:
                    move = (int(move[0][0]), int(move[0][1])), (int(move[1][0]), int(move[1][1]))
                    Game.apply_action(Game.encode(move[0], move[1]))
                my_turn = False
            else:
                my_turn = True
                print("Invalid move!")
        else:
            response = client_socket.recv(1024).decode()
            print(f"Server Response2: {response}")
            if "END" in response:
                print("Game ended!")
                break
            response = response.split(" ")
            if Game.name == "G":
                move = int(response[-2]), int(response[-1])
                Game.apply_action(Game.encode(move))
            else:
                move = (int(response[-4]), int(response[-3])), (int(response[-2]), int(response[-1]))
                Game.apply_action(Game.encode(move[0], move[1]))
            print("Received:", move)
            my_turn = True

    client_socket.close()


if __name__ == "__main__":
    connect_to_server()
