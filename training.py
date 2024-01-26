from AlphaZero import AlphaZero
from Games import Ataxx, Go
import multiprocessing
from sys import argv
import torch

multiprocessing.set_start_method('spawn', force=True)

args = {
    'training_verbose': False,
    'iterations_limit': 25000,
    'num_episodes': 100,
    'validation_episodes': 10,
    'learning_rate': 0.001,
    'num_mcts_sims': 50,
    'mcts_exploration_weight': 1.0,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'UCB_exploration_weight': 1.0,
    'batch_size': 2 ** 9,
    'best_model_dir': './best_model/',
    'device': None,
    'num_processes': None
}

if __name__ == "__main__":
    if len(argv) != 2:
        print('\n\nInvalid number of arguments. Usage: python training.py <GNxM>'
              '\nG: A(Ataxx),G (Go)'
              '\nN,M for Ataxx: (4x4, 5x5, 6x6)'
              '\nN,M for Go: (7x7, 9x9)')
        exit(0)

    game = argv[1][0]
    boardsize = (int(argv[1][1]), int(argv[1][3]))

    if game == 'A':
        if boardsize[0] == boardsize[1] == 4 \
                or boardsize[0] == boardsize[1] == 5 \
                or boardsize[0] == boardsize[1] == 6:
            g = Ataxx(boardsize[0], boardsize[1])
        else:
            print('\n\n For Ataxx Game: N,M must be 4x4, 5x5, 6x6.')
            exit(0)
    elif game == 'G':
        if (boardsize[0] == boardsize[1] == 7
                or boardsize[0] == boardsize[1] == 9):
            g = Go(boardsize[0], boardsize[1])
        else:
            print('\n\n For Go Game: N,M must be 7x7, 9x9.')
            exit(0)
    else:
        print(
            '\n\nInvalid Game. Usage: python training.py <GNxM> \nG:[A (Ataxx),G (Go)] \nG=A:(N,M = 4x4, 5x5 or 6x6), G=G:(N,M = 7x7 or 9x9)')
        exit(0)

    print("Training model for", "Ataxx" if game == "A" else "Go", "with board size", boardsize[0], "x", boardsize[1])

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
        args['num_processes'] = torch.cuda.get_device_properties(args['device']).multi_processor_count**2
        print("CUDA is available!")
        print(f"Name of the device: {torch.cuda.get_device_name(args['device'])}")
        print(f"Number of processes being used: {args['num_processes']}")
    else:
        args['device'] = torch.device('cpu')
        args['num_processes'] = int(multiprocessing.cpu_count())
        print("CUDA is not available! Using CPU.")
        print(f"Number of processes being used: {args['num_processes']}")

    model = AlphaZero(g, args)

    if model.load_model():
        print(f"Loading the model with {model.iter} iterations.")
    else:
        print("No model found. Starting from scratch.")

    print("Starting the learning process:")
    model.learn()
