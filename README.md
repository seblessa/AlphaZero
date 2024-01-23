
# AlphaZero
Assignment for Laboratory of Artificial Intelligence and Data Science Class, 3º Year,1º Semester, Bachelor in Artificial Intelligence and Data Science Project 2 – Develop an Alpha Zero Ataxx Player

# Summary

In this project our goal is to implement an AlphaZero player for the game Ataxx and for the game Go.

The AlphaZero algorithm is a reinforcement learning algorithm
that uses a neural network to approximate the value function and the policy function.
The neural network is trained using self-play and Monte Carlo Tree Search.

**Autores**:
- [Alexandre Marques](https://github.com/AlexandreMarques27)
- [Sebastião Santos Lessa](https://github.com/seblessa/)
- [Margarida Vila Chã](https://github.com/margaridavc/)
- [João Nunes](https://github.com/JoaoNunes20)


# Versões

The versions of the operating systems used to develop and test this application are:
- Fedora 38
- macOS Sonoma 14.0
- Windows 11

Python Versions:
- 3.12.0


# Requirements

To keep everything organized and simple, we will use MiniConda to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n LabIACD python pytorch::pytorch torchvision torchaudio -c pytorch
```
To install the requirements run:

```bash
pip install -r requirements.txt
```

# Usage

There are two usaged modes for this project:

- Training:

```bash
python3 training.py <GNxM>
```

Where *G* is the game (*A* for Ataxx, *G* for Go and *P* for Player (Player vs Player mode))
and *NxM* is the board size.

For Ataxx, the board size can be 4x4, 5x5 or 6x6. For Go, the board size can be 7x7 or 9x9.

- Playing (Testing):

```bash
python3 play.py
```