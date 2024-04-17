# Pong Game AI Using NEAT

This project implements a Pong game with an AI agent trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI agent learns to play the game by evolving neural networks that control the paddles in response to the ball's movement.

## How It Works

The `run_neat.py` script uses the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train an AI agent to play the Pong game. It initializes a population of neural networks (genomes) with random weights and evolves them over generations to improve their performance. Each genome represents a potential AI player. During training, genomes compete against each other in Pong matches, where they learn to control paddles to hit the ball and score points. The training process evaluates genomes based on their performance in the game and assigns fitness scores accordingly. After training, the best-performing genome is saved to a file named `best.pickle`. The `test_ai` function loads the best genome from the file and tests it in a Pong game against a human player or another AI agent, allowing evaluation of the trained AI's performance.

## Files

- `pong_game.py`: Contains the implementation of the Pong game. The game logic, including paddle movement, ball physics, collision detection, and scoring, is implemented in this file.- `Implementing_NEAT.py`: Script to train the AI agent using NEAT.
- `best.pickle`: Binary file containing the best performing agent after training.
- `config.txt`: Configuration file for NEAT.

## Contributors

This project was created by Filip Kozal.
