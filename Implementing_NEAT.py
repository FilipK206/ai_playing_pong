import neat
import pygame
from pong_game import Game
import os

pygame.init()

game = Game(menu_screen=False)
game.run()

def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
