import pygame
from pong_game import Game

pygame.init()

game = Game(menu_screen=False)
game.run()