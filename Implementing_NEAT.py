import neat
import pygame
from pong_game import Game
import os
import pickle

pygame.init()


class PongGame:
    def __init__(self):
        self.game = Game(menu_screen=False)
        self.left_paddle = self.game.opponent
        self.right_paddle = self.game.player
        self.ball = self.game.ball

    def loop(self):
        self.game.static_background()
        self.game.dynamic_background()
        self.game.draw_mov_obj()

        self.game.ball.move()
        self.game.ball.wall_collision()
        self.game.ball_collision()

        self.game.ball_reset_check()

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.player.move_up()
            if keys[pygame.K_s]:
                self.game.player.move_down()

            output = net.activate(
                (self.right_paddle.rect.y, self.ball.rect.y, abs(self.right_paddle.rect.x - self.ball.rect.x)))
            decision = output.index((max(output)))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.opponent.move_up()
            else:
                self.game.opponent.move_down()

            self.loop()
            pygame.display.flip()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output1 = net1.activate(
                (self.left_paddle.rect.y, self.ball.rect.y, abs(self.left_paddle.rect.x - self.ball.rect.x)))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.player.move_up()
            else:
                self.game.player.move_down()

            output2 = net2.activate(
                (self.right_paddle.rect.y, self.ball.rect.y, abs(self.right_paddle.rect.x - self.ball.rect.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.opponent.move_up()
            else:
                self.game.opponent.move_down()

            self.loop()
            pygame.display.flip()

            if self.game.score_player >= 1 or self.game.score_opponent >= 1 or self.game.player_lives <= 4:
                self.calculate_fitness(genome1, genome2, self.game)
                break

    def calculate_fitness(self, genome1, genome2, game):
        genome1.fitness += (game.lives - game.player_lives)
        genome2.fitness += (game.lives - game.opponent_lives)


def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0

        for genome_id2, genome2 in genomes[i + 1:]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            pong = PongGame()

            pong.train_ai(genome1, genome2, config)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
