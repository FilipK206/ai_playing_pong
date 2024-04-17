import neat
import pygame
from pong_game import Game
import os

pygame.init()

class PongGame:
    def __init__(self):
        self.game = Game(menu_screen=False)
        self.left_paddle = self.game.opponent
        self.right_paddle = self.game.player
        self.ball = self.game.ball

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

            self.game.run()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

    def eval_genomes(self, genomes, config):
        for i, (genome_id1, genome1) in enumerate(genomes):
            if i == len(genomes) - 1:
                break
            genome1.fitness = 0

            for genome_id2, genome2 in genomes[i+1:]:
                genome2.fitness = 0 if genome2.fitness == None else genome2.fitness

                self.train_ai(genome1, genome2, config)



    def run_neat(self, config):
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(1))

        winner = p.run(self.eval_genomes, 50)

    if __name__ == "__main__":
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        run_neat(config)
