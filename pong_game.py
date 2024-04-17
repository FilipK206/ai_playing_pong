import pygame
import sys
import random
import math

class Ball:
    def __init__(self, x, y, size, speed, screen_height, screen_width):
        self.rect = pygame.Rect(x, y, size, size)
        self.speed = speed
        self.speed_x = speed * random.choice([-1, 1])
        self.speed_y = speed * random.choice([-1, 1])
        self.screen_height = screen_height
        self.screen_width = screen_width

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

    def reset(self):
        self.rect.center = (self.screen_width / 2, self.screen_height / 2)
        self.speed_x = self.speed * random.choice([-1, 1])
        self.speed_y = self.speed * random.choice([-1, 1])

    def wall_collision(self):
        if self.rect.top <= 0 or self.rect.bottom >= self.screen_height:
            self.speed_y *= -1
class Paddle:
    def __init__(self, x, y, width, height, speed, screen_height):
        self.rect = pygame.Rect(x, y, width, height)
        self.speed = speed
        self.screen_height = screen_height

    def move_up(self):
        self.rect.y -= self.speed

    def move_down(self):
        self.rect.y += self.speed

    def limit_movement(self):
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self.screen_height:
            self.rect.bottom = self.screen_height

class MainMenu:
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.title_font = pygame.font.SysFont("Bahnschrift", 45, bold=False)
        self.begin_font = pygame.font.SysFont("Bahnschrift", 25, bold=False)

        self.background_color = pygame.Color('grey12')
        self.light_grey = (200, 200, 200)

    def static_background(self):
        self.screen.fill(self.background_color)
        pygame.draw.aaline(self.screen, self.light_grey, (self.screen_width / 2, 0),
                           (self.screen_width / 2, self.screen_height))
        pygame.draw.ellipse(self.screen, self.light_grey,
                            [self.screen_width / 2 - 110, self.screen_height / 2 - 110, 220, 220], 2)

    def display_menu(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    run = False

            self.static_background()

            begin_label = self.title_font.render("Welcome to the Pong Game!", 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2 - 270))

            begin_label = self.begin_font.render("Press the mouse to begin...", 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2 - 180))

            pygame.display.update()

class ResultMenu:
    def __init__(self, screen, screen_width, screen_height, title_result, text_result, score_player, score_opponent, padding=50):
        self.screen = screen

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.title_font = pygame.font.SysFont("Bahnschrift", 45, bold=False)
        self.end_font = pygame.font.SysFont("Bahnschrift", 25, bold=False)

        self.title_result = title_result
        self.text_result = text_result

        self.score_player = score_player
        self.score_opponent = score_opponent

        self.background_color = pygame.Color('grey12')

        self.padding = padding

    def display_menu(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(self.background_color)

            begin_label = self.title_font.render(self.title_result, 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2 - (2 * self.padding)))

            begin_label = self.end_font.render(self.text_result, 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2 - self.padding))

            begin_label = self.end_font.render(f"Your score: {self.score_player}", 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2))

            begin_label = self.end_font.render(f"Your opponent score: {self.score_opponent}", 1, (255, 255, 255))
            self.screen.blit(begin_label, (self.screen_width / 2 - begin_label.get_width() / 2, self.screen_height / 2 + self.padding))

            pygame.display.update()

class Game:
    def __init__(self):
        self.screen_width = 1080
        self.screen_height = 720

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Pong')

        self.timer_event = pygame.event.custom_type()
        pygame.time.set_timer(self.timer_event, 1000)
        self.clock = pygame.time.Clock()
        self.game_time_sec = 60

        self.main_font = pygame.font.SysFont("Bahnschrift", 15, bold=False)
        self.timer_font = pygame.font.SysFont("Bahnschrift", 25, bold=False)

        self.background_color = pygame.Color('grey12')
        self.light_grey = (200, 200, 200)

        self.ball = Ball(self.screen_width / 2 - 10, self.screen_height / 2 - 10, 20, 8, self.screen_height, self.screen_width)
        self.player = Paddle(self.screen_width - 25, self.screen_height / 2 - 70, 10, 140, 9, self.screen_height)
        self.opponent = Paddle(15, self.screen_height / 2 - 70, 10, 140, 9, self.screen_height)

        self.score_player = 0
        self.score_opponent = 0

        self.player_lives = 5
        self.opponent_lives = 5

        self.ball_speed = 8
        self.opponent_speed = 8

    def draw_mov_obj(self):
        pygame.draw.rect(self.screen, self.light_grey, self.player.rect)
        pygame.draw.rect(self.screen, self.light_grey, self.opponent.rect)

        pygame.draw.ellipse(self.screen, self.light_grey, self.ball.rect)

    def static_background(self):
        self.screen.fill(self.background_color)
        pygame.draw.aaline(self.screen, self.light_grey, (self.screen_width / 2, 0), (self.screen_width / 2, self.screen_height))
        pygame.draw.ellipse(self.screen, self.light_grey, [self.screen_width / 2 - 110, self.screen_height / 2 - 110, 220, 220], 2)

    def dynamic_background(self):
        score_player_label = self.main_font.render(f"Score Player: {self.score_player}", 1, self.light_grey)
        score_opponent_label = self.main_font.render(f"Score Opponent: {self.score_opponent}", 1, self.light_grey)
        self.screen.blit(score_player_label, (self.screen_width - 150, 0 + 15))
        self.screen.blit(score_opponent_label, (0 + 50, 0 + 15))

        lives_player_label = self.main_font.render(f"Lives: {self.player_lives}", 1, self.light_grey)
        lives_opponent_label = self.main_font.render(f"Lives: {self.opponent_lives}", 1, self.light_grey)
        self.screen.blit(lives_player_label, (self.screen_width - 150, 0 + 35))
        self.screen.blit(lives_opponent_label, (0 + 50, 0 + 35))

        timer_label = self.timer_font.render(f"Timer: {self.game_time_sec} sec", 1, self.light_grey)
        self.screen.blit(timer_label, (self.screen_width / 2 - 70, 0 + 20))

    def ball_paddle_collision(self, ball, paddle):
        paddle_collision_point = ball.rect.y + ball.rect.height / 2 - (paddle.rect.y + paddle.rect.height / 2)
        normalized_collision_point = paddle_collision_point / (paddle.rect.height / 2)
        reflection_angle = normalized_collision_point * (math.pi / 4)
        ball.speed_x = -abs(ball.speed)
        ball.speed_y = ball.speed * math.sin(reflection_angle)

    def ball_collision(self):
        if self.ball.rect.colliderect(self.player.rect):
            self.ball_paddle_collision(self.ball, self.player)
            self.score_player += 1

        if self.ball.rect.colliderect(self.opponent.rect):
            self.ball_paddle_collision(self.ball, self.opponent)
            self.ball.speed_x *= -1
            self.ball.speed_y *= -1
            self.score_opponent += 1

    def opponent_movement(self):
        target_y = self.ball.rect.y + self.ball.rect.height / 2
        current_y = self.opponent.rect.centery

        if current_y < target_y:
            self.opponent.rect.centery += min(self.ball_speed, target_y - current_y)
        elif current_y > target_y:
            self.opponent.rect.centery -= min(self.ball_speed, current_y - target_y)

        self.opponent.limit_movement()

    def player_movement(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.player.move_up()
        if keys[pygame.K_DOWN]:
            self.player.move_down()
        self.player.limit_movement()

    def ball_reset_check(self):
        if self.ball.rect.left <= - 45:
            self.ball.reset()
            self.opponent_lives -= 1
        if self.ball.rect.right >= self.screen_width + 45:
            self.ball.reset()
            self.player_lives -= 1

    def run(self):
        main_menu = MainMenu(self.screen, self.screen_width, self.screen_height)
        main_menu.display_menu()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == self.timer_event:
                    self.game_time_sec -= 1

            self.static_background()
            self.dynamic_background()
            self.draw_mov_obj()

            self.ball.move()
            self.ball.wall_collision()
            self.ball_collision()

            self.opponent_movement()
            self.player_movement()

            self.ball_reset_check()

            pygame.display.flip()

            self.clock.tick(60)

            if self.player_lives < 0:
                result_menu = ResultMenu(self.screen, self.screen_width, self.screen_height, "You have lost.",
                                         "Try next time to be better.", self.score_player, self.score_opponent)
                result_menu.display_menu()
                break

            if self.opponent_lives < 0 or (self.game_time_sec < 0 and self.score_player > self.score_opponent):
                result_menu = ResultMenu(self.screen, self.screen_width, self.screen_height, "You have won!",
                                         "You are the best!", self.score_player, self.score_opponent)
                result_menu.display_menu()
                break

            elif self.game_time_sec < 0 and self.score_player < self.score_opponent:
                result_menu = ResultMenu(self.screen, self.screen_width, self.screen_height, "You have lost.",
                                         "Try next time to be better.", self.score_player, self.score_opponent)
                result_menu.display_menu()
                break

            elif self.game_time_sec < 0 and self.score_player == self.score_opponent:
                result_menu = ResultMenu(self.screen, self.screen_width, self.screen_height, "It is a draw.",
                                         "Try next time to be better.", self.score_player, self.score_opponent)
                result_menu.display_menu()
                break

if __name__ == "__main__":
    pygame.init()
    game = Game()
    game.run()
