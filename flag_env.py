import pygame as pg
import gym
import numpy as np
from random import randrange
from damian_bot import RuleBot

BACKGROUND_COLOR = ((140 * 0.1, 140 * 0.1, 255 * 0.1), (255 * 0.1, 140 * 0.1, 140 * 0.1))
SCREEN_SIZE = (3000, 1500)
TAGGED_TIME = 600
FLAG_SIZE = (35, 50)
SPEED = SCREEN_SIZE[0] // 700
SIZE = SCREEN_SIZE[0] // 200


class Env(gym.Env):
    def __init__(self):
        self.teams = [[Player([SCREEN_SIZE[0] // 9, SCREEN_SIZE[1] // 2], SPEED, SIZE, (255 * 0.1, 40 * 0.1, 40 * 0.1))], 
                      [Player([SCREEN_SIZE[0] - SCREEN_SIZE[0] // 9 - FLAG_SIZE[0], SCREEN_SIZE[1] // 2], SPEED, SIZE, (40 * 0.1, 40 * 0.1, 255 * 0.1))]]
        
        # self.flag1_image = pg.image.load("LearnTheFlag/Images/red_flag.png")
        # self.flag2_image = pg.image.load("LearnTheFlag/Images/blue_flag.png")

        self.flag1_image = pg.image.load("Images/red_flag.png")
        self.flag2_image = pg.image.load("Images/blue_flag.png")

        self.flag1_image = pg.transform.scale(self.flag1_image, FLAG_SIZE) 
        self.flag2_image = pg.transform.scale(self.flag2_image, FLAG_SIZE)

        self.flags = [Flag((SCREEN_SIZE[0] // 20, SCREEN_SIZE[1] // 2), FLAG_SIZE, self.flag1_image), 
                      Flag((SCREEN_SIZE[0] - SCREEN_SIZE[0] // 20, SCREEN_SIZE[1] // 2), FLAG_SIZE, self.flag2_image)]
        self.ui = Ui()

        self.action_space = gym.spaces.Discrete(9, start=0)


        self.observation_space = gym.spaces.Box(
        low = -100,
        high = 1400,
        shape = (8,),
        dtype = 'uint8'
        )

    def return_state(self):
        player1 = self.teams[0][0]
        player2 = self.teams[1][0]
        return np.array([player1.pos[0], player1.pos[1], 
                         player2.pos[0], player2.pos[1], 
                         bool(self.flags[0].held), bool(self.flags[1].held), 
                         player1.tagged, player2.tagged])

    def step(self, action):
        player0 = self.teams[0][0]
        player1 = self.teams[1][0]
        done = False
        reward = [0, 0]
        first_pos = [(player0.pos[0], player0.pos[1]), (player1.pos[0], player1.pos[1])]
        player0.move(action[0])
        player1.move(action[1])
        new_pos = [(player0.pos[0], player0.pos[1]), (player1.pos[0], player1.pos[1])]

        goal = []

        if self.flags[1].held == None:
            if self.flags[0].held == None:
                if abs(self.flags[1].pos[0] - first_pos[0][0]) > abs(self.flags[1].pos[0] - new_pos[0][0]):
                    reward[0] += 1
                else:
                    reward[0] -= 1
                if abs(self.flags[1].pos[1] - first_pos[0][1]) >= abs(self.flags[1].pos[1] - new_pos[0][1]):
                    reward[0] += 1
                else:
                    reward[0] -= 1
        else:
            if new_pos[0][0] < first_pos[0][0]:
                reward[0] += 1
            else:
                reward[0] -= 1
        if self.flags[0].held == None:
            if self.flags[1].held == None:
                if abs(self.flags[0].pos[0] - first_pos[1][0]) > abs(self.flags[0].pos[0] - new_pos[1][0]): 
                    reward[1] += 1
                else:
                    reward[1] -= 1
                if abs(self.flags[0].pos[1] - first_pos[1][1]) >= abs(self.flags[0].pos[1] - new_pos[1][1]):
                    reward[1] += 1
                else:
                    reward[1] -= 1
        else:
            if new_pos[1][0] > first_pos[1][0]:
                reward[1] += 1
            else:
                reward[1] -= 1

        for flag in self.flags:
            flag.move()
        tagged, flagged = self.check_collisions()
        if tagged == player0 or flagged == player1:
            reward[0] -= 1000
            reward[1] += 1000
        elif tagged == player1 or flagged == player0:
            reward[1] -= 1000
            reward[0] += 1000
        
        self.update_tagged()
        win = self.check_for_win()
        if win != 2:
            done = True
            reward[win] += 3000
            
        # position1, position2, flag1held, flag2held, player1jail, player2jail
        state = self.return_state()

        # print(state)
        # print(reward)
        # print("----------------------")
        # print()

        return state, reward, done, None, None

    def render(self):
        self.ui.draw(self.teams[0] + self.teams[1], self.flags)
        pg.display.update()

    def reset(self):
        self.teams = [[Player([SCREEN_SIZE[0] // 9, SCREEN_SIZE[1] // 2], SPEED, SIZE, (255, 40, 40))], 
                      [Player([SCREEN_SIZE[0] - SCREEN_SIZE[0] // 9, SCREEN_SIZE[1] // 2], SPEED, SIZE, (40, 40, 255))]]
        self.flags = [Flag((SCREEN_SIZE[0] // 20, SCREEN_SIZE[1] // 2), FLAG_SIZE, self.flag1_image), 
                      Flag((SCREEN_SIZE[0] - SCREEN_SIZE[0] // 20, SCREEN_SIZE[1] // 2), FLAG_SIZE, self.flag2_image)]
        
        state = self.return_state(), None
        return state

    def check_collision(self, object1, object2):
        if abs(object1.pos[0] - object2.pos[0]) < object1.size + object2.size:
            if abs(object1.pos[1] - object2.pos[1]) < object1.size + object2.size:
                return True
        return False
    
    def check_flag_collision(self, player, flag):
        if player.pos[0] > flag.pos[0] - player.size and abs(player.pos[0] - flag.pos[0]) < player.size + flag.size[0]:
            if player.pos[1] > flag.pos[1] - player.size and abs(player.pos[1] - flag.pos[1]) < player.size + flag.size[1]:
                return True
        return False
            
    def check_collisions(self):
        tagged = None
        flagged = None
        for player1 in self.teams[0]:
            for player2 in self.teams[1]:
                if self.check_collision(player1, player2):
                    tagged = self.handle_tag(player1, player2)

        for player in self.teams[0]:
            if self.flags[1].held == None:
                if self.check_flag_collision(player, self.flags[1]):
                    self.handle_pickup(player, self.flags[1])
                    flagged = player
        for player in self.teams[1]:
            if self.flags[0].held == None:
                if self.check_flag_collision(player, self.flags[0]):
                    self.handle_pickup(player, self.flags[0])
                    flagged = player

        return tagged, flagged

    def handle_tag(self, player_left, player_right):
        if player_left.pos[0] > SCREEN_SIZE[0] / 2:
            self.handle_tagged(player_left)
            return player_left
        else:
            self.handle_tagged(player_right)
            return player_right

    def handle_tagged(self, player):
        player.tagged = TAGGED_TIME
        player.pos = [player.reset_pos[0], player.reset_pos[1]]

        # pretty ugly but probably fine...
        for flag in self.flags:
            if flag.held == player:
                flag.held = None
                flag.pos = [flag.reset_pos[0], flag.reset_pos[1]]

    def handle_pickup(self, player, flag):
        flag.held = player

    def update_tagged(self):
        for team in self.teams:
            for player in team:
                player.tagged = max(0, player.tagged - 1)

    def check_for_win(self):
        for flag in self.flags:
            if flag.held != None:
                half = SCREEN_SIZE[0] // 2
                if flag.reset_pos[0] > half:
                    if flag.pos[0] < half:
                        print("winner0!")
                        return 0
                elif flag.pos[0] > half:
                    print("winner1!")
                    return 1
        return 2
                
class Player():
    def __init__(self, pos, speed, size, color):
        self.pos = pos
        self.reset_pos = (pos[0], pos[1])
        self.color = color
        self.speed = speed
        self.screen_dimension = SCREEN_SIZE
        self.size = size
        self.tagged = 0

        self.convert_dict = {
            0 : (0, 0),
            1 : (1, 0),
            2 : (1, 1),
            3 : (0, 1),
            4 : (-1, 1),
            5 : (-1, 0),
            6 : (-1, -1),
            7 : (0, -1),
            8 : (1, -1)
        }

    def move(self, direction):
        if self.tagged > 0:
            return
        dir_vector = self.convert_dict[direction]
        if dir_vector[0] and dir_vector[1]:
            axis_speed = self.speed * 0.7
        else:
            axis_speed = self.speed
        self.pos[0] += dir_vector[0] * axis_speed
        self.pos[1] += dir_vector[1] * axis_speed

        self.pos[0] = min(self.screen_dimension[0] - self.size, (max(self.pos[0], self.size)))
        self.pos[1] = min(self.screen_dimension[1] - self.size, (max(self.pos[1], self.size)))

class Flag():
    def __init__(self, pos, size, image):
        self.pos = pos
        self.reset_pos = (pos[0], pos[1])
        self.size = size
        self.held = None
        self.image = image

    def move(self):
        if self.held:
            self.pos = self.held.pos


class Ui():
    def __init__(self):
        self.screen = None

    def init_render(self):
        self.screen = pg.display.set_mode((SCREEN_SIZE), pg.RESIZABLE)
        self.draw_background()

    def draw(self, player_list, flag_list):
        self.draw_background()
        for player in player_list:
            self.draw_player(player)
        for flag in flag_list:
            self.draw_flag(flag)
        
    def draw_player(self, player):
        if player.tagged > 0:
            gradient = (TAGGED_TIME - player.tagged) / TAGGED_TIME * 255
            color = (gradient, gradient, gradient)
        else:
            color = player.color
        pg.draw.circle(self.screen, color, player.pos, player.size)

    def draw_flag(self, flag):
        if flag.held == None:
            self.screen.blit(flag.image, flag.pos)
        else:
            self.screen.blit(flag.image, (flag.pos[0], flag.pos[1] - flag.size[1]))

    def draw_background(self):
        pg.draw.rect(self.screen, BACKGROUND_COLOR[1], ((0, 0), (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1])))
        pg.draw.rect(self.screen, BACKGROUND_COLOR[0], ((SCREEN_SIZE[0] // 2, 0), (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1])))
        # self.screen.fill(BACKGROUND_COLOR)


if __name__ == "__main__":

    env = Env()

    env.ui.init_render()
    env.render()
    state, _ = env.reset()
    # pause = input("pause")
    clock = pg.time.Clock()

    rule_bot = RuleBot(1, env.flags[1].pos, env.flags[0].pos, SIZE, SPEED, env.ui.screen)
    bot = True

    player0 = env.teams[0][0]
    player1 = env.teams[1][0]

    while True:
        clock.tick(150)
        action = [0, 0]
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.K_p:
                print("pause")
                unpause = False
                while True:
                    if unpause:
                        print("unpause")
                        break
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            pg.quit()
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_p:
                                unpause = True
                                break
        if pg.key.get_pressed()[pg.K_1]:
            if pg.key.get_pressed()[pg.K_r]:
                player0.size = min(player0.size + 3, 700)
            elif pg.key.get_pressed()[pg.K_o]:
                player1.size = min(player1.size + 3, 700)
            if pg.key.get_pressed()[pg.K_e]:
                player0.size = max(2, player0.size-3)
            elif pg.key.get_pressed()[pg.K_p]:
                player1.size = max(2, player1.size-3)

        if pg.key.get_pressed()[pg.K_a]:
            if pg.key.get_pressed()[pg.K_w]:
                action[0] = 6
            elif pg.key.get_pressed()[pg.K_s]:
                action[0] = 4
            else:
                action[0] = 5
        elif pg.key.get_pressed()[pg.K_d]:
            if pg.key.get_pressed()[pg.K_w]:
                action[0] = 8
            elif pg.key.get_pressed()[pg.K_s]:
                action[0] = 2
            else:
                action[0] = 1
        elif pg.key.get_pressed()[pg.K_w]:
            action[0] = 7
        elif pg.key.get_pressed()[pg.K_s]:
            action[0] = 3


        if pg.key.get_pressed()[pg.K_LEFT]:
            if pg.key.get_pressed()[pg.K_UP]:
                action[1] = 6
            elif pg.key.get_pressed()[pg.K_DOWN]:
                action[1] = 4
            else:
                action[1] = 5
        elif pg.key.get_pressed()[pg.K_RIGHT]:
            if pg.key.get_pressed()[pg.K_UP]:
                action[1] = 8
            elif pg.key.get_pressed()[pg.K_DOWN]:
                action[1] = 2
            else:
                action[1] = 1
        elif pg.key.get_pressed()[pg.K_UP]:
            action[1] = 7
        elif pg.key.get_pressed()[pg.K_DOWN]:
            action[1] = 3

        if bot:
            action[1] = rule_bot.choose_action(state)

        state, reward, done, _, _ = env.step(action)
        # print("reward:", reward)
        # if done:
        #     env.reset()
        #     done = False
        if done:
            env.reset()
            pg.time.wait(1000)
            player0 = env.teams[0][0]
            player1 = env.teams[1][0]
        env.render()
