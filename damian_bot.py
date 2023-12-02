import pygame as pg
import math
import numpy as np
import random as rand

class RuleBot:
    def __init__(self, player_num, own_flag_pos, enemy_flag_pos, size, speed, screen):
        self.current_strat = rand.randrange(0, 2)
        self.middle = 1500
        self.player_num = player_num
        self.own_flag_pos = own_flag_pos
        self.enemy_flag_pos = enemy_flag_pos
        self.screen = screen
        self.size = size
        self.speed = speed
        self.buffer_default = 20
        self.buffer = self.buffer_default
        self.direction = None

        self.convert_dict = {
            (0, 0) : 0,
            (1, 0) : 1,
            (1, 1) : 2,
            (0, 1) : 3,
            (-1, 1) : 4,
            (-1, 0) : 5,
            (-1, -1) : 6,
            (0, -1) : 7,
            (1, -1) : 8,
        }
        
    def normalize(self, vector):
        normalized_vector = vector / np.linalg.norm(vector)
        return normalized_vector

    def choose_action(self, state):
        # state is [player1.pos[0], player1.pos[1], 
                        #  player2.pos[0], player2.pos[1], 
                        #  bool(self.flags[0].held), bool(self.flags[1].held), 
                        #  player1.tagged, player2.tagged]

        if self.player_num == 0:
            self_pos = np.array([state[0], state[1]])
            enemy_pos = np.array([state[2], state[3]])
            flag_stolen = state[5]
            flag_held = state[4]
            self_tagged = state[6]
            enemy_tagged = state[7]
        else:
            self_pos = np.array([state[2], state[3]])
            enemy_pos = np.array([state[0], state[1]])
            flag_stolen = state[4]
            flag_held = state[5]
            self_tagged = state[7]
            enemy_tagged = state[6]

        self.buffer = self.buffer_default

        if self_tagged > 0:
            self.current_strat = 0
        if enemy_tagged > 0:
            self.current_strat = 1
        if self_pos[0] > self.own_flag_pos[0] and self_pos[0] > self.middle:
            self.current_strat = 1
        elif self_pos[0] < self.own_flag_pos[0] and self_pos[0] < self.middle:
            self.current_strat = 1
        elif self_pos[0] > self.own_flag_pos[0] and self_pos[0] < self.middle:
            self.current_strat = 0
        elif self_pos[0] < self.own_flag_pos[0] and self_pos[0] > self.middle:
            self.current_strat = 0

        if self.current_strat == 0:
            action = self.defend(np.array(self_pos), np.array(enemy_pos), self.own_flag_pos, flag_stolen)
        elif self.current_strat == 1:
            action = self.attack(self_pos, enemy_pos, flag_stolen)

        return action

    def defend(self, defender_pos, attacker_pos, flag_pos, flag_held):
        if not flag_held:
            intercept_pos = self.calc_intercept(attacker_pos, defender_pos, flag_pos)
        else:
            intercept_pos = self.calc_intercept(attacker_pos, defender_pos, np.array([self.middle, attacker_pos[1]]))
        if intercept_pos.shape == (0,):
            intercept_pos = attacker_pos

        direction = [0, 0]

        movement_vector = intercept_pos - defender_pos


        # if abs(movement_vector[0]) > 1.4 * abs(movement_vector[1]):
        #     direction = [1, 0]
        # elif abs(movement_vector[1]) > 1.4 * abs(movement_vector[0]):
        #     direction = [0, 1]
        # else:
        #     direction = [1, 1]

        # if movement_vector[0] > 0:
        #     direction[1] *= -1
        # if movement_vector[1] < 0:
        #     direction[1] *= -1

        # if self.player_num == 1:
        #     direction[0] *= -1
        #     direction[1] *= -1

        if movement_vector[0] > 0:
            direction[0] = 1
        elif movement_vector[0] == 0:
            direction[0] = 0
        else:
            direction[0] = -1

        if movement_vector[1] > 0:
            direction[1] = 1
        elif movement_vector[1] == 0:
            direction[1] = 0
        else:
            direction[1] = -1

        action = self.convert_dict[tuple(direction)]
        return action

    def attack(self, pos, enemy_pos, flag_held):
        print("")
        if flag_held:
            target = np.array([self.middle, pos[1]])
        else:
            target = self.enemy_flag_pos
        attack_vect = target - pos
        if attack_vect[0] > 0:
            direction = [1, 0]
        else:
            direction = [-1, 0]

        if abs(attack_vect[1] * 1.2) > abs(attack_vect[0]):
            if attack_vect[1] > 0:
                direction[1] = 1
            else:
                direction[1] = -1

        if direction[0] == 1 and direction[1] == 1:
            axis_speed = self.speed * 0.7
        else:
            axis_speed = self.speed

        if self.check_danger((pos[0] + axis_speed * direction[0], pos[1] + axis_speed * direction[1]), enemy_pos):
            # try to dodge perpendicular
            print("perpendicular check")
            if abs(enemy_pos[1] - pos[1]) > abs(enemy_pos[0] - pos[0]):
                axis = 1
                not_axis = 0
            else:
                axis = 0
                not_axis = 1
            print("axis is {}".format(axis))
            direction[axis] = 0
            if direction[not_axis] == 0:
                direction[not_axis] = -1
            if not self.check_danger((pos[0] + self.speed * direction[0], pos[1] + self.speed * direction[1]), enemy_pos):
                print(direction)
                return self.convert_dict[tuple(direction)]
            direction[axis] *= -1
            if not self.check_danger((pos[0] + self.speed * direction[0], pos[1] + self.speed * direction[1]), enemy_pos):
                print(direction)
                return self.convert_dict[tuple(direction)]
            print("failed")
            
            # back up required then repeat check
            print("buffer reduced and back up")
            self.buffer = 0
            if enemy_pos[axis] > pos[axis]:
                direction[axis] = -1
            else:
                direction[axis] = 1
            axis_speed = self.speed * 0.7
            if not self.check_danger((pos[0] + axis_speed * direction[0], pos[1] + axis_speed * direction[1]), enemy_pos):
                print(direction)
                return self.convert_dict[tuple(direction)]
            direction[not_axis] *= -1
            if not self.check_danger((pos[0] + axis_speed * direction[0], pos[1] + axis_speed * direction[1]), enemy_pos):
                print(direction)
                return self.convert_dict[tuple(direction)]
            print("failed")
            
            # unpredictability?
            if self.direction == None:
                print("no direction")
                if rand.randrange(1, 6) == 1:
                    if enemy_pos[not_axis] > pos[not_axis]:
                        self.directoin = 1
                    else:
                        self.direction = -1
                direction[not_axis] = 0
            else:
                print("direction is {}".format(self.direction))
                direction[not_axis] = self.direction
            
            print(direction)
            
            return self.convert_dict[tuple(direction)]
        action = self.convert_dict[tuple(direction)]
        return action

    def check_danger(self, player1, player2):
        if abs(player1[0] - player2[0]) < self.size * 2 + self.speed + self.buffer:
            if abs(player1[1] - player2[1]) < self.size * 2 + self.speed + self.buffer:
                return True
        return False


    def calc_intercept(self, attack_pos, defend_pos, flag_pos):
        pg.draw.circle(self.screen, (0, 255, 0), (flag_pos), 20)
        distance_vector = attack_pos - defend_pos
        distance = math.sqrt(distance_vector[0]**2 + distance_vector[1]**2)
        speed = 1

        attack_movement_vector = self.normalize(np.array(flag_pos - attack_pos))
        attack_velocity = speed * attack_movement_vector

        # a = 0 because the speeds should be equal - also calculations kinda won't work if a is anything else...
        a = 0
        b = 2 * np.dot(attack_velocity, distance_vector)
        c = distance * -distance

        time = -c / b

        if time < 0:
            # print("intercept not possible :(")
            time *= -1
        
        intercept_pos = np.array([attack_pos[0] + attack_velocity[0] * time,
                                  attack_pos[1] + attack_velocity[1] * time])
        
        pg.draw.circle(self.screen, (255, 255, 0), intercept_pos, 20)
        
        return intercept_pos