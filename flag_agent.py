import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame as pg

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Add
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer


def DeepQNetwork(lr, num_actions, input_dims):
    state_input = Input((input_dims,))

    backbone_1 = Dense(140, activation='relu')(state_input)
    backbone_2 = Dense(300, activation='relu')(backbone_1)
    backbone_3 = Dense(140, activation='relu')(backbone_2)

    value_output = Dense(1)(backbone_3)
    advantage_output = Dense(num_actions)(backbone_3)

    output = Add()([value_output, advantage_output])

    model = tf.keras.Model(state_input, output)
    model.compile(loss='mse', optimizer=Adam(lr))

    return model


class Agent:
    def __init__(self, lr, discount_factor, num_actions, epsilon, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.0001
        self.epsilon_final = 0.03
        self.update_rate = 120
        self.step_counter = 0
        self.tau = 0.01
        self.buffer0 = ReplayBuffer(batch_size * 10, input_dims)
        self.buffer1 = ReplayBuffer(batch_size * 10, input_dims)
        self.q_net0 = DeepQNetwork(lr, num_actions, input_dims)
        self.q_target_net0 = DeepQNetwork(lr, num_actions, input_dims)
        self.q_net1 = DeepQNetwork(lr, num_actions, input_dims)
        self.q_target_net1 = DeepQNetwork(lr, num_actions, input_dims)

    def store_tuple(self, state, action, reward, new_state, done, buffer):
        buffer.store_tuples(state, action, reward, new_state, done)

    def policy(self, q_net, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    def soft_update(self, q_net, target_net):
        for target_weights, q_net_weights in zip(target_net.weights, q_net.weights):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

    def train(self):
        if self.buffer0.counter < self.batch_size or self.step_counter % 10 != 0:
            self.step_counter += 1
            return
        self.train_network(self.q_net0, self.q_target_net0, self.buffer0)
        self.train_network(self.q_net1, self.q_target_net1, self.buffer1)
        
    def train_network(self, q_net, q_target_net, buffer):
        self.soft_update(q_net, q_target_net)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            buffer.sample_buffer(self.batch_size)

        q_predicted = q_net(state_batch)
        q_next = q_target_net(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.discount_factor*q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val
        q_net.train_on_batch(state_batch, q_target)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.step_counter += 1


    def train_model(self, env, num_episodes, graph, file=None, file_type=None):
        if file:
            self.load(file, file_type, env)
        render = True

        if render:
            env.ui.init_render()
        scores0, scores1, episodes, avg_scores, obj = [], [], [], [], []
        goal = 200
        f = 0
        steps = 0
        txt = open("saved_networks.txt", "w")
        train = False

        for i in range(num_episodes):
            if i % 100 == 0 and i != 0:
                self.save(f)
                f += 1
            done = False
            score = np.array([0.0, 0.0])
            state, _ = env.reset()
            while not done:
                for event in pg.event.get():
                        if event.type == pg.QUIT:
                            pg.quit()
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_s:
                                render = not render
                            if event.key == pg.K_0:
                                self.save(f)
                                f += 1
                if render:
                    env.render()
                action = (self.policy(self.q_net0, state), self.policy(self.q_net1, state))
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated
                score[0] += reward[0]
                score[1] += reward[1]
                # print("the reward is", reward)
                self.store_tuple(state, action[0], reward[0], new_state, done, self.buffer0)
                self.store_tuple(state, action[1], reward[1], new_state, done, self.buffer1)
                state = new_state
                if steps % 10 and train == 0:
                    self.train()
                steps += 1
            # if self.buffer.counter > self.buffer.size - 100:
            #         train = True
            scores0.append(score[0])
            scores1.append(score[1])
            obj.append(goal)
            episodes.append(i)
            avg_score0 = np.mean(scores0[-100:])
            avg_score1 = np.mean(scores1[-100:])
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score0: {4}, AVG Score1: {5}".format(i, num_episodes, score, self.epsilon,
                                                                             avg_score0, avg_score1))
        # if avg_score >= 200.0 and score >= 250:
        self.save(f)
        f += 1
        txt.write("Save {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(f, i, num_episodes,
                                                                                            score, self.epsilon,
                                                                                            ))
        txt.close()
        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores0, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('LunarLander_Train.png')

    def save(self, f):
        print("f:", f)
        self.q_target_net0.save(("saved_networks/flag_model_left{0}".format(f)))
        # self.q_target_net0.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(0)))
        self.q_target_net1.save(("saved_networks/flag_model_right{0}".format(f)))
        # self.q_target_net1.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(1)))

        print("Network saved")

    def load(self, file, file_type, env):
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.q_net.load_weights(file)

    def test(self, env, num_episodes, file_type, file, graph):
        clock = pg.time.Clock()
        env.ui.init_render()
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.q_net.load_weights(file)
        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0
        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                env.render()
                clock.tick(300)
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()

                action = 0

                if pg.key.get_pressed()[pg.K_LEFT]:
                    if pg.key.get_pressed()[pg.K_UP]:
                        action = 6
                    elif pg.key.get_pressed()[pg.K_DOWN]:
                        action = 4
                    else:
                        action = 5
                elif pg.key.get_pressed()[pg.K_RIGHT]:
                    if pg.key.get_pressed()[pg.K_UP]:
                        action = 8
                    elif pg.key.get_pressed()[pg.K_DOWN]:
                        action = 2
                    else:
                        action = 1
                elif pg.key.get_pressed()[pg.K_UP]:
                    action = 7
                elif pg.key.get_pressed()[pg.K_DOWN]:
                    action = 3

                ai_action = self.policy(self.q_net, state)
                new_state, reward, terminated, truncated, _ = env.step([action, ai_action])
                done = terminated
                # episode_score += reward
                state = new_state
            # score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('LunarLander_Test.png')

        env.close()
