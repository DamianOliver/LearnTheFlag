from flag_agent import Agent
import gym
import tensorflow as tf
from flag_env import Env

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

env = Env()
# env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
train = 1
test = 1
num_episodes = 10000
graph = True

file_type = 'tf'
file = 'saved_networks/flag_model_right9'

dqn_agents = Agent(lr=0.00075, discount_factor=0.99, num_actions=9, epsilon=1.00, batch_size=2048, input_dims=8)

if train and not test:
    dqn_agents.train_model(env, num_episodes, graph, None, None)
else:
    dqn_agents.test(env, num_episodes, file_type, file, graph)
