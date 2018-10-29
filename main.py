# Augmented random search implementation for BipedalWalker-v2 environmnet

import gym
import numpy as np

EPOCH_NUMBER = 1000
EPISODE_LENGTH = 2000
LEARNING_RATE = 0.02
NUM_DELTAS = 16
NUM_BEST_DELTAS = 16
NOISE = 0.03
SEED = 1
LOG_EVERY = 1


# wrapper for environment
class EpisodeWorker():
    def __init__(self, env_name, normalizer=None):
        self.env_name = env_name
        self.env = gym.make(env_name)
        if not normalizer:
            self.normalizer = Normalizer(self.env.observation_space.shape[0])
        self.output_size = self.env.action_space.shape[0]
        self.input_size = self.env.observation_space.shape[0]

    def shape(self):
        return(self.output_size, self.input_size)

    def run_episode(self, weights, render=False):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < EPISODE_LENGTH:
            self.normalizer.observe(state)
            observation = self.normalizer.normalize(state)
            action = weights.dot(observation)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
            if render:
                self.env.render()
        return sum_rewards


class Normalizer():
    # Normalizer for inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Trainer():
    def __init__(self, env):
        self.env = env
        self.weights = np.zeros(self.env.shape())

    def produce_deltas(self):
        # create random noise
        return [np.random.randn(*self.env.shape()) for _ in range(NUM_DELTAS)]

    def train(self):
        # train model
        for epoch in range(EPOCH_NUMBER):
            deltas = self.produce_deltas()
            
            positive_rewards = []
            negative_rewards = []

            # get reward for positive and negative deltas
            for delta in deltas:
                positive_delta_weights = self.weights + NOISE * delta
                negative_delta_weights = self.weights - NOISE * delta
                positive_rewards.append(self.env.run_episode(positive_delta_weights))
                negative_rewards.append(self.env.run_episode(negative_delta_weights))
            
            # get std
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            rollouts = [{'r_pos': x[0], 'r_neg': x[1], 'delta': x[2]} for x in zip(positive_rewards, negative_rewards, deltas)]
            best_rollouts = sorted(rollouts, key = lambda x: max(x['r_pos'], x['r_neg']), reverse=True)[:NUM_BEST_DELTAS]

            # update weights
            step = np.zeros(self.env.shape())
            for r_pos, r_negr, delta in best_rollouts:
                step += (r_pos - r_negr) * delta
            self.weights += LEARNING_RATE / (NUM_BEST_DELTAS * sigma_rewards) * step

            # evaluate weights
            current_reward = self.env.run_episode(self.weights, render=True)

            print('Episode: {}, Reward: {}'.format(epoch, current_reward))


if __name__ == '__main__':
    np.random.seed(SEED)
    env = EpisodeWorker('BipedalWalker-v2')
    trainer = Trainer(env)

    trainer.train()