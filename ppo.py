import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.input_layer = nn.Linear(4, 8)
        self.hidden_layer1 = nn.Linear(8, 8)
        self.hidden_layer2 = nn.Linear(8, 8)

        self.logits_action = nn.Linear(8, 2)

        self.value_layer = nn.Linear(8, 1)

    def action(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer1(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.logits_action(x)

        action_dist = Categorical(logits=x)
        action_sample = action_dist.sample()
        action_log_prob = action_dist.log_prob(action_sample)

        return x, action_sample, action_log_prob

    def value(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer1(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.value_layer(x)

        return x

    def forward(self, x):
        action_logits, action_sample, action_log_prob = self.action(x)
        return action_logits, action_sample, action_log_prob, self.value(x)


if __name__ == "__main__":

    policy = ActorCritic()

    print("test")

    env = gym.make("CartPole-v1")
    observation = env.reset()
    # collect rollout 
    max_timesteps = 128
    rollout_timesteps = 64
    total_timesteps = 0
    while(total_timesteps < max_timesteps):
        # collect some data
        observations = []
        rewards = []
        actions = []
        action_probs = []
        values = []
        gaes = []

        for _ in range(rollout_timesteps):
            total_timesteps += 1
            action_logits, action_sample, action_log_prob, value = policy(torch.from_numpy(observation))

            observations.append(observation)
            actions.append(action_sample.item())
            action_probs.append(action_log_prob.item())
            values.append(value)

            act = action_sample.item()
            observation, reward, done, info = env.step(act)

            rewards.append(reward)

            if done:
                observation = env.reset()

        # compute advantage estimates
        gamma = 1.0
        lambd = 1.0
        for i in reversed(range(rollout_timesteps - 1)):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = np.power(gamma * lambd, rollout_timesteps - i + 1)


        # print(f"{observations}")
        print(f"{rewards}")



    # eval or something
    for _ in range(1):
        env.render()
        print(f"{policy(torch.from_numpy(observation))}")
        action_logits, action_sample, action_log_prob, value = policy(torch.from_numpy(observation))

        print(f"action_sample={action_sample}")
        print(f"action_log_prob={action_log_prob}")
        # print(f"act={action}")
        print(f"value={value}")

        act = np.argmax(action_logits.detach().numpy())
        observation, reward, done, info = env.step(act)

        if done:
            observation = env.reset()
    env.close()
