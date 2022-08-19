import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

import numpy as np




class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.shared_net = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU()
                )

        self.actor = nn.Sequential(self.shared_net, nn.Linear(8, 2))
        self.critic = nn.Sequential(self.shared_net, nn.Linear(8, 1))

    def action(self, x):
        x = self.actor(x)

        action_dist = Categorical(logits=x)
        action_sample = action_dist.sample()
        action_log_prob = action_dist.log_prob(action_sample)

        return x, action_sample, action_log_prob, action_dist

    def value(self, x):
        return self.critic(x)

    def forward(self, x):
        action_logits, action_sample, action_log_prob, action_dist = self.action(x)
        return action_logits, action_sample, action_log_prob, action_dist, self.value(x)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic()
    policy.to(device)

    print("test")

    env = gym.make("CartPole-v1")
    observation = env.reset()
    # collect rollout
    max_timesteps = 512
    rollout_timesteps = 512
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
            with torch.no_grad():
                action_logits, action_sample, action_log_prob, _, value = policy(torch.from_numpy(observation).to(device))

            observations.append(torch.from_numpy(observation))
            actions.append(action_sample)
            action_probs.append(action_log_prob)
            values.append(value)

            act = action_sample.item()
            observation, reward, done, info = env.step(act)

            rewards.append(torch.from_numpy(np.array(reward)))

            if done:
                observation = env.reset()

        # compute advantage estimates
        gamma = 0.9
        lambd = 1.0
        gae = 0.0
        for i in reversed(range(rollout_timesteps - 1)):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae += np.power(gamma * lambd, rollout_timesteps - i + 1) * delta
            # print(f"gae={gae}")
            gaes.append(gae)

        gaes.reverse()
        # print(f"gaes={gaes}")
        # print(f"values={values}")

        # print(f"{observations}")
        # print(f"{rewards}")

        # now train that shit
        observations = torch.squeeze(torch.stack(observations)).detach().to(device)
        actions = torch.squeeze(torch.stack(actions)).detach().to(device)
        action_probs = torch.squeeze(torch.stack(action_probs)).detach().to(device)
        rewards = torch.squeeze(torch.stack(rewards)).detach().to(device)
        gaes = torch.squeeze(torch.stack(gaes)).detach().to(device)
        print(f"observations={observations}")
        print(f"Timesteps: {total_timesteps}")
        optimizer = optim.Adam(policy.parameters(), lr=0.0003)
        epsilon = 0.2
        for _ in range(10):
            for obs, action, old_probs, reward, advantage in zip(observations, actions, action_probs, rewards, gaes):
                _, _, new_probs, action_dist, value = policy(obs.to(device))
                entropy = action_dist.entropy().mean()

                ratio = (new_probs - old_probs).exp()
                # print(f"ratio={ratio}")
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage

                actor_loss = - torch.min(surrogate1, surrogate2).mean()
                critic_loss = (reward - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                # print(f"loss={loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    ep_reward = 0.0
    # eval or something
    for _ in range(200):
        env.render()
        # print(f"{policy(torch.from_numpy(observation))}")
        action_logits, action_sample, action_log_prob, _, value = policy(torch.from_numpy(observation).to(device))

        # print(f"action_sample={action_sample}")
        # print(f"action_log_prob={action_log_prob}")
        # print(f"act={action}")
        # print(f"value={value}")

        act = np.argmax(action_logits.detach().cpu().numpy())
        observation, reward, done, info = env.step(act)
        ep_reward += reward

        if done:
            observation = env.reset()
            print(f"total reward={ep_reward}")
            ep_reward = 0
    env.close()
