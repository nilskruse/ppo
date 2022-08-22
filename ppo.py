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

        self.actor = nn.Sequential(
                nn.Linear(4, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 2))

        self.critic = nn.Sequential(
                nn.Linear(4, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1))

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

    env = gym.make("CartPole-v1")
    observation = env.reset()
    # collect rollout
    max_timesteps = 20000
    rollout_timesteps = 2048
    total_timesteps = 0
    while(total_timesteps < max_timesteps):
        # collect some data
        observations = []
        rewards = []
        actions = []
        action_probs = []
        values = []
        gaes = []
        dones = []

        for _ in range(rollout_timesteps):
            total_timesteps += 1
            policy.eval()
            with torch.no_grad():
                action_logits, action_sample, action_log_prob, _, value = policy(torch.from_numpy(observation).to(device))

            observations.append(torch.from_numpy(observation))
            actions.append(action_sample)
            action_probs.append(action_log_prob)
            values.append(value)

            act = action_sample.item()
            observation, reward, done, info = env.step(act)

            rewards.append(torch.from_numpy(np.array(reward)))
            dones.append(done)

            if done:
                observation = env.reset()

        _, _, _, _, value = policy(torch.from_numpy(observation).to(device))
        values.append(value)

        # compute advantage estimates
        gamma = 0.9
        gae_lambda = 1.0

        gae = 0

        for i in reversed(range(len(rewards))):
            done_factor = 1.0
            if dones[i]:
                done_factor = 0.0

            temp = gamma * values[i + 1] * done_factor - rewards[i]
            delta = rewards[i] + temp
            gae = delta + gamma * gae_lambda * done_factor * gae
            gaes.append(gae)
        gaes = gaes[::-1]


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

        # print(f"observations={observations}")
        # print(f"actions={actions}")
        print(f"Timesteps: {total_timesteps}")
        # optimizer = optim.Adam(policy.parameters(), lr=0.001)
        optimizer = optim.Adam([
            {'params': policy.actor.parameters(), 'lr': 0.0003},
            {'params': policy.critic.parameters(), 'lr': 0.001}])
        epsilon = 0.2
        for _ in range(10):
            optimizer.zero_grad()
            policy.train()
            _, _, new_probs, action_dist, value = policy(observations)
            # print(f"new_probs={new_probs}")
            entropy = action_dist.entropy().mean()

            ratio = (new_probs - action_probs).exp()
            # print(f"ratio={ratio}")
            # gaes = rewards - value.detach()
            # gaes = rewards2 - value.detach()
            surrogate1 = ratio * gaes
            surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * gaes

            actor_loss = - torch.min(surrogate1, surrogate2).mean()
            critic_loss = (reward - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            # print(f"critic_loss={critic_loss}")
            # print(f"actor_loss={actor_loss}")
            # print(f"loss={loss}")
            # print(f"entropy={entropy}")
            # print(f"value={value}")

            loss.backward()
            optimizer.step()

    ep_reward = 0.0
    # eval or something
    for _ in range(2000):
        env.render()
        # print(f"{policy(torch.from_numpy(observation))}")
        action_logits, action_sample, action_log_prob, _, value = policy(torch.from_numpy(observation).to(device))

        # print(f"action_sample={action_sample}")
        # print(f"action_log_prob={action_log_prob}")
        # print(f"act={action}")
        # print(f"value={value}")
        # print(f"action_logits={action_logits}")
        # act = F.softmax(action_logits.detach().cpu().numpy())
        act = F.softmax(action_logits)
        # print(f"act probs={act}")
        act = np.argmax(act.detach().cpu().numpy())
        # print(f"act={act}")
        observation, reward, done, info = env.step(act)
        ep_reward += reward

        if done:
            observation = env.reset()
            print(f"total reward={ep_reward}")
            ep_reward = 0
    env.close()
