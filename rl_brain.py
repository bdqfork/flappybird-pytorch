import random
from collections import deque

import torch
import torch.nn as nn

from network import DeepQNetwork

ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
# OBSERVE = 1000000
EXPLORE = 2000000
INITIAL_EPSILON = 0.1
# INITIAL_EPSILON = 0
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
# 神经网络参数替换间隔
REPLAYCE_INTERVAL = 500
SAVE_INTERVAL = 1000
PATH = 'saved_networks/dqn.pt'


class RL_Brain(object):
    def __init__(self):
        self.isTrain = True
        self.time_step = 0
        self.current_state = None
        self.memory = deque()
        self.epsilon = INITIAL_EPSILON
        self.build()

    def init_weights(self, model):
        if type(model) == nn.Conv2d and type(model) == nn.Linear:
            model.weight.data.normal_(0.0, 0.01)
        if type(model) == nn.Conv2d:
            model.bias.data.fill_(0.01)

    def build(self):
        input = 4
        output = ACTIONS
        self.q_eval = DeepQNetwork(input, output)
        self.q_eval.apply(self.init_weights)
        self.q_target = DeepQNetwork(input, output)
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=1e-6)
        self.load_model()

    def load_model(self):
        try:
            checkpoint = torch.load(PATH)
            self.q_eval.load_state_dict(checkpoint['q_eval_model_state_dict'])
            self.q_target.load_state_dict(
                checkpoint['q_target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss = checkpoint['loss']
            self.q_eval.train()
            self.q_target.train()
            print("Successfully loaded")
        except Exception:
            self.loss = nn.MSELoss()
            print("Could not find old network weights")

    def trainNetwork(self, epoch):
        if epoch % REPLAYCE_INTERVAL == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print('\ntarget_params_replaced\n')
        current_states, q_target = self.getTrainData()
        self.optimizer.zero_grad()
        outputs = self.q_eval(current_states)
        loss = self.loss(outputs, q_target)
        loss.backward()
        self.optimizer.step()
        # save network every 1000 iteration
        if epoch % SAVE_INTERVAL == 0:
            torch.save({'epoch': epoch,
                        'q_eval_model_state_dict': self.q_eval.state_dict(),
                        'q_target_model_state_dict': self.q_eval.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.loss}, PATH)
            print('save network')
        return loss

    def getTrainData(self):
        current_states, next_states, terminals, rewards, action_indexs = \
            self.getFeatrue()
        q_eval_next = self.q_eval(next_states)
        q_target_next = self.q_target(next_states)
        q_eval = self.q_eval(current_states)
        q_target = q_eval.clone()
        for i in range(0, BATCH):
            if terminals[i]:
                q_target[i][action_indexs[i]] = rewards[i]
            else:
                _, max_act_next = torch.max(q_eval_next[i], 0)
                q_target[i][action_indexs[i]] = rewards[i] + \
                    GAMMA * q_target_next[i][max_act_next]
        return current_states, q_target

    def getFeatrue(self):
        batch_memory = random.sample(self.memory, BATCH)
        current_states = torch.empty((BATCH, 4, 80, 80))
        next_states = torch.empty((BATCH, 4, 80, 80))
        terminals = []
        rewards = []
        action_indexs = []
        for i in range(0, BATCH):
            current_state = batch_memory[i][0]
            current_states[i] = current_state
            next_state = batch_memory[i][3]
            next_states[i] = next_state
            rewards.append(batch_memory[i][2])
            terminals.append(batch_memory[i][4])
            action = batch_memory[i][1]
            _, action_index = torch.max(action, -1)
            action_indexs.append(action_index)
        return current_states, next_states, terminals, rewards, action_indexs

    def getAction(self):
        action = torch.zeros(2)
        q_max = None
        if random.random() <= self.epsilon and self.isTrain:
            print("----------Random Action----------")
            action_index = random.randint(0, ACTIONS - 1)
            action[action_index] = 1
        else:
            q_eval = self.q_eval(self.current_state)[0]
            q_max, action_index = torch.max(q_eval, -1)
            action[action_index] = 1
        return action, q_max

    def setPerception(self, time_step, action, reward, observation, terminal):
        nextObservation = torch.cat(
            (observation, self.current_state[:, :3, :, :]), 1)
        self.memory.append((self.current_state, action,
                            reward, nextObservation, terminal))
        loss = None
        if len(self.memory) > REPLAY_MEMORY:
            self.memory.popleft()
        if time_step > OBSERVE and self.isTrain:
            loss = self.trainNetwork(time_step)
        self.current_state = nextObservation
        if self.epsilon > FINAL_EPSILON and time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return loss

    def setInitState(self, observation):
        observation = torch.from_numpy(
            observation).type(torch.FloatTensor)
        current_state = torch.stack(
            (observation, observation, observation, observation), 0)
        self.current_state = current_state.resize_((1, 4, 80, 80))
