# AI for Self Driving Car

# Importing the libraries

import logging
import random
import os
import torch
import torch.nn as neural_network
import torch.nn.functional as functional
import torch.optim as optim
from torch.autograd import Variable

# Initialize logger
logger: logging.Logger = logging.getLogger('ai')


class Network(neural_network.Module):
    """
    Creating the architecture of the Neural Network
    """

    def __init__(self, input_size, num_actions):
        r"""Initializer for our neural network.

        Arguments:
            input_size (int, required): Number of input neurons.
            num_actions (int, required): Number of output neurons.
        """
        super(Network, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        # Connection between the input layer and the hidden layer
        self.full_conn_1 = neural_network.Linear(in_features=input_size,
                                                 out_features=30)
        # Connection between the hidden layer and the output layer
        self.full_conn_2 = neural_network.Linear(in_features=30,
                                                 out_features=num_actions)

    def forward(self, state):
        """Forward propagation, activates neurons, returns the Q-values
        for each possible action, depending on the input state."""
        # Rectifier function to activate hidden neurons
        x = functional.relu(self.full_conn_1(state))
        q_values = self.full_conn_2(x)
        return q_values


class ReplayMemory(object):
    """
    Implementing Experience Replay - Relates to Markov Decision Process
    """

    def __init__(self, capacity: int):
        # How many events (experience) to store in memory
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        """Appends new events to memory, and ensures that we always have
        a memory containing the max capacity of events."""
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # Delete the oldest memory
            del self.memory[0]

    def sample(self, batch_size: int):
        """Get random samples from memory."""
        # Zip function reshapes the input list or tuple
        samples = zip(*random.sample(self.memory, batch_size))
        # Torch variables that contain a tensor and a gradient. The gradient
        # represents a calculated loss of a leaf node in a computation graph.
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn:
    """
    Implementation for our Deep Q-Learning (Q = Quality)
    """

    def __init__(self, input_size: int, num_actions: int, gamma):
        # Delay coefficient
        self.gamma = gamma
        # Sliding window of the last 100 rewards
        self.reward_window = []
        self.model = Network(input_size, num_actions)
        self.memory = ReplayMemory(100000)
        # Used for a stochastic grid and data weights, LR = Learning Rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Last state is a tensor vector of 5 dimensions
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # Actions are 0, 1 or 2, see also: Map.action2rotation
        self.last_action = 0
        # Float number between 0.0 and 1.0
        self.last_reward: float = 0.0

    def __str__(self):
        return "Deep Q Network :: Last State: %s, " \
               "Last Action: %s, Last Reward: %s" % \
               (self.last_state, self.last_action, self.last_reward)

    def select_action(self, tensor_state: torch.Tensor):
        r"""Uses the soft max function to select the next action.

        Arguments:
            tensor_state (required): Torch tensor output state of our neural network.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("state: %s" % tensor_state)

        # Probability distribution
        # Temperature decreases values for low probabilities, and increases values
        # for high probabilities, effectively increasing action certainty
        temperature = 150  # T=150
        probs = functional.softmax(
            self.model(Variable(tensor_state, volatile=True)) * temperature, dim=1)
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    # Stochastic gradient descent
    def learn(self, batch_state: torch.Tensor, batch_next_state: torch.Tensor,
              batch_reward: torch.Tensor, batch_action: torch.Tensor):
        """
        From the AI Handbook:
        We get the prediction, then the target, then the loss. Then we back propagate
        the loss error and update the weights according to how much they contributed
        to the error.

        :param batch_state: Current state (Tensor, tuple).
        :param batch_next_state: Next state.
        :param batch_reward: The reward.
        :param batch_action: The action from memory.
        """
        prediction_outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        # Temporal difference (Tensor)
        temporal_diff_loss = functional.smooth_l1_loss(prediction_outputs, target)
        self.optimizer.zero_grad()
        # td_loss.backward(retain_graph=True)
        temporal_diff_loss.backward()
        self.optimizer.step()

    def update(self, reward: float, new_signal: list) -> torch.Tensor:
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))

        action = self.select_action(new_state)

        # If we have at least 100 memories, it is time to start learning
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self) -> float:
        # Check if list is empty (prevent divide by zero arithmetic error)
        if not self.reward_window:
            return 0.0
        return sum(self.reward_window) / len(self.reward_window)

    def save(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, 'last_brain.pth')

    def load(self):
        # Check if file exists
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found...")
