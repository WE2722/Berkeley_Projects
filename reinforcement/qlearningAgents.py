# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Initialize Q-values as a Counter (dictionary with default 0)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Return Q-value for (state, action) pair
        # Counter returns 0.0 by default if key doesn't exist
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        
        # Return 0.0 if no legal actions (terminal state)
        if not legalActions:
            return 0.0
        
        # Return maximum Q-value over all legal actions
        maxQValue = float('-inf')
        for action in legalActions:
            qValue = self.getQValue(state, action)
            maxQValue = max(maxQValue, qValue)
        
        return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        
        # Return None if no legal actions
        if not legalActions:
            return None
        
        # Find all actions with maximum Q-value
        maxQValue = self.computeValueFromQValues(state)
        bestActions = []
        
        for action in legalActions:
            if self.getQValue(state, action) == maxQValue:
                bestActions.append(action)
        
        # Break ties randomly
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        
        # Return None if no legal actions
        if not legalActions:
            return None
        
        # Epsilon-greedy action selection
        if util.flipCoin(self.epsilon):
            # Explore: choose random action
            action = random.choice(legalActions)
        else:
            # Exploit: choose best action according to Q-values
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Q-learning update rule:
        # Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
        
        # Get current Q-value
        currentQ = self.getQValue(state, action)
        
        # Get maximum Q-value for next state
        maxNextQ = self.computeValueFromQValues(nextState)
        
        # Compute new Q-value using the update rule
        sample = reward + self.discount * maxNextQ
        newQ = currentQ + self.alpha * (sample - currentQ)
        
        # Update Q-value
        self.qValues[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Get features for this (state, action) pair
        features = self.featExtractor.getFeatures(state, action)
        
        # Compute Q-value as dot product of weights and features
        # Q(s,a) = sum_i f_i(s,a) * w_i
        qValue = 0.0
        for feature, value in features.items():
            qValue += self.weights[feature] * value
        
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        # Get features for current (state, action) pair
        features = self.featExtractor.getFeatures(state, action)
        
        # Compute difference (same as in Q-learning)
        # difference = (R + gamma * max_a' Q(s',a')) - Q(s,a)
        currentQ = self.getQValue(state, action)
        maxNextQ = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * maxNextQ) - currentQ
        
        # Update each weight:
        # w_i <- w_i + alpha * difference * f_i(s,a)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass