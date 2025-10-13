# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
import random
import game
import util

import search
import searchAgents


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action)
                      for action in legal]
        scored = [(self.evaluationFunction(state), action)
                  for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


def scoreEvaluation(state):
    return state.getScore()


class SmartAgent(Agent):
    """
    Reactive evaluation-based agent for the reinforcement environment.
    """
    def __init__(self):
        pass

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.smartEvaluation(succ), action) for succ, action in successors]
        bestScore = max(scored, key=lambda x: x[0])[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

    def smartEvaluation(self, state):
        score = state.getScore()
        food = state.getFood()
        foodList = food.asList()
        if len(foodList) == 0:
            return float('inf')
        pacPos = state.getPacmanPosition()
        dists = [util.manhattanDistance(pacPos, f) for f in foodList]
        minFoodDist = min(dists) if dists else 0
        score -= 2.0 * minFoodDist
        score -= 10.0 * len(foodList)
        ghostStates = state.getGhostStates()
        for g in ghostStates:
            gpos = g.getPosition()
            dist = util.manhattanDistance(pacPos, gpos)
            if g.scaredTimer > 0:
                score += max(0, 5.0 - dist)
            else:
                if dist == 0:
                    score -= 500
                else:
                    score -= 20.0 / float(dist)
        return score


class PlanningSmartAgent(Agent):
    """Plans a shortest path to the nearest food using BFS or A* and follows it."""
    def __init__(self, **agentArgs):
        self.currentPath = []
        self.planner = agentArgs.get('planner', 'bfs')

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if self.currentPath:
            action = self.currentPath.pop(0)
            if action not in state.getLegalPacmanActions():
                self.currentPath = []
            else:
                return action

        problem = searchAgents.AnyFoodSearchProblem(state)
        if self.planner == 'astar':
            heuristic = lambda pos, prob=None: util.manhattanDistance(pos, prob.startState) if prob else 0
            path = search.aStarSearch(problem, heuristic)
        else:
            path = search.bfs(problem)
        if path is None or len(path) == 0:
            return Directions.STOP
        self.currentPath = list(path)
        return self.currentPath.pop(0)
