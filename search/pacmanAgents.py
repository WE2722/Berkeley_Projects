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
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

def scoreEvaluation(state):
    return state.getScore()


class SmartAgent(Agent):
    """A reactive evaluation-based agent with tunable weights.

    Agent options (passed via -a, e.g. -a foodWeight=2,ghostPenalty=20):
      - foodWeight: multiplier for distance-to-food term (default 2.0)
      - remainWeight: multiplier for remaining-food-count term (default 10.0)
      - ghostPenalty: base penalty for ghost proximity (default 20.0)
      - scaredBonus: reward when near scared ghosts (default 5.0)
    """

    def __init__(self, **agentArgs):
        self.foodWeight = float(agentArgs.get('foodWeight', 2.0))
        self.remainWeight = float(agentArgs.get('remainWeight', 10.0))
        self.ghostPenalty = float(agentArgs.get('ghostPenalty', 20.0))
        self.scaredBonus = float(agentArgs.get('scaredBonus', 5.0))

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Score each successor using the evaluation function
        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.smartEvaluation(succ), action) for succ, action in successors]
        bestScore = max(scored, key=lambda x: x[0])[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

    def smartEvaluation(self, state):
        """Return a numeric score for a state. Higher is better."""
        # Base score from game
        score = state.getScore()

        # Food info
        food = state.getFood()
        foodList = food.asList()
        if len(foodList) == 0:
            return float('inf')  # winning state

        # Distance to closest food (manhattan)
        pacPos = state.getPacmanPosition()
        dists = [util.manhattanDistance(pacPos, f) for f in foodList]
        minFoodDist = min(dists) if dists else 0
        # prefer closer food (scaled)
        score -= self.foodWeight * minFoodDist
        # prefer fewer remaining pellets (scaled)
        score -= self.remainWeight * len(foodList)

        # Ghosts: penalize being near non-scared ghosts
        ghostStates = state.getGhostStates()
        for g in ghostStates:
            gpos = g.getPosition()
            dist = util.manhattanDistance(pacPos, gpos)
            if g.scaredTimer > 0:
                # encourage chasing scared ghosts mildly
                score += max(0, self.scaredBonus - dist)
            else:
                # strong penalty for being too close
                if dist == 0:
                    score -= 500
                else:
                    score -= (self.ghostPenalty) / float(dist)

        return score


class PlanningSmartAgent(Agent):
    """
    A planning agent that computes a shortest path to the nearest food using
    the search machinery (AnyFoodSearchProblem + BFS) and then follows that
    path until it's exhausted, at which point it replans.

    Usage: python pacman.py -p PlanningSmartAgent
    """
    def __init__(self):
        self.currentPath = []
        self.planner = 'bfs'

    def __init__(self, **agentArgs):
        """agentArgs may contain 'planner'= 'bfs' or 'astar'"""
        self.currentPath = []
        self.planner = agentArgs.get('planner', 'bfs')

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # If we have a planned path, follow it while it's still valid
        if self.currentPath:
            action = self.currentPath.pop(0)
            # If action is no longer legal, clear plan and replan
            if action not in state.getLegalPacmanActions():
                self.currentPath = []
            else:
                return action

        # Need to plan: use AnyFoodSearchProblem from searchAgents
        problem = searchAgents.AnyFoodSearchProblem(state)
        if self.planner == 'astar':
            # Use a simple Manhattan heuristic wrapper
            heuristic = lambda pos, prob=None: util.manhattanDistance(pos, prob.startState) if prob else 0
            path = search.aStarSearch(problem, heuristic)
        else:
            path = search.bfs(problem)
        if path is None or len(path) == 0:
            return Directions.STOP
        self.currentPath = list(path)
        return self.currentPath.pop(0)
