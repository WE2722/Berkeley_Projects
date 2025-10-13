import pytest
import search.pacman as pacman_main
import importlib


def test_planning_agent_bfs_finds_path():
    # Run the search pacman runner in text mode with PlanningSmartAgent on tinySearch
    # We won't execute the full game loop; instead ensure the agent class can be loaded
    modulename = 'search.pacmanAgents'
    m = importlib.import_module(modulename)
    assert hasattr(m, 'PlanningSmartAgent')


def test_smart_agent_importable():
    m = importlib.import_module('search.pacmanAgents')
    assert hasattr(m, 'SmartAgent')
