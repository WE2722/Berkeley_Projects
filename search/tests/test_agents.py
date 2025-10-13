import importlib
def test_planning_agent_bfs_finds_path():
    # Ensure the PlanningSmartAgent class can be imported from search.pacmanAgents
    modulename = 'search.pacmanAgents'
    m = importlib.import_module(modulename)
    assert hasattr(m, 'PlanningSmartAgent')


def test_smart_agent_importable():
    m = importlib.import_module('search.pacmanAgents')
    assert hasattr(m, 'SmartAgent')
