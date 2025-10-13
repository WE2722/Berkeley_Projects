import importlib
import importlib
import os
import sys

# Insert the 'search' folder on sys.path so modules written for the original
# top-level layout (which import `pacman`, `pacmanAgents`, etc.) resolve.
SEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SEARCH_DIR not in sys.path:
    sys.path.insert(0, SEARCH_DIR)


def test_planning_agent_bfs_finds_path():
    # Import pacmanAgents as a top-level module (it lives in the 'search' folder)
    m = importlib.import_module('pacmanAgents')
    assert hasattr(m, 'PlanningSmartAgent')


def test_smart_agent_importable():
    m = importlib.import_module('pacmanAgents')
    assert hasattr(m, 'SmartAgent')
