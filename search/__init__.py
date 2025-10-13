"""
Package shim for the `search` package.

Some modules in this project do `import search` expecting the symbols
defined in `search/search.py` to be available at module-level. When the
`search` directory is a package (has an __init__.py), `import search` will
bind to this package, not the `search.py` module. To remain compatible we
re-export the public names from the `search.py` module here.
"""
from importlib import import_module
from pathlib import Path
__all__ = []

# Attempt to import the internal module `search.search` and re-export its
# public names so `import search` behaves like the original module import.
try:
	_mod = import_module(__name__ + '.search')
	for _name in dir(_mod):
		if not _name.startswith('_'):
			globals()[_name] = getattr(_mod, _name)
			__all__.append(_name)
except Exception:
	# If import fails (e.g., during partial installs), keep package minimal.
	pass
