"""Marks ``tests`` as a package.

This is load-bearing, not decoration. Without it pytest's default ``prepend``
import mode adds each test file's own directory to ``sys.path`` (``tests/unit``
for a module here, since there is no ``__init__.py`` to walk up through) and
never the project root, so anything importing across test modules by its full
path fails with "No module named 'tests'". ``test_search_diagnostics.py``
imports ``FakeEmbedder`` from ``test_semantic.py``, which is exactly that case.

With this file present pytest walks the module's basedir up to the project root
and imports each test as ``tests.unit.test_x``. That also means one module
object per file: the helper ``test_search_diagnostics`` imports is the same one
pytest collected, rather than a second copy under a different name.

It also makes ``pytest -q`` and ``python -m pytest`` agree. The latter puts the
working directory on ``sys.path`` by itself, so a cross-module import works
there and fails here, which is how this was missed.
"""
