"""
Dummy conftest.py for flex.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import os
import ray


@pytest.fixture
def set_test_dir():
    test_dir = os.path.dirname(__file__)
    ray.init(runtime_env={"working_dir": test_dir})
