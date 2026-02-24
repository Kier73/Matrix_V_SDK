import pytest

@pytest.fixture
def sizes():
    """Default matrix sizes for stress testing."""
    return [64, 128, 256]


