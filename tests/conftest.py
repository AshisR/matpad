"""Shared fixtures for MatPad test suite."""
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport

from backend.main import app


@pytest.fixture
def identity2():
    return np.eye(2).tolist()

@pytest.fixture
def mat2x2():
    return [[1.0, 2.0], [3.0, 4.0]]

@pytest.fixture
def mat2x2_b():
    return [[5.0, 6.0], [7.0, 8.0]]

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
