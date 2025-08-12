"""Tests for the BallServer class."""

from typing import Any

import numpy as np
import pytest

from cvx.ball.server import BallServer  # Adjust import path as needed


@pytest.fixture(scope="module")
def server() -> BallServer:
    """Fixture providing a BallServer instance.

    Returns:
        BallServer: An initialized BallServer instance for testing.
    """
    return BallServer(host="localhost", port=5008)


@pytest.fixture
def sample_points() -> np.ndarray:
    """Fixture providing sample 2D points for testing.

    Returns:
        np.ndarray: A 2D array of points forming a square.
    """
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


@pytest.fixture
def expected_ball_square(sample_points: np.ndarray) -> dict[str, Any]:
    """Fixture providing expected results for square points.

    Args:
        sample_points: The sample points used for testing.

    Returns:
        Dict[str, Any]: A dictionary containing the expected radius, midpoint, and points.
    """
    return {
        "radius": 0.7071067811865476,  # √2/2
        "midpoint": np.array([0.5, 0.5]),
        "points": sample_points,
    }


def test_initialization(server: BallServer) -> None:
    """Test BallServer initialization.

    Args:
        server: The BallServer instance to test.

    Verifies:
        The server is properly initialized with an empty storage dictionary.
    """
    assert isinstance(server, BallServer)
    assert server._storage == {}


def test_compute_ball_square(server: BallServer, expected_ball_square: dict[str, Any]) -> None:
    """Test ball computation for points forming a square.

    Args:
        server: The BallServer instance to test.
        expected_ball_square: The expected results for the square points.

    Verifies:
        The computed ball matches the expected radius, midpoint, and contains all points.
    """
    # Prepare input data
    input_data: dict[str, np.ndarray] = {"input": expected_ball_square["points"]}

    # Compute result
    np_dict: dict[str, Any] = server.f(input_data)

    np.testing.assert_allclose(np_dict["points"], expected_ball_square["points"], atol=1e-5)
    np.testing.assert_allclose(np_dict["radius"], expected_ball_square["radius"], rtol=1e-5)
    np.testing.assert_allclose(np_dict["midpoint"], expected_ball_square["midpoint"], rtol=1e-5)


def test_invalid_input_empty(server: BallServer) -> None:
    """Test handling of empty input.

    Args:
        server: The BallServer instance to test.

    Verifies:
        The server raises a ValueError when given an empty points array.
    """
    # Empty points array
    empty_points: np.ndarray = np.array([]).reshape(0, 2)

    # Test that it raises an appropriate error
    with pytest.raises(ValueError):  # Replace with specific exception type
        server.f({"input": empty_points})


def test_invalid_input_wrong_key(server: BallServer) -> None:
    """Test handling of input with wrong dictionary key.

    Args:
        server: The BallServer instance to test.

    Verifies:
        The server raises a KeyError when the input dictionary has the wrong key.
    """
    # Points with wrong dictionary key
    points: np.ndarray = np.array([[0, 0], [1, 1]])

    with pytest.raises(KeyError):
        server.f({"wrong_key": points})
