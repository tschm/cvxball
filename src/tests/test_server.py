import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.flight as fl
import pytest

from cvx.ball.server import BallServer  # Adjust to your actual import path

from .utils.reader import TableReader


@pytest.fixture(scope="module")
def server():
    """Fixture to initialize and properly kill the BallServer for each test."""
    server = BallServer("127.0.0.1", 5007)

    # Function to run the server in a separate thread
    def run_server():
        server.serve()

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server a moment to start
    time.sleep(1)

    yield server

    # After the test, ensure the server is properly cleaned up
    server_thread.join(timeout=5)  # Ensure the server thread has time to shutdown


@pytest.fixture(scope="module")
def client(server):
    """Fixture to initialize and properly kill the BallServer for each test."""
    # Connect to the server (flight client)
    flight_client = fl.connect("grpc+tcp://127.0.0.1:5007")

    yield flight_client  # Provide the flight client to the test

    # After the test, ensure the server is properly cleaned up
    flight_client.close()  # Close the client connection


@pytest.fixture
def mock_table():
    """Fixture to create a mock Arrow Table for tests."""
    return np.array([[1, 2], [3, 4]])


def test_client(client, mock_table):
    # Simulate a 'do_put' request
    BallServer.write(client, {"input": mock_table})
    results = BallServer.get(client)
    assert results["radius"] == pytest.approx(1.4142135605902473)
    assert results["midpoint"] == pytest.approx(np.array([2.0, 3.0]))


def test_compute(client, mock_table):
    results = BallServer.compute(client, {"input": mock_table})
    assert results["radius"] == pytest.approx(1.4142135605902473)
    assert results["midpoint"] == pytest.approx(np.array([2.0, 3.0]))


def test_do_put_server(server, mock_table):
    command = "compute_ball"
    descriptor = fl.FlightDescriptor.for_command(command)

    reader = TableReader(mock_table)

    server.do_put(None, descriptor, reader, None)


def test_do_get_server(server, mock_table):
    command = BallServer.__name__
    descriptor = BallServer.descriptor()

    data = {"input": mock_table}
    table = pa.table({key: pa.array([{"data": value.flatten(), "shape": value.shape}]) for key, value in data.items()})

    # fill the storage for the correct command
    reader = TableReader(table)
    server.do_put(None, descriptor, reader, None)

    # from the ticket we can extract the correct storage
    ticket = fl.Ticket(command)
    server.do_get(None, ticket)


def test_wrong_command(server, mock_table):
    command = BallServer.__name__
    descriptor = BallServer.descriptor()

    # fill the storage for the correct command
    data = {"input": mock_table}
    table = pa.table({key: pa.array([{"data": value.flatten(), "shape": value.shape}]) for key, value in data.items()})

    reader = TableReader(table)
    server.do_put(None, descriptor, reader, None)

    with pytest.raises(fl.FlightServerError):
        command = "Dunno"
        # from the ticket we can extract the correct storage
        ticket = fl.Ticket(command)
        server.do_get(None, ticket)


def test_faulty_data(server, mock_table):
    command = BallServer.__name__
    descriptor = BallServer.descriptor()

    # fill the storage for the correct command
    data = {"input": mock_table}
    table = pa.table({key: pa.array([{"data": value.flatten(), "shape": [1, 1]}]) for key, value in data.items()})

    reader = TableReader(table)
    server.do_put(None, descriptor, reader, None)

    with pytest.raises(fl.FlightServerError):
        # from the ticket we can extract the correct storage
        ticket = fl.Ticket(command)
        server.do_get(None, ticket)
