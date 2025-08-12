"""Tests for the Flight server connection.

This module contains a simple test to verify that the Flight server
is accessible and responding to requests.
"""

import pyarrow.flight as flight


def test_mini() -> None:
    """Test the connection to the Flight server.

    This test attempts to connect to the Flight server and retrieve
    flight information. It prints the result of the connection attempt
    but does not assert any specific outcomes, making it more of a
    diagnostic test than a validation test.

    Note:
        This test is primarily for diagnostic purposes and does not
        fail if the connection is unsuccessful.
    """
    # Connect to the Flight server using grpc+tls
    client = flight.FlightClient("grpc+tls://cvxball-710171668953.us-central1.run.app:443")

    # Test the connection
    try:
        flight_info: flight.FlightInfo | None = client.get_flight_info(flight.FlightDescriptor.for_path(b"test"))
        print("Connection successful!")
        print(flight_info)
    except flight.FlightUnavailableError as e:
        print("Connection failed!")
        print(e)
    except Exception as e:
        print("An error occurred:")
        print(e)
