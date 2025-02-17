import pyarrow.flight as fl

from .utils.alter import np_2_pa, pa_2_np


class NumpyClient(fl.FlightClient):
    def __init__(self, client: fl.FlightClient):
        self._client = client

    @property
    def client(self):
        return self._client

    # @classmethod
    def descriptor(self, command):
        """
        Returns a Flight Descriptor for the server, using the class name as the command.

        :return: Flight Descriptor for the command.
        """
        # command = cls.__name__  # Use the class name as the command.
        descriptor = fl.FlightDescriptor.for_command(command)  # Create a descriptor for the command.
        return descriptor

    def write(self, command, data):
        """
        Write data to the client by creating a Flight Descriptor and sending the data.

        :param client: The Flight client.
        :param data: The data to send (in a dictionary format).
        """
        descriptor = self.descriptor(command)  # Create the Flight Descriptor.

        # Convert the data into an Arrow Table with the required structure.
        table = np_2_pa(data)

        # Send the data to the client using the descriptor.
        writer, _ = self.client.do_put(descriptor, table.schema)
        writer.write_table(table)  # Write the Arrow Table to the client.
        writer.close()  # Close the writer.

    def get(self, command):
        """
        Get the data from the client by issuing a GET request using the class descriptor.

        :param client: The Flight client.
        :return: A dictionary of results containing the requested data.
        """
        ticket = fl.Ticket(command)  # Create a Ticket using the class name as the command.
        reader = self.client.do_get(ticket)  # Send the GET request to the client.
        return reader.read_all()  # Read all results from the client.

    def compute(self, command, data):
        """
        Perform both write and get operations: send data to the client and retrieve results.

        :param client: The Flight client.
        :param data: The data to send to the client.
        :return: The results returned by the client after computation.
        """
        self.write(command, data)  # Write data to the client.
        return pa_2_np(self.get(command))
