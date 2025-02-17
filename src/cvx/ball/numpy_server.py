import threading

import loguru
import numpy as np
import pyarrow as pa
import pyarrow.flight as fl


class NumpyServer(fl.FlightServerBase):
    def __init__(self, host, port, logger=None, **kwargs):
        uri = f"grpc+tcp://{host}:{port}"
        super().__init__(uri, **kwargs)
        self._logger = logger or loguru.logger
        self._storage = {}  # Dictionary to store uploaded data
        self._lock = threading.Lock()  # Lock for thread safety

    @property
    def logger(self):
        return self._logger

    @staticmethod
    def _handle_arrow_table(table, logger) -> dict[str, np.ndarray]:
        # Directly work with the Arrow Table (no Polars)
        logger.info(f"Handling Arrow Table: {table}")
        logger.info(f"Names: {table.schema.names}")

        matrices = {}
        for name in table.schema.names:
            logger.info(f"Name: {name}")
            struct = table.column(name)[0].as_py()

            # Extract the matrix and shape data from the Arrow Table
            matrix_data = np.array(struct["data"])  # .to_numpy()  # Flattened matrix data
            shape = np.array(struct["shape"])  # .to_numpy()  # Shape of the matrix

            logger.info(f"Matrix (flattened): {matrix_data}")
            logger.info(f"Shape: {shape}")

            if len(matrix_data) != np.prod(shape):
                raise fl.FlightServerError("Data length does not match the provided shape")

            # Reshape the flattened matrix data based on the shape
            matrix = matrix_data.reshape(shape)
            logger.info(f"Reshaped Matrix: {matrix}")

            matrices[name] = matrix

        return matrices

    @staticmethod
    def _extract_command_from_ticket(ticket):
        """Helper method to extract the command from a Flight Ticket."""
        return ticket.ticket.decode("utf-8")

    def do_put(self, context, descriptor, reader, writer):
        with self._lock:
            # Read and store the data
            command = descriptor.command.decode("utf-8")
            self.logger.info(f"Processing PUT request for command: {command}")

            table = reader.read_all()
            self.logger.info(f"Table: {table}")

            # Store the table using the command as key
            self._storage[command] = table

            self.logger.info(f"Data stored for command: {command}")

        return fl.FlightDescriptor.for_command(command)

    def do_get(self, context, ticket):
        # Get the command from the ticket
        command = self._extract_command_from_ticket(ticket)
        self.logger.info(f"Processing GET request for command: {command}")

        # Retrieve the stored table
        if command not in self._storage:
            raise fl.FlightServerError(f"No data found for command: {command}")

        table = self._storage[command]
        self.logger.info(f"Retrieved data for command: {command}")

        matrices = NumpyServer._handle_arrow_table(table, logger=self.logger)
        result_table = self.f(matrices)

        self.logger.info("Computation completed. Returning results.")
        stream = fl.RecordBatchStream(result_table)

        return stream

    @classmethod
    def start(cls, port=5008, logger=None, **kwargs):
        logger = logger or loguru.logger  # pragma: no cover
        server = cls("127.0.0.1", port=port, logger=logger, **kwargs)  # pragma: no cover
        server.logger.info(f"Starting {cls} Flight server on port {port}...")  # pragma: no cover
        server.serve()  # pragma: no cover

    @classmethod
    def descriptor(cls):
        command = cls.__name__
        descriptor = fl.FlightDescriptor.for_command(command)
        return descriptor

    @classmethod
    def write(cls, client, data):
        descriptor = cls.descriptor()
        d = {key: pa.array([{"data": value.flatten(), "shape": value.shape}]) for key, value in data.items()}
        table = pa.table(d)
        writer, _ = client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()

    @classmethod
    def get(cls, client):
        ticket = fl.Ticket(cls.__name__)  # Create a Ticket with the command
        reader = client.do_get(ticket)
        result_table = reader.read_all()
        results = {name: result_table.column(name)[0].as_py() for name in result_table.schema.names}
        return results

    @classmethod
    def compute(cls, client, data):
        cls.write(client, data)
        return cls.get(client)

    @staticmethod
    def scalar(x):
        return pa.array([x], type=pa.float64())

    @staticmethod
    def vector(x):
        return pa.array([x], type=pa.list_(pa.float64()))

    @staticmethod
    def results_table(d):
        return pa.Table.from_pydict(d)
