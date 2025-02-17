import pyarrow as pa
from pyarrow.flight import FlightStreamReader


class TableReader(FlightStreamReader):
    """
    A Flight reader implementation that wraps a PyArrow Table.
    This allows a pa.Table to be used in Flight server implementations
    where a FlightStreamReader is expected.

    Args:
        table (pa.Table): The PyArrow Table to wrap.
    """

    def __init__(self, table: pa.Table) -> None:
        """
        Initialize the reader with a PyArrow Table.
        Converts the table into record batches for streaming access.

        Args:
            table (pa.Table): The source table to read from
        """
        self._table: pa.Table = table

    def read_all(self) -> pa.Table:
        """
        Read the entire table at once.
        Useful when you need all the data and don't want to stream it.

        Returns:
            pa.Table: The complete table
        """
        return self._table
