# db/queries.py
from sqlalchemy import select
from .connection import get_table

def get_all_data(table_name: str):
    """
    Fetch all rows from the given table.

    Args:
        table_name (str): Name of the table in the database.

    Returns:
        sqlalchemy.sql.Select: A select statement for the table.
    """
    table = get_table(table_name)  # Reflect the table
    stmt = select(table)           # SELECT * FROM table
    return stmt