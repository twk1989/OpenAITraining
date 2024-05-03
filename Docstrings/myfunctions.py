from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import text
import numpy as np

def dataframe_to_database(df, table_name):
    # Create Temp DB in RAM
    temp_db = create_engine(f'sqlite:///:memory:', echo=False) # use SQL lite which is built in with SQL Alchemy

    # Push Pandas DF -> Temp DB
    df.to_sql(name=table_name, con=temp_db, index=False)
    return temp_db

def create_table_definition(df, table_name):
    # We need to tell GPT what the table structure looks like before it can understand the schema enough to create a SQL query
    table_definition = '''Given the following sqlite SQL definition, write queries based on the request
### sqlite SQL table, with its properties:
#
# {}({})
#
'''.format(table_name, ",".join(x for x in df.columns))
    return table_definition