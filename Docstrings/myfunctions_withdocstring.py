def create_table_definition(df, table_name):
    """This function takes a pandas DataFrame and a table name as input and generates a table definition string based on the columns of the DataFrame. The table definition string is formatted in sqlite SQL syntax and includes the table name and column names.
	
	Parameters:
	- df: pandas DataFrame containing the data for which the table definition is to be created
	- table_name: Name of the table for which the definition is to be created
	
	Returns:
	- table_definition: A string containing the table definition in sqlite SQL syntax
	
	Example    """
    # We need to tell GPT what the table structure looks like before it can understand the schema enough to create a SQL query
    table_definition = '''Given the following sqlite SQL definition, write queries based on the request
### sqlite SQL table, with its properties:
#
# {}({})
#
'''.format(table_name, ",".join(x for x in df.columns))
    return table_definition


def dataframe_to_database(df, table_name):
    """This function takes a pandas DataFrame and a table name as input, and creates a temporary in-memory database using SQLite. It then pushes the DataFrame into the database with the specified table name. The function returns the temporary database created.
	
	Parameters:
	- df: pandas DataFrame
	- table_name: str
	
	Returns:
	- temp_db: SQLAlchemy engine object
	
	Example Usage:
	df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    """
    # Create Temp DB in RAM
    temp_db = create_engine(f'sqlite:///:memory:', echo=False) # use SQL lite which is built in with SQL Alchemy

    # Push Pandas DF -> Temp DB
    df.to_sql(name=table_name, con=temp_db, index=False)
    return temp_db
