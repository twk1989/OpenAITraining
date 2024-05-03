from openai import OpenAI
from ctransformers import AutoModelForCausalLM # pip install ctransformers[cuda]>=0.2.24
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd

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

if __name__ == "__main__":
    client = OpenAI()

    # Take tabular data and convert to SQL database.
    df = pd.read_csv("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\NLP-to-SQL\\sales_data_sample.csv")
    # df = pd.read_json("C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\NLP-to-SQL\\registers_sample.json")
    # registers_data = df['Content']['Registers']
    # registers_df = pd.json_normalize(registers_data)
    # df = registers_df.drop('BitFields', axis=1)
    # table_name = "Registers"

    table_name = "Sales"
    database = dataframe_to_database(df, table_name)

    # Perform SQL query on Temp DB
    # with database.connect() as conn:
    #     # makes the connection
    #     # run code indentation/block
    #     # auto close connection -> we can't keep the connection to this RAM database open all the time
    #     # result = conn.execute(text("SELECT * FROM Sales"))
    #     # result = conn.execute(text("SELECT SUM(SALES) FROM Sales"))
    #     result = conn.execute(text("SELECT QTR_ID, SUM(SALES) AS TOTAL_SALES FROM Sales GROUP BY QTR_ID;"))
    # print(result.all())

    # Convert NLP to SQL via OpenAI API
    table_definition = create_table_definition(df, table_name)
    
    # Take natural language request.
    user_input = input("Tell OpenAi what you want to know about the data: ")

    prompt = table_definition + f"### A query to answer: {user_input}\nSELECT"
    print(prompt)

    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0,
    #     max_tokens=150,
    #     stop=[";"]
    # )
    # query = response.choices[0].message.content

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model_file = "C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\llama-2-7b.Q4_0.gguf"
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGUF",
                                                 model_file=model_file,
                                                 model_type="llama",
                                                 gpu_layers=0,
                                                 temperature=0,
                                                 max_new_tokens=150,
                                                 stop=[";"])
    # model.config.temperature=0
    # model.config.max_new_tokens=150
    # model.config.stop=[";"]
    query = model(prompt)

    print(query)

    if query.startswith(" "):
        query = "SELECT"+query

    # Send SQL request to database
    with database.connect() as conn:
        # makes the connection
        # run code indentation/block
        # auto close connection -> we can't keep the connection to this RAM database open all the time
        # result = conn.execute(text("SELECT * FROM Sales"))
        result = conn.execute(text(query))
    print(result.all())