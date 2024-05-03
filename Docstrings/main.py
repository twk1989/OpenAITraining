from openai import OpenAI
import inspect
import myfunctions
from pathlib import Path

def hello(name):  
    print(f"Hello {name}")

def docstring_prompt(code):
    prompt = f"{code}\n # A high quality python docstring of the above Python function:\n \"\"\""
    return prompt

def merge_docstring_and_function(original_function, docstring):
    # first line of the function (def myfunc():)
    # """
    # Docstring (response from completion API)
    # """
    # Rest of the function
    split = original_function.split('\n')
    first_part,second_part = split[0],split[1:]
    # docstring = '\t'.join(docstring.splitlines(True))
    # merged_function = first_part + '\n    """\n' + docstring + '    """\n' + '\n'.join(second_part)
    merged_function = first_part + "\n" + '    """' + docstring + '    """' + "\n" + "\n".join(second_part)
    return merged_function

def get_all_functions(module):
    return [mem for mem in inspect.getmembers(module, inspect.isfunction)
         if mem[1].__module__ == module.__name__]

# def get_all_functions(module):
#     with open(module.__file__, 'r') as f:
#         source = f.readlines()

#     functions = []
#     current_function = ""
#     in_function = False
#     for line in source:
#         if line.strip().startswith("def "):
#             if current_function:
#                 functions.append(current_function)
#             current_function = line
#             in_function = True
#         elif in_function:
#             current_function += line
#             if line.strip() == "":
#                 in_function = False
#     if current_function:
#         functions.append(current_function)
#     return functions

client = OpenAI()

functions_to_prompt = myfunctions
# all_funcs = get_all_functions(myfunctions)
# Get all functions defined in the 'functions_to_prompt' module
all_funcs = []

# Iterate over all members of the module 'functions_to_prompt'
for member in inspect.getmembers(functions_to_prompt, inspect.isfunction):
    # Check if the member is defined in the 'functions_to_prompt' module
    if member[1].__module__ == functions_to_prompt.__name__:
        # If yes, add it to the list of all functions
        all_funcs.append(member)

functions_with_prompts = []
for func in all_funcs:
    # get source code
    code = inspect.getsource(func[1])

    # create prompt to get docstring
    prompt = docstring_prompt(code)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100,
        top_p=1.0,
        stop=["\"\"\""]
    )
    doc_string = response.choices[0].message.content
    merged_code = merge_docstring_and_function(code,doc_string)
    functions_with_prompts.append(merged_code)

functions_to_prompt_name = Path(functions_to_prompt.__file__).stem
with open(f"C:\\workspace\\04_Test_Project\\OpenAI\\Training\\Code\\Docstrings\\{functions_to_prompt_name}_withdocstring.py", "w") as f:
    f.write("\n\n".join(functions_with_prompts))   