from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4-turbo",
    temperature=0.8,
    messages=[
        {"role": "system", "content": "You are a singer who love Donald Trump"},
        {"role": "user", "content": "Write a rap about Donald Trump, make sure it rhymes"}
    ]
)
print(response.choices[0].message.content)
