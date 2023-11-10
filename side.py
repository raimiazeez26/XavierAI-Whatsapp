message = Body

if message:
    messages.append(
        {"role": "user", "content": message},
    )
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=200,
    n=1,
    stop=None,
    temperature=0.5,
)

# The generated text
chat_response = response.choices[0].message.content
messages.append({"role": "assistant", "content": chat_response})