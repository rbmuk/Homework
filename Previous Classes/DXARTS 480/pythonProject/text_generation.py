import openai

def get_answer(api_key: str, question: str):
    # Initialize the OpenAI client with your API key
    client = openai.OpenAI(api_key=api_key)

    # Define your system message and user question
    system_message = "You are a really helpful person. Answer the following question as clearly as possible."
    question = "Why is the sky blue?"

    # Create the messages list as per the new API structure
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]

    # Make the API call using the updated chat interface
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    # Extract and print the response
    return response.choices[0].message.content.strip()