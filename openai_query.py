from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_question_about_abaumannii(query):

    model="gpt-3.5-turbo"

    system_content = """
    You are a biomedicial researcher. You give very concise answers, between 50 and 100 words.
    The user will ask a question related to a pathogen. If the question is not related to a pathogen, you should not respond and, rather, you should ask the user to input another question.
    Always greet the user. Welcome them to the H3D Symposium in Livingstone, Zambia.
    In addition, you should repeat user's question explicitly in a concise way and then answer it.
    """
    
    user_content = query

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(model = model, messages = messages)

    return response.choices[0].message.content


answer = ask_question_about_abaumannii("What diseases are caused by Acinetobacter baumannii?")

print(answer)