from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI

openai_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_response(model_name, messages=None):
    return openai_client.chat.completions.create(model=model_name, messages=messages)

def main():
    messages=[{"role":"user","content":"how is the current job market in Germany when will it improve"}]
    response=get_llm_response(model_name="gpt-4o-mini", messages=messages)
    print(f"Resposne: {response}")
    print(f"Resposne: {response.choices[0].message.content}")

if __name__=="__main__":
    main()