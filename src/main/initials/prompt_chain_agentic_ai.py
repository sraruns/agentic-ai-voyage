from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
# Add the src directory to Python path so we can import from models
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, src_dir)

from main.models.modelloader import get_openai_model_response

load_dotenv(override=True)
# Add the src directory to Python path

llm_model="gpt-4.1-mini"  # Use a real model that exists

question="Pick me a business that i can explore for agentic ai solutions"
messages = [
    {"role": "user", "content": question},
]
response = get_openai_model_response(llm_model,messages=messages)
agentic_ai_business_solution = response.choices[0].message.content
print(f"Agentic AI business solution: {agentic_ai_business_solution}")


question="whats the pain points of the business "+str(agentic_ai_business_solution)
messages = [
    {"role": "user", "content": question},
]
response = get_openai_model_response( llm_model,messages=messages)
pain_points = response.choices[0].message.content
print(f"Pain points: {pain_points}")

question="can you propose an agentic ai solution for the pain point "+str(pain_points)
messages = [
    {"role": "user", "content": question},
]
response = get_openai_model_response(llm_model,messages=messages)
pain_points_solution = response.choices[0].message.content
print(f"Pain points solution: {pain_points_solution}")


from IPython.display import display, Markdown
display(Markdown(pain_points_solution))






