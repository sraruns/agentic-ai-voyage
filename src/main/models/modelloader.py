from openai import OpenAI

from dotenv import load_dotenv
import os
import yaml

load_dotenv(override=True)



def get_openai_model_response(model_name, messages=None):   
    config = yaml.load(open("config.yaml"),Loader=yaml.SafeLoader)
    if model_name == "gpt-5-mini":
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Using {model_name} with config:  {config}")
        
        # Get the model-specific config
        model_config = config.get(model_name, {})
        
        return openai_client.chat.completions.create(
            model=model_name,
            temperature=model_config.get("temperature", 0.5),
            max_completion_tokens=model_config.get("max_tokens", 50),
            messages=messages,
        )
    
    elif model_name == "gpt-4.1-mini":
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Using {model_name} with config: {config}")
        
        # Get the model-specific config
        model_config = config.get(model_name, {})
        
        return openai_client.chat.completions.create(
            model=model_name,
            temperature=model_config.get("temperature", 0.5),
            max_completion_tokens=model_config.get("max_tokens", 50),
            messages=messages,
        )   

  