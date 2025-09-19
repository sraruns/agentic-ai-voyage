import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from pypdf import PdfReader
import gradio as gr

# Initialize environment and global variables
load_dotenv(override=True)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_profile_text():
    """Extract text from the Profile.pdf file."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to the PDF file
    pdf_path = os.path.join(script_dir, "documents", "Profile.pdf")
    
    pdf_reader = PdfReader(pdf_path)
    profile_text = ""  # Initialize the variable
    for page in pdf_reader.pages:
        profile_text += page.extract_text()
    return profile_text

# Extract profile text once when the module loads
profile_text = extract_profile_text()

def record_user_details_tool(name, email, phone):
    """Record user contact details."""
    print(f"User details recorded: {name}, {email}, {phone}")
    return {"status": "recorded successfully"}

record_user_details_tool_schema = {
    "type": "function",
    "function": {
        "name": "record_user_details_tool",
        "description": "Record user contact details",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User's full name"},
                "email": {"type": "string", "description": "User's email address"},
                "phone": {"type": "string", "description": "User's phone number"}
            },
            "required": ["name", "email", "phone"]
        }
    }
}

def initial_chat_system_prompt():
    """Create the initial chat system prompt."""
    system_prompt = f"""You are a helpful assistant, act and answer as Arun Kumar for all the questions related to him
All the required data is in this {profile_text}. You can use this data to answer the questions.
Create a summary of the {profile_text} and use it to answer the questions. in the summary ignore reference to Liveperson.
If you don't know a relevant answer, say so.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer, say a professional light hearted joke and 
say you don't know the answer and ask the user to share his details so that you can answer the question later.
If the user wants to share his details use the tool record_user_details_tool to record the details.
Finally say happy to have a conversation with the user.
"""
    return system_prompt

def validate_response_system_prompt_text(system_prompt):
    """Create the validation system prompt."""
    validate_prompt = f"""
You are a helpful assistant, act and validate the answer provided by another llm as Arun Kumar
All the required data is in this {profile_text}. You can use this data to validate the answer.
The other llm was given these instructions {system_prompt}
Validate the answer provided by the other llm and give feedback.
if the answer is fine return done else return continue.
"""
    return validate_prompt

def validate_response(message, history):
    """Validate the response from the main chat."""
    validation_prompt = validate_response_system_prompt_text(initial_chat_system_prompt())
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": validation_prompt}, 
            {"role": "user", "content": message}
        ]
    )
    print(f"Validation Response: {response.choices[0].message.content}")
    return response.choices[0].message.content.lower().strip()

def create_chat_response(history, message):
    """Create a chat response using OpenAI."""
    messages = [{"role": "system", "content": initial_chat_system_prompt()}]
    
    # Add history if it exists
    if history:
        for msg in history:
            if isinstance(msg, dict):
                messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[record_user_details_tool_schema]
    )

def handle_tool_call(response, history):
    """Handle function/tool calls from the AI."""
    tool_call = response.choices[0].message.tool_calls[0]
    tool_call_id = tool_call.id
    tool_call_args = json.loads(tool_call.function.arguments)
    
    # Execute the function
    if tool_call.function.name == "record_user_details_tool":
        tool_call_result = record_user_details_tool(**tool_call_args)
    else:
        tool_call_result = {"error": "Unknown function"}
    
    # Add to history
    history.append({"role": "assistant", "content": response.choices[0].message.content, "tool_calls": response.choices[0].message.tool_calls})
    history.append({"role": "tool", "content": json.dumps(tool_call_result), "tool_call_id": tool_call_id})
    
    print(f"Tool Call Result: {tool_call_result}")
    return tool_call_result

def chat_with_user(message, history):
    """Main chat function for Gradio interface."""
    try:
        # Convert Gradio history format to OpenAI format
        formatted_history = []
        if history:
            for exchange in history:
                if isinstance(exchange, list) and len(exchange) == 2:
                    user_msg, assistant_msg = exchange
                    formatted_history.append({"role": "user", "content": user_msg})
                    formatted_history.append({"role": "assistant", "content": assistant_msg})
        
        # Create response
        response = create_chat_response(formatted_history, message)
        
        # Handle tool calls if any
        if response.choices[0].message.tool_calls:
            print(f"Tool Call: {response.choices[0].message.tool_calls}")
            handle_tool_call(response, formatted_history)
            
            # Get final response after tool call
            response = create_chat_response(formatted_history, "Please provide a final response based on the tool call result.")
        
        final_response = response.choices[0].message.content
        print(f"Response: {final_response}")
        return final_response
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg

def main():
    """Launch the Gradio interface."""
    interface = gr.ChatInterface(
        chat_with_user, 
        title="Arun's AI Assistant", 
        description="Ask anything about Arun's background and experience.",
        examples=["What is your name?", "Tell me about your experience", "What are your skills?"]
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()

