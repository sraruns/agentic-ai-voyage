import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import json
import gradio as gr


class AboutMe:
    
    def __init__(self):
        # Load .env file from workspace root (two levels up)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        env_path = os.path.join(workspace_root, '.env')
        load_dotenv(dotenv_path=env_path, override=True)
        self.profile_text = self.extract_profile_text()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def record_user_details_tool(self, name, email, phone):
        """Tool function to record user contact details"""
        contact_info = f"User Contact Details:\nName: {name}\nEmail: {email}\nPhone: {phone}"
        print(contact_info)
        # Here you could save to a database or file
        return f"Thank you {name}! I've recorded your contact details. I'll get back to you soon."

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

    def initial_chat_system_prompt(self):
        system_prompt = f"""You are a helpful assistant, act and answer as Arun Kumar for all the questions related to him.
        All the required data is in this {self.profile_text}. You can use this data to answer the questions.
        Create a summary of the {self.profile_text} and use it to answer the questions.
        If you don't know a relevant answer, say so.
        Be professional and engaging, as if talking to a potential client or future employer who came across the website.
        If you don't know the answer, say a professional light hearted joke and 
        say you don't know the answer and ask the user to share their details so that you can answer the question later.
        If the user wants to share their details use the tool record_user_details_tool to record the details.
        Finally say happy to have a conversation with the user.
        """
        return system_prompt

    def validate_response_system_prompt(self, system_prompt):
        validate_response_system_prompt = f"""
        You are a helpful assistant, act and validate the answer provided by another llm as Arun Kumar.
        All the required data is in this {self.profile_text}. You can use this data to validate the answer.
        The other llm was given these instructions {system_prompt}
        Validate the answer provided by the other llm and give feedback.
        Check if the response is:
        1. Accurate according to Arun's profile
        2. Professional and appropriate
        3. Relevant to the question asked
        If the answer is fine return "done" else return "continue".
        Only return one word: either "done" or "continue".
        """
        return validate_response_system_prompt

    def extract_profile_text(self):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the full path to the PDF file
        pdf_path = os.path.join(script_dir, "documents", "profile.pdf")
        try:
            pdf_reader = PdfReader(pdf_path)
            profile_text = ""
            for page in pdf_reader.pages:
                profile_text += page.extract_text()
            return profile_text
        except FileNotFoundError:
            return "Profile document not found. Please ensure the profile.pdf file exists in the documents folder."

    def validate_response(self, response_content, message):
        """Validate the response for quality and accuracy"""
        try:
            validation_prompt = self.validate_response_system_prompt(self.initial_chat_system_prompt())
            validation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": validation_prompt},
                    {"role": "user", "content": f"Question: {message}\nResponse to validate: {response_content}"}
                ]
            )
            
            validation_result = validation_response.choices[0].message.content.lower().strip()
            print(f"Validation Result: {validation_result}")
            
            # Return True if validation passes, False if it needs improvement
            return "done" in validation_result
            
        except Exception as e:
            print(f"Validation error: {e}")
            # If validation fails, assume the response is okay to avoid infinite loops
            return True

    def create_chat_response(self, messages):
        return self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[self.record_user_details_tool_schema]
        )

    def handle_tool_call(self, response):
        tool_call = response.choices[0].message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        if tool_name == "record_user_details_tool":
            tool_result = self.record_user_details_tool(**tool_args)
        else:
            tool_result = f"Unknown tool: {tool_name}"
        
        return {
            "role": "tool",
            "content": tool_result,
            "tool_call_id": tool_call.id
        }

    def chat_with_user(self, message, history):
        try:
            # Convert Gradio history format to OpenAI messages format
            messages = [{"role": "system", "content": self.initial_chat_system_prompt()}]
            
            # Add conversation history
            for entry in history:
                if isinstance(entry, dict):
                    messages.append(entry)
                elif isinstance(entry, list) and len(entry) == 2:
                    # Gradio format: [user_message, assistant_message]
                    messages.append({"role": "user", "content": entry[0]})
                    if entry[1]:  # Check if assistant message exists
                        messages.append({"role": "assistant", "content": entry[1]})
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Quality control loop with validation
            max_attempts = 2  # Prevent infinite loops
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                
                # Get response from OpenAI
                response = self.create_chat_response(messages)
                
                # Handle tool calls if present
                if response.choices[0].message.tool_calls:
                    print(f"Tool Call: {response.choices[0].message.tool_calls}")
                    
                    # Add assistant message with tool call
                    messages.append(response.choices[0].message)
                    
                    # Handle the tool call
                    tool_message = self.handle_tool_call(response)
                    messages.append(tool_message)
                    
                    # Get final response after tool call
                    response = self.create_chat_response(messages)
                
                assistant_response = response.choices[0].message.content
                
                # Validate the response
                if self.validate_response(assistant_response, message):
                    print(f"✅ Response validated and approved (attempt {attempt})")
                    print(f"Response: {assistant_response}")
                    return assistant_response
                else:
                    print(f"❌ Response needs improvement (attempt {attempt})")
                    # Add feedback to messages for next attempt
                    messages.append({"role": "assistant", "content": assistant_response})
                    messages.append({"role": "user", "content": "Please improve your response to be more accurate and professional based on Arun's profile information."})
            
            # If we reach here, return the last response even if not perfect
            print(f"⚠️ Using response after {max_attempts} attempts")
            print(f"Response: {assistant_response}")
            return assistant_response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            print(f"Error in chat_with_user: {e}")
            return error_msg


def main():
    about_me = AboutMe()
    gr.ChatInterface(
        about_me.chat_with_user, 
        title="Arun's AI Assistant", 
        description="Ask anything about Arun's background and experience.",
        type="messages"
    ).launch()

if __name__ == "__main__":
    main()