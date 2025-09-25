from agents import Agent, input_guardrail, trace, Runner, GuardrailFunctionOutput
from agents.tool import function_tool
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from sib_api_v3_sdk import Configuration, ApiClient, TransactionalEmailsApi, SendSmtpEmail

load_dotenv(override=True)
instructions1="You are a funny sales representative. you need to create a sales pitch for a prime real estate in Frankfurt in your style."
instructions2="You are a renowned sales representative. you need to create a sales pitch for a prime real estate in Frankfurt in your style."
instructions3="You are a professional sales representative. you need to create a sales pitch for a prime real estate in Frankfurt in your style."

funny_sales_rep=Agent(
    name="funny_sales_rep",
    instructions=instructions1,
    model="gpt-4o-mini"
)

# If you want to use other models, you can use the following code
# AsyncOpenAI
# OpenAIChatCompletionsModel

""" GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)

deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)

sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
 """
# If you want to use other models, you can use the above code

serious_sales_rep=Agent(
    name="serious_sales_rep",
    instructions=instructions2,
    model="gpt-4o-mini"
)

professional_sales_rep=Agent(
    name="professional_sales_rep",
    instructions=instructions3,
    model="gpt-4o-mini"
)
# Convert agents to tools
funny_tool = funny_sales_rep.as_tool(tool_name="get_funny_sales_pitch",tool_description="Get a funny sales pitch for Frankfurt real estate from the funny sales representative")
serious_tool = serious_sales_rep.as_tool(tool_name="get_serious_sales_pitch",tool_description="Get a serious sales pitch for Frankfurt real estate from the serious sales representative") 
professional_tool = professional_sales_rep.as_tool(tool_name="get_professional_sales_pitch",tool_description="Get a professional sales pitch for Frankfurt real estate from the professional sales representative")

@function_tool
def send_email(subject: str, html_body: str):
    """Send an email"""
    configuration = Configuration()
    configuration.api_key['api-key'] = os.environ.get('BREVO_API_KEY')
    api_instance = TransactionalEmailsApi(ApiClient(configuration))
    sender = {"name": "Your Name", "email": "akselvadev@gmail.com"}
    to = [{"email": "arunselva@akselva.com", "name": "Recipient"}]
    send_smtp_email = SendSmtpEmail(sender=sender, to=to, subject=subject, html_content=html_body)
    api_response = api_instance.send_transac_email(send_smtp_email)
    return api_response

# Combine all tools
tools = [funny_tool, serious_tool, professional_tool, send_email] 

instructions_sales_manager="""You are a sales manager of a real estate company. You have to sell a prime real estate in Frankfurt. 

Critical Rules:
1. Call get_funny_sales_pitch, get_serious_sales_pitch, and get_professional_sales_pitch EXACTLY ONCE EACH to collect all three pitches.
2. Compare and evaluate all three sales pitches based on effectiveness, appeal, and persuasiveness.
3. Select and present the BEST sales pitch to the customer.
4. ONLY if the customer expresses clear interest or asks for follow-up, then use the send_email tool.
5. Do NOT generate your own sales pitch - only choose from the ones provided by your representatives.
6. Do NOT call the same tool multiple times unless explicitly requested."""

class NameChecker(BaseModel):
    is_name_in_message: bool
    name: str

guadrail_name_agent=Agent(
    name="guadrail_name_checker",
    instructions="Check if any individual name is mentioned in the email",
    output_type=NameChecker,
    model="gpt-4o-mini"
)

@input_guardrail
async def check_name(ctx,agent, message):
    result = await Runner.run(guadrail_name_agent, message, context=ctx.context)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info=is_name_in_message, tripwire_triggered=is_name_in_message)

agent_sales_manager=Agent(
    name="sales_manager",
    instructions=instructions_sales_manager,
    tools=tools,
    model="gpt-4o-mini",
    input_guardrails=[check_name]
)


    
async def main():
    with trace("sales_manager"):
        result = await Runner.run(agent_sales_manager, "Please sell the a prime real estate in Frankfurt to the customer with an email addressed from Martin")
        print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())