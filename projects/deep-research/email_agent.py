from agents import Agent, function_tool
from sib_api_v3_sdk import ApiClient, Configuration, SendSmtpEmail, TransactionalEmailsApi
import os

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

email_agent=Agent(
    name="email_agent",
    instructions="You are a email agent. You are given a email and you need to send the email. Derive a proper subject and body for the email. And use the email tool send_email to send the email.",
    model="gpt-4o-mini",
    tools=[send_email]
)
