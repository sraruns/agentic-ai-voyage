import gradio as gr
from dotenv import load_dotenv
from deep_websearch_agent import DeepWebSearchAgent

async def run_deep_websearch_agent(query: str):
   async for chunk in DeepWebSearchAgent().run(query):
        yield chunk

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo_ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run_deep_websearch_agent, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run_deep_websearch_agent, inputs=query_textbox, outputs=report)


demo_ui.launch(inbrowser=True)



    