import asyncio
from agents import Agent, gen_trace_id, trace, Runner
from pydantic import BaseModel, Field
from web_search_agent import web_search_agent
from search_planner_agent import WebSearchItem, WebSearchPlan, agent_search_planner
from email_agent import email_agent
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class DeepWebSearchAgent(BaseModel):
    async def run(self, query: str):
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"

            search_plan_results = await self.search_planner(query)
            yield f"Search plan generated"

            search_results = await self.run_searches_parallel(search_plan_results)
            yield f"Search plans are executed and results are generated"

            report_results = await self.generate_report(query, search_results)
            yield f"From the search results, a proper report is generated"

            await self.email(report_results)
            yield f"The report is sent to the email"
            yield report_results.final_output.report

    
    async def search_planner(self, query: str):
        search_plan_results = await Runner.run(agent_search_planner, query)
        print(f"Will perform {len(search_plan_results.final_output.search_plan)} searches")
        return search_plan_results.final_output_as(WebSearchPlan)

    async def run_searches_parallel(self, search_plan_list: WebSearchPlan):
        tasks = [asyncio.create_task(self.search_web(plan)) for plan in search_plan_list.search_plan]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
                print(f"Searching... {len(results)}/{len(tasks)} completed")
        print("Finished searching")
        return results
       
    async def search_web(self, search_item: WebSearchItem):
        print(f"Searching... {search_item.search_item}")
        input_item=f"Search for {search_item.search_item} with the following reasoning: {search_item.search_reasoning}"
        search_results = await Runner.run(web_search_agent, input_item)
        return search_results

    class Report(BaseModel):
        report: str = Field(description="The report of the search results")
        summary: str = Field(description="The summary of the search results")
        follow_up_questions: list[str] = Field(description="The follow up questions to the search results")

    async def email(self, report_results:str):

        email_results = await Runner.run(email_agent, str(report_results))
        return email_results

    async def generate_report(self, query: str, search_results: list[str]):
       
        final_report_instructions = (
                "You are a senior researcher tasked with writing a cohesive report for a research query. "
                "You will be provided with the original query, and some initial research done by a research assistant.\n"
                "You should first come up with an outline for the report that describes the structure and "
                "flow of the report. Then, generate the report and return that as your final output.\n"
                "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
                "for 5-10 pages of content, at least 1000 words."
            )
        agent_report=Agent(
            name="report_agent",
            instructions=final_report_instructions,
            output_type=self.Report,
            model="gpt-4o-mini"
        )
        input_report=f"Original query: {query}\n\nInitial research: {search_results}"
        report_results = await Runner.run(agent_report, input_report)
        return report_results

    