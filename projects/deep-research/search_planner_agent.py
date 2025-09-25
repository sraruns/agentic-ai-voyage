
from agents import Agent
from pydantic import BaseModel, Field


class WebSearchItem(BaseModel):
# This is the search plan
    search_item: str = Field(description="This is the search query")
# This is the reasoning for the search plan
    search_reasoning: str = Field(description="This is the reasoning for the search query")


class WebSearchPlan(BaseModel):
# This is the search plan
    search_plan: list[WebSearchItem] = Field(description="This is a list of web search plans")


agent_search_planner=Agent(
    name="search_planner_agent",
    instructions="You are a search planner agent. You are given a query and you need to plan the search plan for the query. The search plan should be a list of web search plans with a maximum of 3 plans.",
    output_type=WebSearchPlan,
    model="gpt-4o-mini"
)
