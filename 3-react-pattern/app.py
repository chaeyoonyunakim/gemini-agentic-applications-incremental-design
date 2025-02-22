from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import random

text_model = ChatGoogleGenerativeAI(
    max_tokens=None, model="gemini-2.0-flash", temperature=0, top_k=1
)

@tool
def get_top_gainers_and_losers() -> str:
    """Get today's top gainers and losers in the market.

    Returns:
        str: A string containing information about top gaining and losing stocks for the day.
    """
    return f"Apple is top loser today, it is valued now at $105."

@tool
def forecast_stock_price(initial_price: int) -> str:
    """Forecast the stock price of a given company.

    Args:
        initial_price (int): The current stock price to base the forecast on

    Returns:
        str: A string containing the forecasted stock price
    """
    return f"${initial_price + random.randint(1, 5000)}."


tools = [get_top_gainers_and_losers, forecast_stock_price]

text_model_with_tools = text_model.bind_tools(tools)

sys_msg = SystemMessage(
    content="""You are an investment analyst equipped with tools such as stock price forecasting and getting top gainers and losers in the market.
You are running in a test environment, so you can assume that the results are relevant and that you are free to give advice regarding the stock market."""
)
def agent(state: MessagesState):
    return {"messages": [text_model_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
)
builder.add_edge("tools", "agent")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="What are the top gainers and losers in the stock market?")]},
    config,
)
print(result)
print("--------------------------------")

result2 = graph.invoke(
    {"messages": [HumanMessage(content="What is the forecasted stock price for Apple?")]},
    config,
)
print(result2)
print("--------------------------------")

result3 = graph.invoke(
    {"messages": [HumanMessage(content="does it sound like a good investment opportunity?")]},
    config,
)
print(result3)