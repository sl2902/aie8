"""Build a LangGraph MCP app that interacts with the MCP server"""

import os
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model

load_dotenv()
model = init_chat_model("openai:gpt-4.1")


client = MultiServerMCPClient(
    {
        "server": {
            "command": "python",
            "args": [f'{os.getenv("MCP_SERVER_BASE_PATH")}/13_MCP/server.py'],
            "transport": "stdio",
        }
    }
)

async def main():
    tools = await client.get_tools()

    def call_model(state: MessagesState) -> MessagesState:
        response = model.bind_tools(tools).invoke(state["messages"])
        return {
            "messages": response
        }

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()

    roll_dice_response = await graph.ainvoke({"messages": "Roll a dice"})
    metal_price_response = await graph.ainvoke({"messages": "What is the price of gold?"})

    print(f"Roll dice response: {roll_dice_response["messages"][2].content}")
    print(f"Metal price response: ${metal_price_response["messages"][2].content}")


if __name__ == "__main__":
    asyncio.run(main())


