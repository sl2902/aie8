"""LangGraph helpfulness agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from .models import get_openai_model
from .rag import ProductionRAGChain
from .guardrails import (
    create_guardrails_guard,
    create_factuality_guard,
    validate_input,
    validate_output,
    GuardrailsState
)
from guardrails import Guard


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Retrieve information from the RAG chain."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def tool_call_or_helpful(state):
    """Determine next step: tool call, helpfulness check, or end."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"

    # Check if we've reached maximum conversation length
    if len(state["messages"]) > 10:
        return END

    # Only check helpfulness if we have at least a query and response
    if len(state["messages"]) < 2:
        return "continue"

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """\
    Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

    Initial Query:
    {initial_query}

    Final Response:
    {final_response}"""

    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_check_model = ChatOpenAI(model="gpt-4o-mini")
    helpfulness_chain = helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()

    try:
        helpfulness_response = helpfulness_chain.invoke({
            "initial_query": initial_query.content, 
            "final_response": final_response.content
        })
        
        return "end" if "Y" in helpfulness_response else "continue"
    except Exception as e:
        print(f"Error in helpfulness check: {e}")
        return "continue"


def create_langgraph_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a LangGraph agent following the helpfulness agent pattern.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages - agent node."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build graph following helpfulness agent pattern
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes: agent (model) and action (tools)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    
    # Set entry point to agent
    graph.set_entry_point("agent")
    
    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        tool_call_or_helpful,
        {
            "continue" : "agent",
            "action" : "action",
            "end" : END
        }
    )
    
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_guarded_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    enable_refinement: bool = True
):
    """Create a production-safe LangGraph agent with guardrails validation.
    
    This agent combines helpfulness evaluation with guardrails validation for
    production safety. It validates inputs before processing and outputs before
    returning to users.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        input_guard: Guard for input validation. If None, creates default guard.
        output_guard: Guard for output validation. If None, creates default guard.
        enable_refinement: If True, allows agent to refine failed outputs
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Create default guards if not provided
    if input_guard is None:
        input_guard = create_guardrails_guard(
            enable_jailbreak_detection=True,
            enable_pii_protection=True,
            enable_profanity_check=True
        )
    
    if output_guard is None:
        output_guard = create_guardrails_guard(
            enable_profanity_check=True,
            enable_pii_protection=True
        )
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def input_guard_node(state: AgentState) -> Dict[str, Any]:
        """Validate user input before processing."""
        messages = state.get("messages", [])
        if not messages:
            return {"validation_results": {"input": {"passed": True}}}
        
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return {"validation_results": {"input": {"passed": True}}}
        
        try:
            result = validate_input(input_guard, last_message.content, raise_on_failure=False)
            return {
                "validation_results": {
                    "input": {
                        "passed": result["validation_passed"],
                        "error": result.get("error")
                    }
                }
            }
        except Exception as e:
            return {
                "validation_results": {
                    "input": {
                        "passed": False,
                        "error": str(e)
                    }
                }
            }
    
    def output_guard_node(state: AgentState) -> Dict[str, Any]:
        """Validate agent output before returning."""
        messages = state.get("messages", [])
        if not messages:
            return {"validation_results": {"output": {"passed": True}}}
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return {"validation_results": {"output": {"passed": True}}}
        
        try:
            result = validate_output(output_guard, last_message.content, raise_on_failure=False)
            return {
                "validation_results": {
                    "output": {
                        "passed": result["validation_passed"],
                        "error": result.get("error")
                    }
                }
            }
        except Exception as e:
            return {
                "validation_results": {
                    "output": {
                        "passed": False,
                        "error": str(e)
                    }
                }
            }
    
    def guard_failure_node(state: AgentState) -> Dict[str, Any]:
        """Handle guard failures - return error message."""
        validation_results = state.get("validation_results", {})
        error_msg = "I apologize, but I cannot process this request due to safety guidelines."
        
        # Determine which guard failed
        if validation_results.get("input", {}).get("passed") == False:
            error_msg = "Your input was blocked by safety filters. Please rephrase your question."
        elif validation_results.get("output", {}).get("passed") == False:
            error_msg = "I cannot provide this response due to content safety policies."
        
        error_message = AIMessage(content=error_msg)
        return {"messages": [error_message]}
    
    def route_after_input_guard(state: AgentState) -> str:
        """Route after input validation."""
        validation_results = state.get("validation_results", {})
        if validation_results.get("input", {}).get("passed") == False:
            return "guard_failure"
        return "agent"
    
    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Perform helpfulness evaluation."""
        # This is a passthrough - helpfulness logic is in tool_call_or_helpful
        # We just need this node to exist for routing
        return state
    
    def route_after_output_guard(state: AgentState) -> str:
        """Route after output validation."""
        validation_results = state.get("validation_results", {})
        if validation_results.get("output", {}).get("passed") == False:
            if enable_refinement:
                return "agent"  # Allow refinement
            return "guard_failure"
        return "helpfulness"
    
    def route_after_helpfulness(state: AgentState) -> str:
        """Route after helpfulness check."""
        return tool_call_or_helpful(state)
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.add_node("guard_failure", guard_failure_node)
    
    # Set entry point
    graph.set_entry_point("input_guard")
    
    # Add edges
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {
            "agent": "agent",
            "guard_failure": "guard_failure"
        }
    )
    
    graph.add_conditional_edges(
        "agent",
        tool_call_or_helpful,
        {
            "continue": "agent",
            "action": "action",
            "end": "output_guard"
        }
    )
    
    graph.add_edge("action", "agent")
    
    graph.add_conditional_edges(
        "output_guard",
        route_after_output_guard,
        {
            "helpfulness": "helpfulness",
            "agent": "agent",
            "guard_failure": "guard_failure"
        }
    )
    
    graph.add_conditional_edges(
        "helpfulness",
        route_after_helpfulness,
        {
            "continue": "agent",
            "action": "action",
            "end": END
        }
    )
    
    graph.add_edge("guard_failure", END)
    
    return graph.compile()