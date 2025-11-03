"""LangGraph graph for A2A client

This module implements a LangGraph client that communicates with a LangGraph agent
server using the A2A (Agent-to-Agent) protocol. The client can send messages,
handle responses, and manage multi-turn conversations.
"""

import os
from loguru import logger
from typing import Annotated, List, Dict, Optional, Any
from uuid import uuid4
import asyncio
from dotenv import load_dotenv

load_dotenv()

import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing_extensions import TypedDict

# A2A protocol imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


# ============================================================================
# State Definition
# ============================================================================

class ClientState(TypedDict):
    """State schema for the A2A client LangGraph.
    
    This state tracks:
    - messages: The conversation history (messages to/from the A2A server)
    - a2a_client: The initialized A2A client (set during initialization)
    - agent_card: The fetched agent card metadata
    - task_id: Current task ID for multi-turn conversations
    - context_id: Current context ID for multi-turn conversations
    - user_query: The most recent user query
    - response: The most recent response from the A2A server
    """
    messages: Annotated[List[BaseMessage], add_messages]
    a2a_client: Optional[A2AClient]
    agent_card: Optional[AgentCard]
    task_id: Optional[str]
    context_id: Optional[str]
    user_query: Optional[str]
    response: Optional[Dict[str, Any]]


# ============================================================================
# Tools for the Client Agent
# ============================================================================

@tool
def call_a2a_agent(query: str, task_id: str = None, context_id: str = None) -> str:
    """Call the A2A agent server with a query.
    
    Use this tool when you need to get information from the A2A agent server.
    This tool will send your query to the A2A agent and return its response.
    
    Args:
        query: The question or request to send to the A2A agent
        task_id: Optional task ID for continuing a conversation (usually not needed)
        context_id: Optional context ID for continuing a conversation (usually not needed)
        
    Returns:
        The response text from the A2A agent server
    """
    # Note: This tool function is just a placeholder for the tool definition.
    # The actual A2A call happens in the a2a_call_node using async functions.
    # ToolNode won't execute this, but we intercept it in route_after_agent
    # and route to a2a_call_node instead.
    return "This tool call will be handled by the a2a_call_node"


def get_client_tools() -> list:
    """Get the list of tools available to the client agent.
    
    Returns:
        List of tools the client agent can use
    """
    return [call_a2a_agent]


# ============================================================================
# A2A Client Initialization Functions
# ============================================================================

async def fetch_agent_card(
    httpx_client: httpx.AsyncClient,
    base_url: str = "http://localhost:10000",
    http_kwargs: Optional[Dict[str, Any]] = None
) -> AgentCard:
    """Fetch the agent card from the A2A server.
    
    Args:
        httpx_client: The httpx async client to use for requests
        base_url: The base URL of the A2A server
        http_kwargs: Additional HTTP kwargs (e.g., headers for auth)
        
    Returns:
        AgentCard: The fetched agent card (public or extended)
        
    Raises:
        RuntimeError: If the agent card cannot be fetched
    """
    logger.info(f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}")
    
    # Initialize A2ACardResolver
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )
    
    try:
        # Fetch public agent card
        public_card = await resolver.get_agent_card()
        logger.info("Successfully fetched public agent card")
        logger.debug(public_card.model_dump_json(indent=2, exclude_none=True))
        
        final_card = public_card
        
        # Try to fetch extended card if supported
        if public_card.supports_authenticated_extended_card:
            try:
                logger.info(
                    f"Public card supports authenticated extended card. "
                    f"Attempting to fetch from: {base_url}{EXTENDED_AGENT_CARD_PATH}"
                )
                http_kwargs = http_kwargs or {}
                extended_card = await resolver.get_agent_card(
                    relative_card_path=EXTENDED_AGENT_CARD_PATH,
                    http_kwargs=http_kwargs,
                )
                logger.info("Successfully fetched authenticated extended agent card")
                logger.debug(extended_card.model_dump_json(indent=2, exclude_none=True))
                final_card = extended_card
                logger.info("Using AUTHENTICATED EXTENDED agent card")
            except Exception as e_extended:
                logger.warning(
                    f"Failed to fetch extended agent card: {e_extended}. "
                    "Will proceed with public card."
                )
        else:
            logger.info("Public card does not indicate support for an extended card. Using public card.")
        
        return final_card
        
    except Exception as e:
        logger.error(f"Critical error fetching agent card: {e}", exc_info=True)
        raise RuntimeError("Failed to fetch the agent card. Cannot continue.") from e


async def initialize_a2a_client(
    base_url: str = "http://localhost:10000",
    timeout: float = 60.0,
    http_kwargs: Optional[Dict[str, Any]] = None
) -> tuple[A2AClient, AgentCard, httpx.AsyncClient]:
    """Initialize an A2A client with the agent card.
    
    This function:
    1. Creates an httpx AsyncClient
    2. Fetches the agent card from the server
    3. Creates and returns an A2AClient
    
    Args:
        base_url: The base URL of the A2A server
        timeout: Timeout for HTTP requests in seconds
        http_kwargs: Additional HTTP kwargs (e.g., headers for auth)
        
    Returns:
        tuple: (A2AClient, AgentCard, httpx.AsyncClient)
            Note: The httpx.AsyncClient should be kept alive while using the A2AClient
            
    Raises:
        RuntimeError: If the agent card cannot be fetched
    """
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    
    try:
        # Fetch agent card
        agent_card = await fetch_agent_card(httpx_client, base_url, http_kwargs)
        
        # Create A2A client
        a2a_client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card
        )
        
        logger.info("A2AClient initialized successfully")
        
        return a2a_client, agent_card, httpx_client
        
    except Exception as e:
        await httpx_client.aclose()
        raise RuntimeError(f"Failed to initialize A2A client: {e}") from e


# ============================================================================
# Message Sending Functions
# ============================================================================

async def send_message_to_a2a_agent(
    a2a_client: A2AClient,
    message_text: str,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None
) -> Dict[str, Any]:
    """Send a message to the A2A agent and get the response.
    
    Args:
        a2a_client: The initialized A2AClient
        message_text: The text message to send
        task_id: Optional task ID for multi-turn conversations
        context_id: Optional context ID for multi-turn conversations
        
    Returns:
        Dict containing the response data with keys:
        - result: The response result
        - task_id: Task ID for continuation
        - context_id: Context ID for continuation
        - response_text: The text content of the response
    """
    # Prepare message payload
    message_payload: Dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [
                {'kind': 'text', 'text': message_text}
            ],
            'message_id': uuid4().hex,
        },
    }
    
    # Add task_id and context_id for multi-turn conversations
    if task_id:
        message_payload['message']['task_id'] = task_id
    if context_id:
        message_payload['message']['context_id'] = context_id
    
    # Create request
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(**message_payload)
    )
    
    # Send message
    response = await a2a_client.send_message(request)
    
    # Debug: Log the full response structure to understand its format
    try:
        response_dump = response.model_dump(mode='json', exclude_none=True)
        logger.info(f"Full A2A response structure: {response_dump}")
    except Exception as e:
        logger.warning(f"Could not dump response: {e}")
    
    # Extract response data
    result = response.root.result
    response_data = {
        'result': result,
        'task_id': result.id if hasattr(result, 'id') else None,
        'context_id': result.context_id if hasattr(result, 'context_id') else None,
    }
    
    # Extract text from response parts - try multiple structures
    response_parts = []
    response_text = ''
    
    # Log result structure for debugging
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result attributes: {dir(result)}")
    
    # Method 1: Check if result has a message attribute with parts
    if hasattr(result, 'message') and result.message:
        message = result.message
        logger.info(f"Found message: {type(message)}, attributes: {dir(message) if hasattr(message, '__dict__') else 'N/A'}")
        
        if hasattr(message, 'parts') and message.parts:
            logger.info(f"Found message.parts: {message.parts}")
            for part in message.parts:
                logger.info(f"Part type: {type(part)}, part: {part}")
                # Try accessing text as attribute
                if hasattr(part, 'text') and part.text:
                    response_parts.append(part.text)
                # Try accessing text as dict key
                elif isinstance(part, dict) and 'text' in part and part['text']:
                    response_parts.append(part['text'])
                # Try accessing text via get method
                elif isinstance(part, dict):
                    text_val = part.get('text') or part.get('content')
                    if text_val:
                        response_parts.append(str(text_val))
                # If part is a string, use it directly
                elif isinstance(part, str):
                    response_parts.append(part)
    
    # Method 2: Check if result has content directly
    if not response_parts and hasattr(result, 'content'):
        content = result.content
        logger.info(f"Found content attribute: {type(content)}")
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    response_parts.append(item['text'])
                elif hasattr(item, 'text'):
                    response_parts.append(item.text)
                elif isinstance(item, str):
                    response_parts.append(item)
    
    # Method 3: Check if there's a text attribute directly on result
    if not response_parts and not response_text:
        if hasattr(result, 'text'):
            text_val = result.text
            logger.info(f"Found text attribute: {type(text_val)}")
            if isinstance(text_val, str):
                response_text = text_val
        elif isinstance(result, dict) and 'text' in result:
            response_text = result['text']
    
    # Method 4: Try converting result to dict and extracting
    if not response_parts and not response_text:
        try:
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump(mode='json', exclude_none=True)
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {}
            
            # Try to find text in nested structures
            def extract_text_from_dict(d, depth=0):
                if depth > 5:  # Prevent infinite recursion
                    return []
                texts = []
                if isinstance(d, dict):
                    if 'text' in d and d['text']:
                        texts.append(str(d['text']))
                    if 'content' in d and d['content']:
                        if isinstance(d['content'], str):
                            texts.append(d['content'])
                    for v in d.values():
                        texts.extend(extract_text_from_dict(v, depth + 1))
                elif isinstance(d, list):
                    for item in d:
                        texts.extend(extract_text_from_dict(item, depth + 1))
                elif isinstance(d, str) and len(d) > 10:  # Likely text content
                    texts.append(d)
                return texts
            
            found_texts = extract_text_from_dict(result_dict)
            if found_texts:
                response_parts.extend(found_texts)
                logger.info(f"Extracted text from dict structure: {found_texts}")
        except Exception as e:
            logger.warning(f"Error extracting from dict: {e}")
    
    # Join parts or use direct text
    if response_parts:
        response_text = ' '.join(str(part) for part in response_parts if part)
    elif not response_text:
        # Last resort: try to convert result to string
        try:
            if hasattr(result, 'model_dump'):
                response_dict = result.model_dump(mode='json', exclude_none=True)
            else:
                response_dict = str(result)
            logger.warning(f"Could not extract text from response, dumping full result: {response_dict}")
            response_text = str(response_dict) if not isinstance(response_dict, dict) else ''
        except:
            response_text = ''
    
    response_data['response_text'] = response_text
    
    if not response_text:
        logger.warning("No response text extracted from A2A agent response")
    
    return response_data


# ============================================================================
# LangGraph Node Functions
# ============================================================================

def build_agent_model():
    """Build the LLM model with tools bound to it.
    
    Returns:
        ChatOpenAI model with tools bound
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = get_client_tools()
    return model.bind_tools(tools)


async def agent_node(state: ClientState) -> ClientState:
    """Agent node that uses LLM to decide whether to call tools or respond.
    
    This node:
    1. Invokes the LLM with the conversation history
    2. LLM can decide to call tools (like calling A2A server)
    3. Returns the LLM response with potential tool calls
    
    Args:
        state: The current client state
        
    Returns:
        Updated state with LLM response (which may include tool calls)
    """
    model = build_agent_model()
    messages = state.get("messages", [])
    
    # Invoke model with messages
    response = model.invoke(messages)
    
    logger.info(f"Agent node response: {response.content[:50] if hasattr(response, 'content') else 'N/A'}...")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Agent requested {len(response.tool_calls)} tool call(s)")
    
    return {
        **state,
        "messages": [response]
    }


def route_after_agent(state: ClientState) -> str:
    """Route after agent node based on whether tool calls are needed.
    
    Args:
        state: The current client state
        
    Returns:
        "tools" if tool calls are needed, "a2a_call" if A2A tool is called, "end" otherwise
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Check if there are tool calls
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        # Check if any tool call is for A2A agent
        for tool_call in tool_calls:
            # Handle both dict and object formats
            tool_name = None
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
            elif hasattr(tool_call, "name"):
                tool_name = tool_call.name
            
            if tool_name == "call_a2a_agent":
                logger.info("Agent requested A2A agent call")
                return "a2a_call"
        
        # Other tool calls go to tools node
        logger.info("Agent requested tool calls")
        return "tools"
    
    # No tool calls, agent is done
    return "end"


async def tools_node(state: ClientState) -> ClientState:
    """Execute tool calls (except A2A calls which go to a2a_call node).
    
    Args:
        state: The current client state
        
    Returns:
        Updated state with tool execution results
    """
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    
    # Filter out A2A agent tool calls (they go to a2a_call node)
    tool_calls = getattr(last_message, "tool_calls", []) or []
    non_a2a_tool_calls = [tc for tc in tool_calls if tc.get("name") != "call_a2a_agent"]
    
    if not non_a2a_tool_calls:
        # All tool calls were A2A calls, skip this node
        return state
    
    # Execute tools using ToolNode
    tools = get_client_tools()
    tool_node = ToolNode([tool for tool in tools if tool.name != "call_a2a_agent"])
    
    # Execute tools
    tool_result = await tool_node.ainvoke(state)
    
    return tool_result


async def a2a_call_node(state: ClientState) -> ClientState:
    """Execute A2A agent tool calls.
    
    This node:
    1. Extracts A2A tool calls from the agent's response
    2. Calls the A2A server with the query
    3. Adds the response back to messages
    
    Args:
        state: The current client state
        
    Returns:
        Updated state with A2A response added to messages
    """
    messages = state.get("messages", [])
    if not messages:
        return state
    
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    
    # Find A2A agent tool calls - handle both dict and object formats
    a2a_calls = []
    for tc in tool_calls:
        tool_name = None
        if isinstance(tc, dict):
            tool_name = tc.get("name")
        elif hasattr(tc, "name"):
            tool_name = tc.name
        
        if tool_name == "call_a2a_agent":
            a2a_calls.append(tc)
    
    if not a2a_calls:
        return state
    
    a2a_client = state.get("a2a_client")
    if not a2a_client:
        # Need to initialize first
        logger.warning("A2A client not initialized, initializing now...")
        base_url = os.getenv("A2A_SERVER_URL", "http://localhost:10000")
        timeout = float(os.getenv("A2A_TIMEOUT", "60.0"))
        a2a_client, agent_card, httpx_client = await initialize_a2a_client(
            base_url=base_url,
            timeout=timeout
        )
        state["a2a_client"] = a2a_client
        state["agent_card"] = agent_card
    
    # Process each A2A tool call
    tool_results = []
    for tool_call in a2a_calls:
        # Extract args and tool_call_id - handle both dict and object formats
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
        else:
            args = getattr(tool_call, "args", {})
            tool_call_id = getattr(tool_call, "id", "")
        
        # Extract query from args
        if isinstance(args, dict):
            query = args.get("query", "")
            task_id = args.get("task_id") or state.get("task_id")
            context_id = args.get("context_id") or state.get("context_id")
        else:
            query = getattr(args, "query", "")
            task_id = getattr(args, "task_id", None) or state.get("task_id")
            context_id = getattr(args, "context_id", None) or state.get("context_id")
        
        if not query:
            logger.warning(f"Empty query in A2A tool call: {tool_call}")
            continue
        
        logger.info(f"Calling A2A agent with query: {query[:50]}...")
        
        try:
            response_data = await send_message_to_a2a_agent(
                a2a_client=a2a_client,
                message_text=query,
                task_id=task_id,
                context_id=context_id
            )
            
            response_text = response_data.get("response_text", "")
            task_id = response_data.get("task_id")
            context_id = response_data.get("context_id")
            
            # Update state with new task/context IDs
            state["task_id"] = task_id
            state["context_id"] = context_id
            state["response"] = response_data
            
            # Create tool result message
            tool_result = ToolMessage(
                content=response_text,
                tool_call_id=tool_call_id
            )
            tool_results.append(tool_result)
            
        except Exception as e:
            logger.error(f"Failed to call A2A agent: {e}", exc_info=True)
            tool_result = ToolMessage(
                content=f"Error calling A2A agent: {str(e)}",
                tool_call_id=tool_call_id
            )
            tool_results.append(tool_result)
    
    # Add tool results to messages
    return {
        **state,
        "messages": tool_results
    }


async def initialize_client_node(state: ClientState) -> ClientState:
    """Initialize the A2A client and agent card.
    
    This node:
    1. Fetches the agent card from the server
    2. Initializes the A2AClient
    3. Stores the client and card in the state
    
    Args:
        state: The current client state
        
    Returns:
        Updated state with a2a_client and agent_card set
    """
    base_url = os.getenv("A2A_SERVER_URL", "http://localhost:10000")
    timeout = float(os.getenv("A2A_TIMEOUT", "60.0"))
    
    logger.info(f"Initializing A2A client for server: {base_url}")
    
    try:
        a2a_client, agent_card, httpx_client = await initialize_a2a_client(
            base_url=base_url,
            timeout=timeout
        )
        
        return {
            **state,
            "a2a_client": a2a_client,
            "agent_card": agent_card,
        }
    except Exception as e:
        logger.error(f"Failed to initialize A2A client: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize A2A client: {e}") from e


async def send_message_node(state: ClientState) -> ClientState:
    """Send a message to the A2A agent server.
    
    This node:
    1. Extracts the user query from the state
    2. Sends the message to the A2A server
    3. Updates the state with the response
    
    Args:
        state: The current client state
        
    Returns:
        Updated state with response, task_id, and context_id set
    """
    a2a_client = state.get("a2a_client")
    if not a2a_client:
        raise RuntimeError("A2A client not initialized. Run initialize_client_node first.")
    
    # Get the user query from the last human message
    user_query = state.get("user_query")
    if not user_query:
        # Try to extract from messages
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                user_query = last_message.content
            else:
                # Look for the most recent HumanMessage
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_query = msg.content
                        break
        
        if not user_query:
            raise ValueError("No user query found in state")
    
    logger.info(f"Sending message to A2A agent: {user_query[:50]}...")
    
    # Get task_id and context_id for multi-turn conversations
    task_id = state.get("task_id")
    context_id = state.get("context_id")
    
    try:
        # Send message to A2A agent
        response_data = await send_message_to_a2a_agent(
            a2a_client=a2a_client,
            message_text=user_query,
            task_id=task_id,
            context_id=context_id
        )
        
        # Extract response information
        response_text = response_data.get("response_text", "")
        task_id = response_data.get("task_id")
        context_id = response_data.get("context_id")
        
        # Add AI message to conversation
        ai_message = AIMessage(content=response_text)
        
        logger.info(f"Received response from A2A agent: {response_text[:50]}...")
        
        # Get existing messages and append the new AI message
        existing_messages = state.get("messages", [])
        
        return {
            **state,
            "messages": existing_messages + [ai_message],
            "task_id": task_id,
            "context_id": context_id,
            "response": response_data,
        }
        
    except Exception as e:
        logger.error(f"Failed to send message to A2A agent: {e}", exc_info=True)
        raise RuntimeError(f"Failed to send message to A2A agent: {e}") from e


# ============================================================================
# Conditional Routing Functions
# ============================================================================

def should_initialize_client(state: ClientState) -> str:
    """Route based on whether the A2A client is already initialized.
    
    Args:
        state: The current client state
        
    Returns:
        "skip" if client is already initialized, "initialize" otherwise
    """
    a2a_client = state.get("a2a_client")
    if a2a_client is not None:
        logger.info("A2A client already initialized, skipping initialization")
        return "skip"
    return "initialize"


def route_after_message(state: ClientState) -> str:
    """Route after sending a message based on response status.
    
    Args:
        state: The current client state
        
    Returns:
        "end" if successful, "error" if there was an error
    """
    response = state.get("response")
    if response is None:
        logger.warning("No response received, routing to error")
        return "error"
    
    # Check if response has an error
    if isinstance(response, dict):
        result = response.get("result")
        # Check for error indicators (adjust based on actual A2A response structure)
        if result and hasattr(result, "error"):
            logger.error(f"Response contains error: {result.error}")
            return "error"
    
    logger.info("Message sent successfully, routing to end")
    return "end"


# ============================================================================
# LangGraph Graph Builder
# ============================================================================

def build_a2a_client_graph():
    """Build and compile the LangGraph for the A2A client agent with tool-calling.
    
    The graph flow with tools:
    1. START -> agent (LLM decides what to do)
    2. agent -> route_after_agent (conditional router)
    3. route_after_agent:
       - "tools" → tools_node (for non-A2A tools)
       - "a2a_call" → a2a_call_node (for A2A server calls)
       - "end" → END (agent is done)
    4. tools_node → agent (loop back for more agent decisions)
    5. a2a_call_node → agent (loop back for more agent decisions)
    
    This creates a ReAct-style agent that can:
    - Use its LLM to reason about the user's request
    - Decide when to call the A2A server vs other tools
    - Process tool results and continue reasoning
    
    Returns:
        Compiled StateGraph ready for execution
    """
    graph = StateGraph(ClientState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("a2a_call", a2a_call_node)
    
    # Set entry point to agent
    graph.set_entry_point("agent")
    
    # Add conditional edge after agent decides
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "a2a_call": "a2a_call",
            "end": END
        }
    )
    
    # Loop back to agent after tool execution
    graph.add_edge("tools", "agent")
    graph.add_edge("a2a_call", "agent")
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("A2A client LangGraph with tool-calling agent built and compiled successfully")
    
    return compiled_graph


# ============================================================================
# Main Execution Function
# ============================================================================

async def run_a2a_client(
    user_query: str,
    base_url: str = "http://localhost:10000",
    timeout: float = 60.0
) -> Dict[str, Any]:
    """Run the A2A client graph with a user query.
    
    This is a convenience function that:
    1. Builds the graph
    2. Initializes the state with the user query
    3. Executes the graph
    4. Returns the final state
    
    Args:
        user_query: The user's query to send to the A2A agent
        base_url: The base URL of the A2A server
        timeout: Timeout for HTTP requests in seconds
        
    Returns:
        Dict containing the final state with response information
    """
    # Build the graph
    graph = build_a2a_client_graph()
    
    # Initialize state
    initial_state: ClientState = {
        "messages": [HumanMessage(content=user_query)],
        "a2a_client": None,
        "agent_card": None,
        "task_id": None,
        "context_id": None,
        "user_query": user_query,
        "response": None,
    }
    
    # Set environment variables for nodes
    os.environ["A2A_SERVER_URL"] = base_url
    os.environ["A2A_TIMEOUT"] = str(timeout)
    
    # Execute the graph
    logger.info(f"Executing A2A client graph with query: {user_query[:50]}...")
    final_state = await graph.ainvoke(initial_state)
    
    logger.info("A2A client graph execution completed")
    
    return final_state


async def main() -> None:
    """Main function demonstrating usage of the A2A client graph.
    
    Example usage:
        python langgraph_graph_a2a_client.py
    """
    # Example query
    query = "What are the latest developments in artificial intelligence that you know about in 2025?"
    
    logger.info("=" * 80)
    logger.info("A2A Client LangGraph Example")
    logger.info("=" * 80)
    
    try:
        # Run the client
        result = await run_a2a_client(
            user_query=query,
            base_url="http://localhost:10000",
            timeout=60.0
        )
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("RESULT")
        logger.info("=" * 80)
        
        response = result.get("response", {})
        response_text = response.get("response_text", "")
        
        logger.info(f"User Query: {query}")
        logger.info(f"\nAgent Response:\n{response_text}")
        
        if result.get("task_id"):
            logger.info(f"\nTask ID: {result['task_id']}")
        if result.get("context_id"):
            logger.info(f"Context ID: {result['context_id']}")
        
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"Error running A2A client: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
