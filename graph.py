from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import ollama
import pprint
from global_action_repository.drive_access_request_handler import drive_gar
import graph_utils


pp = pprint.PrettyPrinter(indent=2, width=100)

# This part is not used in the graph logic but kept as per original file
client = ollama.Client()
model = "deepseek-r1:8b"
# client.generate(model=model,prompt=prompt)

from typing import List, TypedDict

class ExecutionMemoryDict(TypedDict):
    action: str
    observation: str
    feedback: str

class GlobalAction(TypedDict):
    action: str
    action_type: str
    user_interaction_metadata: List[str]
    API: str
    params: str

class State(TypedDict):
    sop_workflow: str
    execution_memory: List[ExecutionMemoryDict]
    global_action_repository: List[GlobalAction]
    current_step: str
    tool_type: str
    selected_action: str
    current_prompt: str
    api_url: str
    api_params: str
    user_message: str

def state_decision_node(state: State):
    if "parsed_steps" not in state:
        steps = state["sop_workflow"].split("\n")
        state["parsed_steps"] = [s.strip() for s in steps if s.strip()]
        state["step_index"] = 0
    
    if state["step_index"] >= len(state["parsed_steps"]):
        print("All SOP steps executed.")
        return END
    
    state["current_step"] = state["parsed_steps"][state["step_index"]]
    state["step_index"] += 1  # Increment after setting current_step
    print(f"state:{state}")
    return state

def action_retrieval_node(state: State):
    """
    We use an embedding model to identify the action from the
    set of possible actions from GAR
    """
    word_emb = graph_utils.get_embedding(state["current_step"])
    max_similarity = -1
    most_similar_word = None

    for candidate in [entry["action"] for entry in state["global_action_repository"]]:
        candidate_emb = graph_utils.get_embedding(candidate)
        similarity = graph_utils.cosine_similarity(word_emb, candidate_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_word = candidate
    state["selected_action"] = most_similar_word
    return state


def action_execution_node(state: State):
    selected_action = state["selected_action"]
    action_details = next((item for item in state["global_action_repository"] if item["action"] == selected_action), None)
    print(f'action_details: {action_details}')
    if not action_details:
        raise ValueError(f"No action found for: {selected_action}")

    action_type = action_details["action_type"]
    state["tool_type"] = action_type  # For routing in `route_action`
    
    if action_type == "ask_user_input":
        # Use the LLM to generate the user question prompt
        print("INSIDE HERE ")
        prompt = f"Generate a user prompt for this action: {selected_action}"
        response = client.generate(model=model, prompt=prompt)
        question = response["response"].strip()
        print(f'question: {question}')
        state["current_prompt"] = question

    elif action_type == "api_call":
        # Generate the parameters required for the API call from the current context
        prompt = f"""Extract parameters for this API: {selected_action}
        Current context: {state.get("execution_memory", [])}
        Params needed: {action_details["params"]}
        """
        response = client.generate(model=model, prompt=prompt)
        params = response["response"].strip()
        state["api_params"] = params
        state["api_url"] = action_details["API"]

    elif action_type == "show_message_to_user":
        # Use metadata to generate the message
        prompt = f"""Generate a message to show the user based on:
        Message Info: {action_details["user_interaction_metadata"]}
        """
        response = client.generate(model=model, prompt=prompt)
        message = response["response"].strip()
        state["user_message"] = message

    elif action_type == "external_knowledge":
        # Generate search query using current context
        prompt = f"""Generate a search query based on the action and execution memory.
        Action: {selected_action}
        Context: {state.get("execution_memory", [])}
        """
        response = client.generate(model=model, prompt=prompt)
        search_query = response["response"].strip()
        state["search_query"] = search_query

    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    return state



import requests  # Make sure to install this or replace with aiohttp if async

def api_node(state: State):
    """Represents an action that calls an external API."""
    url = state.get("api_url")
    params = state.get("api_params")

    # You can parse `params` if it's a stringified JSON from LLM output
    if isinstance(params, str):
        try:
            import json
            params = json.loads(params)
        except Exception as e:
            raise ValueError(f"Failed to parse API parameters: {params}\nError: {str(e)}")

    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}

    # Append to execution memory
    state["execution_memory"].append({
        "action": state["selected_action"],
        "observation": result,
        "feedback": "api_call_completed"
    })

    return state


def external_knowledge_node(state: State):
    """Mocks querying an external knowledge source using a generated search query."""
    search_query = state.get("search_query", "N/A")

    # Simulate knowledge search
    mock_result = {
        "search_query": search_query,
        "answer": f"Mocked answer for query: '{search_query}'"
    }

    state["execution_memory"].append({
        "action": state["selected_action"],
        "observation": mock_result,
        "feedback": "external_knowledge_query_completed"
    })

    return state


def user_interaction_node(state: State):
    """Handles showing messages to or collecting input from the user."""
    action_type = state.get("tool_type")
    if action_type == "ask_user_input":
        prompt = state.get("current_prompt", "No prompt provided.")
        mock_user_input = "mock_user_input_value"  # Simulate user input
        observation = {"user_response": mock_user_input}
    elif action_type == "show_message_to_user":
        message = state.get("user_message", "No message available.")
        # In real setting, just display this to user and proceed
        observation = {"message_shown": message}
    else:
        observation = {"note": "Unknown user interaction type."}

    state["execution_memory"].append({
        "action": state["selected_action"],
        "observation": observation,
        "feedback": "user_interaction_completed"
    })
    return state


# Conditional routing function
def route_action(state: State) -> Literal["api_tool", "external_knowledge_tool", "user_interaction", "__end__"]:
    tool_type = state.get("tool_type")
    if tool_type == "api_call":
        return "api_tool"
    elif tool_type == "external_knowledge":
        return "external_knowledge_tool"
    elif tool_type in ("ask_user_input", "show_message_to_user"):
        return "user_interaction"
    else:
        return "__end__"


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("state_decision", state_decision_node)
workflow.add_node("action_retrieval", action_retrieval_node)
workflow.add_node("action_execution", action_execution_node)
workflow.add_node("api_tool", api_node)
workflow.add_node("external_knowledge_tool", external_knowledge_node)
workflow.add_node("user_interaction", user_interaction_node)

# Set the entry point
workflow.set_entry_point("state_decision")

# Add edges according to the specified flow
workflow.add_edge("state_decision", "action_retrieval")
workflow.add_edge("action_retrieval", "action_execution")

# Add the conditional edge after action_execution
workflow.add_conditional_edges("action_execution", route_action)

# # Add edges to loop back to the state_decision node
workflow.add_edge("api_tool", "state_decision")
workflow.add_edge("external_knowledge_tool", "user_interaction")
workflow.add_edge("user_interaction", "state_decision")

# Compile the workflow
chain = workflow.compile()

# Invoke
print("Invoking workflow...\n")
initial_state = {
    "sop_workflow": open("/Users/srikarnamburi/Documents/sop-agentic-workflow/sop_worflows/drive_access.txt").read(),
    "execution_memory": [],
    "global_action_repository": drive_gar
}
for event in chain.stream(initial_state):
    if isinstance(event, dict):
        for key, value in event.items():
            print(f"Output from node: '{key}'")
            print("Current state:")
            pp.pprint(value)
            print()
    else:
        print(f"Event: {event}\n")

print("\nWorkflow finished.")