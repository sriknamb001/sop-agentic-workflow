from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import ollama

# This part is not used in the graph logic but kept as per original file
client = ollama.Client()
model = "deepseek-r1:8b"
# client.generate(model=model,prompt=prompt)

# Graph state
class State(TypedDict):
    """
    Represents the state of our graph.
    """
    input: str
    iterations: int


# Nodes
def state_decision_node(state: State):
    """
    decide the next action the agent should execute,
    i.e., decide the state of the agent
    """
    return state


def action_retrieval_node(state: State):
    """
    We use an embedding model to identify the action from the
    set of possible actions from GAR
    """
    return state


def action_execution_node(state: State):
    """
    Generate the data required for executing the selected action
    """
    return state


def api_node(state: State):
    """Represents an action that calls an external API."""
    return state

def external_knowledge_node(state: State):
    """Represents an action that queries a knowledge base."""
    return state

def user_interaction_node(state: State):
    """Represents an action that requires user input."""
    return state

# Conditional routing function
def route_action(state: State) -> Literal["api_tool", "external_knowledge_tool", "user_interaction", "__end__"]:
    """
    Routes the workflow to the correct tool based on the iteration count.
    This is a simple router to demonstrate conditional edges. After three
    iterations, it ends the workflow.
    """
    print("---ROUTING---")
    iterations = state.get("iterations", 0)
    if iterations == 1:
        print("Routing to: api_tool")
        return "api_tool"
    elif iterations == 2:
        print("Routing to: external_knowledge_tool")
        return "external_knowledge_tool"
    elif iterations == 3:
        print("Routing to: user_interaction")
        return "user_interaction"
    else:
        print("Routing to: END")
        return END

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

# Add edges to loop back to the state_decision node
workflow.add_edge("api_tool", "state_decision")
workflow.add_edge("external_knowledge_tool", "user_interaction")
workflow.add_edge("user_interaction", "state_decision")

# Compile the workflow
chain = workflow.compile()

# Invoke
print("Invoking workflow...\n")
initial_state = {"input": "What is the weather in SF?", "iterations": 0}
for event in chain.stream(initial_state):
    for key, value in event.items():
        print(f"Output from node: '{key}'")
        print(f"Current state: {value}\n")
print("\nWorkflow finished.")