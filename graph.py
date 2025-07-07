import re
from typing import Literal, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import ollama
import pprint
import json
import requests
from datetime import datetime
from global_action_repository.drive_access_request_handler import drive_gar

from graph_utils import LLMTools
import prompts
from sop_worflows import drive_access

pp = pprint.PrettyPrinter(indent=2, width=100)

# Ollama client setup
client = ollama.Client()
model = "llama3:8b"

# State definitions
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
    available_steps: List[str]
    completed_steps: List[str]
    workflow_complete: bool
    tool_results: Dict[str, Any]
    user_context: Dict[str, Any]


# ===================
# WORKFLOW NODES
# ===================
def intelligent_step_selection_node(state: State):
    """Select next step and determine required tools"""
    # Initialize available steps if not present
    if not state.get("available_steps"):
        available_steps = [s.strip() for s in state["sop_workflow"].split("\n") if s.strip()]
        state["available_steps"] = available_steps
        state["completed_steps"] = []
    else:
        available_steps = state["available_steps"]
    
    execution_memory = state.get("execution_memory", [])
    completed_steps = state.get("completed_steps", [])
    
    print(f"\n🔄 STEP SELECTION:")
    print(f"Available steps: {len(available_steps)}")
    print(f"Completed steps: {len(completed_steps)}")
    print(f"Execution memory entries: {len(execution_memory)}")
    
    # Check if workflow is complete
    if len(completed_steps) >= len(available_steps):
        print("✅ All workflow steps completed!")
        return {
            **state,
            "workflow_complete": True
        }
    
    # ⚠️ FIRST-RUN LOGIC: If execution memory is empty, pick the first step directly
    if len(execution_memory) == 0:
        next_step = available_steps[0]
        print(f"🚀 First step selected: {next_step}")
        return {
            **state,
            "current_step": next_step,
            "required_tools": [],
            "workflow_complete": False
        }
    
    # Use LLM to determine next step and required tools
    context_prompt = prompts.build_step_selection_prompt(
        available_steps=available_steps,
        completed_steps=completed_steps,
        execution_memory=execution_memory
    )
    
    try:
        response = client.generate(model=model, prompt=context_prompt)
        print("🔍 LLM Response: ", response["response"].strip())
        match = re.search(r"\{.*\}", response["response"].strip(), re.DOTALL)
        result = None
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Invalid JSON:", e)
        else:
            print("No JSON found")
        next_step = result.get("next_step", "")
        required_tools = result.get("required_tools", [])
        reasoning = result.get("reasoning", "")
        
        print(f"🎯 Selected step: {next_step}")
        print(f"🔧 Required tools: {required_tools}")
        print(f"💭 Reasoning: {reasoning}")
        
        if next_step == "WORKFLOW_COMPLETE" or next_step not in available_steps:
            return {
                **state,
                "workflow_complete": True
            }
        
        return {
            **state,
            "current_step": next_step,
            "required_tools": required_tools,
            "workflow_complete": False
        }
        
    except Exception as e:
        print(f"❌ Error in step selection: {e}")
        # Fallback to first uncompleted step
        remaining_steps = [step for step in available_steps if step not in completed_steps]
        if remaining_steps:
            return {
                **state,
                "current_step": remaining_steps[0],
                "required_tools": [],
                "workflow_complete": False
            }
        else:
            return {
                **state,
                "workflow_complete": True
            }


def llm_tool_execution_node(state: State):
    """Execute the current step using LLM with available tools"""
    tools = LLMTools()
    current_step = state["current_step"]
    execution_memory = state.get("execution_memory", [])
    user_context = state.get("user_context", {})
    
    print(f"\n🤖 LLM TOOL EXECUTION for step: {current_step}")
    
    # Create context for LLM
    context_prompt = prompts.build_tool_execution_prompt(
        current_step=current_step,
        tools=tools,
        user_context=user_context,
        execution_memory=execution_memory
    )
    
    try:
        response = client.generate(model=model, prompt=context_prompt)
        print("🔍 LLM Response: ", response["response"].strip())
        match = re.search(r"\{.*\}", response["response"].strip(), re.DOTALL)
        llm_response = None
        if match:
            json_str = match.group(0)
            try:
                llm_response = json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Invalid JSON:", e)
        else:
            print("No JSON found")        
        print("LLM Response JSON TOOL EXECUTION:", llm_response)
        tool_calls = llm_response.get("tool_calls", [])
        reasoning = llm_response.get("reasoning", "")
        print(f"💭 LLM Reasoning: {reasoning}")
        print(f"🔧 Tool Calls: {tool_calls}")
        tool_results = []
        # Execute each tool call
        print()
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})
            
            print(f"🔧 Executing tool: {tool_name}")
            result = tools.execute_tool(tool_name, parameters)
            tool_results.append({
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result
            })
            
            # Update user context with results
            if tool_name == "ask_user_input" and result.get("success"):
                user_context.update({
                    "last_user_input": result.get("user_response"),
                    "last_prompt": result.get("prompt_shown")
                })
        
        # Add to execution memory
        # Add to execution memory
        execution_entry = {
            "action": current_step,
            "observation": tool_results,
            "feedback": f"Completed using tools: {[tc['tool_name'] for tc in tool_calls]}"
        }
        execution_memory.append(execution_entry)

        print("\n🧠 Updated Execution Memory:")
        pp.pprint(execution_memory)

        
        # Mark step as completed
        completed_steps = state.get("completed_steps", [])
        if current_step not in completed_steps:
            completed_steps.append(current_step)
        
        return {
            **state,
            "execution_memory": execution_memory,
            "completed_steps": completed_steps,
            "user_context": user_context,
            "tool_results": {
                "last_execution": tool_results,
                "reasoning": reasoning
            }
        }
        
    except Exception as e:
        print(f"❌ Error in LLM tool execution: {e}")
        
        # Fallback - mark step as completed with error
        execution_memory.append({
            "action": current_step,
            "observation": f"Error: {str(e)}",
            "feedback": "completed_with_error"
        })
        
        completed_steps = state.get("completed_steps", [])
        if current_step not in completed_steps:
            completed_steps.append(current_step)
        
        return {
            **state,
            "execution_memory": execution_memory,
            "completed_steps": completed_steps
        }

# ===================
# WORKFLOW ROUTING
# ===================

def route_from_step_selection(state: State) -> Literal["llm_tool_execution", "__end__"]:
    if state.get("workflow_complete", False):
        return "__end__"
    return "llm_tool_execution"

# ===================
# BUILD WORKFLOW
# ===================

def build_workflow():
    """Build the simplified workflow graph"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("intelligent_step_selection", intelligent_step_selection_node)
    workflow.add_node("llm_tool_execution", llm_tool_execution_node)
    
    # Set entry point
    workflow.set_entry_point("intelligent_step_selection")
    
    # Add edges
    workflow.add_conditional_edges("intelligent_step_selection", route_from_step_selection)
    workflow.add_edge("llm_tool_execution", "intelligent_step_selection")
    
    return workflow.compile()

# ===================
# MAIN EXECUTION
# ===================

def main():
    """Main execution function"""
    # Sample SOP workflow
    sop_content = drive_access.sop_content.strip()
    
    print("🚀 Starting LLM Tools Workflow...\n")
    
    initial_state = {
        "sop_workflow": sop_content,
        "execution_memory": [],
        "global_action_repository": drive_gar,
        "available_steps": [],
        "completed_steps": [],
        "workflow_complete": False,
        "tool_results": {},
        "user_context": {}
    }
    
    # Build and run workflow
    chain = build_workflow()
    config = {"recursion_limit": 20}
    
    try:
        for event in chain.stream(initial_state, config=config):
            if isinstance(event, dict):
                for key, value in event.items():
                    print(f"\n{'='*50}")
                    print(f"📍 Node: '{key}'")
                    print(f"{'='*50}")
                    
                    # Print key state information
                    if "current_step" in value:
                        print(f"🎯 Current Step: {value['current_step']}")
                    if "completed_steps" in value:
                        print(f"✅ Completed Steps: {len(value['completed_steps'])}")
                    if "workflow_complete" in value:
                        print(f"🏁 Workflow Complete: {value['workflow_complete']}")
                    
                    print()
        
        print("\n🎉 Workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow execution error: {e}")

if __name__ == "__main__":
    main()