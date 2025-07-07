from typing import Literal, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import ollama
import pprint
import json
import requests
from datetime import datetime

pp = pprint.PrettyPrinter(indent=2, width=100)

# Ollama client setup
client = ollama.Client()
model = "llama3:8b"

# Mock Global Action Repository
drive_gar = [
    {
        "action": "get_request_id",
        "action_type": "ask_user_input",
        "user_interaction_metadata": [
            "Prompt: Please enter the drive access request ID."
        ],
        "API": "",
        "params": ""
    },
    {
        "action": "check_request_status",
        "action_type": "api_call",
        "user_interaction_metadata": [],
        "API": "GET /requests/{request_id}",
        "params": "{ 'request_id': <request_id> }"
    },
    {
        "action": "handle_approved",
        "action_type": "show_message_to_user",
        "user_interaction_metadata": ["Message: Drive access already approved."],
        "API": "",
        "params": ""
    },
    {
        "action": "handle_in_progress_or_disapproved_under_72",
        "action_type": "show_message_to_user",
        "user_interaction_metadata": [
            "Message: Request is being processed. Please check back later."
        ],
        "API": "",
        "params": ""
    },
    {
        "action": "create_reapproval_ticket",
        "action_type": "create_ticket",
        "user_interaction_metadata": [],
        "API": "POST /tickets",
        "params": "{ 'title': 'Drive Access Re-Approval Needed', 'description': 'Initial request delayed beyond 72 hours. Requires new approval.' }"
    },
    {
        "action": "inform_user_ticket_created",
        "action_type": "show_message_to_user",
        "user_interaction_metadata": [
            "Message: Access request delayed. A new approval ticket has been created."
        ],
        "API": "",
        "params": ""
    }
]

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
# LLM TOOL DEFINITIONS
# ===================

class LLMTools:
    """Collection of tools that the LLM can invoke"""
    
    def __init__(self):
        self.mock_requests_db = {
            101: {"request_id": 101, "status": "approved", "age_hours": 10, "requester_email": "user1@company.com"},
            102: {"request_id": 102, "status": "in-progress", "age_hours": 50, "requester_email": "user2@company.com"},
            103: {"request_id": 103, "status": "disapproved", "age_hours": 80, "requester_email": "user3@company.com"},
            104: {"request_id": 104, "status": "disapproved", "age_hours": 100, "requester_email": "user4@company.com"},
        }
        self.mock_tickets = []
        self.mock_notifications = []
    
    def get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions for LLM"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_user_input",
                    "description": "Ask the user for input with a specific prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "The question/prompt to show the user"},
                            "context": {"type": "string", "description": "Additional context about what input is needed"}
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_message_to_user",
                    "description": "Display a message to the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "The message to display"},
                            "message_type": {"type": "string", "enum": ["info", "success", "warning", "error"], "description": "Type of message"}
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "api_call",
                    "description": "Make an API call to retrieve or update data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "endpoint": {"type": "string", "description": "API endpoint to call"},
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "description": "HTTP method"},
                            "params": {"type": "object", "description": "Parameters for the API call"},
                            "purpose": {"type": "string", "description": "Purpose of the API call"}
                        },
                        "required": ["endpoint", "method"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Ticket title"},
                            "description": {"type": "string", "description": "Detailed description of the issue"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "description": "Ticket priority"},
                            "request_id": {"type": "integer", "description": "Related request ID"}
                        },
                        "required": ["title", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_notification",
                    "description": "Send a notification to a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Email address of recipient"},
                            "subject": {"type": "string", "description": "Email subject"},
                            "message": {"type": "string", "description": "Email message body"},
                            "request_id": {"type": "integer", "description": "Related request ID"}
                        },
                        "required": ["recipient", "subject", "message"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a specific tool with given parameters"""
        if tool_name == "ask_user_input":
            return self._ask_user_input(parameters)
        elif tool_name == "show_message_to_user":
            return self._show_message_to_user(parameters)
        elif tool_name == "api_call":
            return self._api_call(parameters)
        elif tool_name == "create_ticket":
            return self._create_ticket(parameters)
        elif tool_name == "send_notification":
            return self._send_notification(parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _ask_user_input(self, params: Dict) -> Dict:
        """Mock user input collection"""
        prompt = params.get("prompt", "No prompt provided")
        context = params.get("context", "")
        
        print(f"\n🤖 USER INPUT REQUEST: {prompt}")
        if context:
            print(f"📝 Context: {context}")
        
        # Generate realistic mock response based on prompt content
        mock_response = self._generate_mock_user_response(prompt)
        print(f"👤 SIMULATED USER RESPONSE: {mock_response}")
        
        return {
            "success": True,
            "user_response": mock_response,
            "prompt_shown": prompt,
            "timestamp": datetime.now().isoformat()
        }
    
    def _show_message_to_user(self, params: Dict) -> Dict:
        """Mock message display"""
        message = params.get("message", "No message provided")
        message_type = params.get("message_type", "info")
        
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(message_type, "📢")
        print(f"\n{icon} MESSAGE TO USER: {message}")
        
        # Generate user acknowledgment
        acknowledgment = self._generate_user_acknowledgment(message)
        print(f"👤 USER ACKNOWLEDGMENT: {acknowledgment}")
        
        return {
            "success": True,
            "message_displayed": message,
            "message_type": message_type,
            "user_acknowledgment": acknowledgment,
            "timestamp": datetime.now().isoformat()
        }
    
    def _api_call(self, params: Dict) -> Dict:
        """Mock API calls"""
        endpoint = params.get("endpoint", "")
        method = params.get("method", "GET")
        api_params = params.get("params", {})
        purpose = params.get("purpose", "")
        
        print(f"\n🔗 API CALL: {method} {endpoint}")
        print(f"📝 Parameters: {api_params}")
        if purpose:
            print(f"🎯 Purpose: {purpose}")
        
        # Mock different API endpoints
        if "/requests/" in endpoint:
            return self._mock_requests_api(endpoint, method, api_params)
        elif "/tickets" in endpoint:
            return self._mock_tickets_api(endpoint, method, api_params)
        else:
            return {"error": "Unknown API endpoint", "endpoint": endpoint}
    
    def _create_ticket(self, params: Dict) -> Dict:
        """Mock ticket creation"""
        title = params.get("title", "No title")
        description = params.get("description", "No description")
        priority = params.get("priority", "medium")
        request_id = params.get("request_id")
        
        ticket_id = len(self.mock_tickets) + 1
        ticket = {
            "ticket_id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "request_id": request_id,
            "status": "open",
            "created_at": datetime.now().isoformat()
        }
        self.mock_tickets.append(ticket)
        
        print(f"\n🎫 TICKET CREATED: #{ticket_id} - {title}")
        print(f"📋 Description: {description}")
        print(f"⚡ Priority: {priority}")
        
        return {
            "success": True,
            "ticket": ticket,
            "message": f"Ticket #{ticket_id} created successfully"
        }
    
    def _send_notification(self, params: Dict) -> Dict:
        """Mock notification sending"""
        recipient = params.get("recipient", "")
        subject = params.get("subject", "")
        message = params.get("message", "")
        request_id = params.get("request_id")
        
        notification = {
            "id": len(self.mock_notifications) + 1,
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "request_id": request_id,
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }
        self.mock_notifications.append(notification)
        
        print(f"\n📧 NOTIFICATION SENT TO: {recipient}")
        print(f"📨 Subject: {subject}")
        print(f"💬 Message: {message}")
        
        return {
            "success": True,
            "notification": notification,
            "message": f"Notification sent to {recipient}"
        }
    
    def _mock_requests_api(self, endpoint: str, method: str, params: Dict) -> Dict:
        """Mock requests API responses"""
        if method == "GET":
            # Extract request_id from endpoint or params
            request_id = params.get("request_id")
            if not request_id:
                # Try to extract from endpoint
                import re
                match = re.search(r'/requests/(\d+)', endpoint)
                if match:
                    request_id = int(match.group(1))
            
            if request_id and request_id in self.mock_requests_db:
                request_data = self.mock_requests_db[request_id]
                print(f"✅ REQUEST FOUND: {request_data}")
                return {
                    "success": True,
                    "request": request_data
                }
            else:
                print(f"❌ REQUEST NOT FOUND: {request_id}")
                return {
                    "success": False,
                    "error": "Request not found",
                    "request_id": request_id
                }
        
        return {"error": "Unsupported API operation"}
    
    def _mock_tickets_api(self, endpoint: str, method: str, params: Dict) -> Dict:
        """Mock tickets API responses"""
        if method == "GET":
            return {
                "success": True,
                "tickets": self.mock_tickets
            }
        return {"error": "Unsupported tickets API operation"}
    
    def _generate_mock_user_response(self, prompt: str) -> str:
        """Generate realistic user responses based on prompt"""
        prompt_lower = prompt.lower()
        
        if "request id" in prompt_lower or "request_id" in prompt_lower:
            return "102"
        elif "email" in prompt_lower:
            return "user@company.com"
        elif "reason" in prompt_lower:
            return "Need access for Q4 budget analysis project"
        elif "manager" in prompt_lower:
            return "manager@company.com"
        elif "confirm" in prompt_lower or "approve" in prompt_lower:
            return "Yes, I confirm"
        elif "priority" in prompt_lower:
            return "high"
        else:
            return "Please proceed with the next step"
    
    def _generate_user_acknowledgment(self, message: str) -> str:
        """Generate user acknowledgment based on message"""
        message_lower = message.lower()
        
        if "approved" in message_lower:
            return "Great! Thank you for the approval."
        elif "ticket" in message_lower and "created" in message_lower:
            return "Thank you for creating the ticket."
        elif "processing" in message_lower:
            return "Understood. I'll wait for the process to complete."
        elif "error" in message_lower:
            return "I see there's an issue. What should I do next?"
        else:
            return "Acknowledged. Thank you for the information."

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
    
    execution_memory = state.get("execution_memory", [])
    completed_steps = state.get("completed_steps", [])
    available_steps = state.get("available_steps", [])
    
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
    
    # Use LLM to determine next step and required tools
    context_prompt = f"""
    You are managing a workflow execution. Analyze the current state and determine:
    1. The next step to execute
    2. What tools you need to use for that step
    
    Available SOP Steps:
    {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(available_steps)])}
    
    Completed Steps:
    {chr(10).join([f"- {step}" for step in completed_steps])}
    
    Recent Execution Memory:
    {chr(10).join([f"- {mem['action']}: {mem['feedback']}" for mem in execution_memory[-3:]])}
    
    Available Tools:
    - ask_user_input: Get input from user
    - show_message_to_user: Display message to user
    - api_call: Make API calls to check status/data
    - create_ticket: Create support tickets
    - send_notification: Send notifications
    
    Rules:
    1. Don't select completed steps
    2. Consider step dependencies
    3. Choose the most logical next step
    4. Identify which tools you'll need
    
    Respond with JSON format:
    {{
        "next_step": "exact text of next step or WORKFLOW_COMPLETE",
        "required_tools": ["tool1", "tool2"],
        "reasoning": "why this step is next"
    }}
    """
    
    try:
        response = client.generate(model=model, prompt=context_prompt)
        result = json.loads(response["response"].strip())
        
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
    context_prompt = f"""
    You are executing a workflow step. You have access to various tools to complete this step.
    
    Current Step: {current_step}
    
    Available Tools:
    {json.dumps(tools.get_tool_definitions(), indent=2)}
    
    User Context (from previous interactions):
    {json.dumps(user_context, indent=2)}
    
    Recent Execution Memory:
    {json.dumps(execution_memory[-3:], indent=2)}
    
    Your task is to:
    1. Analyze what this step requires
    2. Use the appropriate tools to complete it
    3. Provide a summary of what was accomplished
    
    You can call tools by responding with JSON in this format:
    {{
        "tool_calls": [
            {{
                "tool_name": "tool_name",
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2"
                }}
            }}
        ],
        "reasoning": "Why you're using these tools"
    }}
    
    If no tools are needed, respond with:
    {{
        "tool_calls": [],
        "reasoning": "This step is complete or no tools needed",
        "completion_status": "complete"
    }}
    """
    
    try:
        response = client.generate(model=model, prompt=context_prompt)
        llm_response = json.loads(response["response"].strip())
        
        tool_calls = llm_response.get("tool_calls", [])
        reasoning = llm_response.get("reasoning", "")
        
        print(f"💭 LLM Reasoning: {reasoning}")
        
        tool_results = []
        
        # Execute each tool call
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
        execution_memory.append({
            "action": current_step,
            "observation": tool_results,
            "feedback": f"Completed using tools: {[tc['tool_name'] for tc in tool_calls]}"
        })
        
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
    sop_content = """Check if user has provided request ID
Query the request status from the system
If request is approved, inform user
If request is in progress or disapproved and under 72 hours, inform user to wait
If request is over 72 hours old, create reapproval ticket
Notify user about ticket creation
Send notification to manager if needed"""
    
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