import json
from typing import Any, Dict, List

from graph import ExecutionMemoryDict
from graph_utils import LLMTools


def build_step_selection_prompt(available_steps: List[str], completed_steps: List[str], execution_memory: List[ExecutionMemoryDict]) -> str:
    return f"""
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


def build_tool_execution_prompt(current_step: str, tools: LLMTools, user_context: Dict[str, Any], execution_memory: List[ExecutionMemoryDict]) -> str:
    return f"""
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
