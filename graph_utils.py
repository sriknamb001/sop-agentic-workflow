from datetime import datetime
import numpy as np
import ollama
from typing import Dict, List

client = ollama.Client()
embedding_model = "deepseek-r1:8b"


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