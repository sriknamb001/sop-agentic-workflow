from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import uvicorn

app = FastAPI(title="SOP Workflow API", version="1.0.0")

# =======================
# Models
# =======================
class Request(BaseModel):
    request_id: int
    status: str
    created_at: datetime
    requester_email: Optional[str] = None
    reason: Optional[str] = None
    manager_email: Optional[str] = None

class Ticket(BaseModel):
    ticket_id: int
    request_id: int
    reason: str
    created_at: datetime
    priority: str = "medium"
    status: str = "open"

class UserInput(BaseModel):
    action: str
    user_response: str
    context: Optional[dict] = None

class NotificationRequest(BaseModel):
    recipient: str
    subject: str
    message: str
    request_id: Optional[int] = None

# =======================
# In-memory DB
# =======================
REQUESTS_DB = {
    101: Request(
        request_id=101, 
        status="approved", 
        created_at=datetime.now() - timedelta(hours=10),
        requester_email="user1@company.com",
        reason="Q4 budget analysis project files",
        manager_email="john.smith@company.com"
    ),
    102: Request(
        request_id=102, 
        status="in-progress", 
        created_at=datetime.now() - timedelta(hours=50),
        requester_email="user2@company.com",
        reason="Marketing campaign assets",
        manager_email="jane.doe@company.com"
    ),
    103: Request(
        request_id=103, 
        status="disapproved", 
        created_at=datetime.now() - timedelta(hours=80),
        requester_email="user3@company.com",
        reason="Personal project files",
        manager_email="bob.wilson@company.com"
    ),
    104: Request(
        request_id=104, 
        status="disapproved", 
        created_at=datetime.now() - timedelta(hours=100),
        requester_email="user4@company.com",
        reason="Outdated project documentation",
        manager_email="alice.brown@company.com"
    ),
}

TICKET_QUEUE: List[Ticket] = []
NOTIFICATIONS_LOG: List[dict] = []

# =======================
# Service Layer
# =======================
def get_request_by_id(request_id: int) -> Request:
    req = REQUESTS_DB.get(request_id)
    if not req:
        raise ValueError("Request ID not found.")
    return req

def create_ticket(request_id: int, reason: str, priority: str = "medium") -> Ticket:
    ticket_id = len(TICKET_QUEUE) + 1
    ticket = Ticket(
        ticket_id=ticket_id,
        request_id=request_id,
        reason=reason,
        created_at=datetime.now(),
        priority=priority,
        status="open"
    )
    TICKET_QUEUE.append(ticket)
    return ticket

def send_notification(recipient: str, subject: str, message: str, request_id: Optional[int] = None) -> dict:
    """Mock notification sending"""
    notification = {
        "id": len(NOTIFICATIONS_LOG) + 1,
        "recipient": recipient,
        "subject": subject,
        "message": message,
        "request_id": request_id,
        "sent_at": datetime.now(),
        "status": "sent"
    }
    NOTIFICATIONS_LOG.append(notification)
    return notification

def process_drive_access_sop(request_id: int) -> dict:
    request = get_request_by_id(request_id)
    age_hours = (datetime.now() - request.created_at).total_seconds() / 3600
    
    if request.status == "approved":
        return {
            "message": "Drive access already approved.",
            "request_id": request_id,
            "status": "approved",
            "requester": request.requester_email
        }
    
    if request.status in ["in-progress", "disapproved"]:
        if age_hours <= 72:
            return {
                "message": "Request is being processed. Please check back later.",
                "request_id": request_id,
                "status": request.status,
                "age_hours": round(age_hours, 2)
            }
        else:
            ticket = create_ticket(
                request_id, 
                reason="Drive Access Re-Approval Needed",
                priority="high"
            )
            
            # Send notification to manager
            if request.manager_email:
                send_notification(
                    recipient=request.manager_email,
                    subject=f"Drive Access Request {request_id} - Action Required",
                    message=f"Drive access request {request_id} has been delayed and requires re-approval.",
                    request_id=request_id
                )
            
            return {
                "message": "Access request delayed. A new approval ticket has been created.",
                "ticket_id": ticket.ticket_id,
                "request_id": request_id,
                "notification_sent": bool(request.manager_email)
            }

def update_request_status(request_id: int, new_status: str) -> dict:
    """Update request status"""
    if request_id not in REQUESTS_DB:
        raise ValueError("Request ID not found.")
    
    REQUESTS_DB[request_id].status = new_status
    return {
        "message": f"Request {request_id} status updated to {new_status}",
        "request_id": request_id,
        "new_status": new_status
    }

# =======================
# API Routes
# =======================
@app.get("/")
def root():
    return {"message": "SOP Workflow API is running.", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "total_requests": len(REQUESTS_DB),
        "total_tickets": len(TICKET_QUEUE),
        "notifications_sent": len(NOTIFICATIONS_LOG)
    }

@app.get("/sop/drive-access")
def drive_access_sop(request_id: int = Query(..., description="ID of the access request")):
    """Main SOP endpoint for drive access workflow"""
    try:
        return process_drive_access_sop(request_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/requests/{request_id}")
def get_request(request_id: int):
    """Get specific request details"""
    try:
        request = get_request_by_id(request_id)
        return request
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/requests")
def list_requests():
    """List all requests"""
    return {"requests": list(REQUESTS_DB.values())}

@app.patch("/requests/{request_id}/status")
def update_request_status_endpoint(request_id: int, status: str = Query(..., description="New status")):
    """Update request status"""
    try:
        return update_request_status(request_id, status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/tickets")
def list_tickets():
    """List all tickets"""
    return {"tickets": TICKET_QUEUE}

@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    """Get specific ticket details"""
    ticket = next((t for t in TICKET_QUEUE if t.ticket_id == ticket_id), None)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@app.post("/tickets")
def create_ticket_endpoint(request_id: int, reason: str, priority: str = "medium"):
    """Create a new ticket"""
    try:
        # Verify request exists
        get_request_by_id(request_id)
        ticket = create_ticket(request_id, reason, priority)
        return ticket
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/notifications")
def send_notification_endpoint(notification: NotificationRequest):
    """Send a notification"""
    result = send_notification(
        recipient=notification.recipient,
        subject=notification.subject,
        message=notification.message,
        request_id=notification.request_id
    )
    return result

@app.get("/notifications")
def list_notifications():
    """List all sent notifications"""
    return {"notifications": NOTIFICATIONS_LOG}

@app.post("/user-input")
def process_user_input(input_data: UserInput):
    """Process user input for workflow steps"""
    return {
        "message": "User input processed successfully",
        "action": input_data.action,
        "response": input_data.user_response,
        "processed_at": datetime.now(),
        "context": input_data.context
    }

@app.get("/workflow/status/{request_id}")
def get_workflow_status(request_id: int):
    """Get comprehensive workflow status"""
    try:
        request = get_request_by_id(request_id)
        related_tickets = [t for t in TICKET_QUEUE if t.request_id == request_id]
        related_notifications = [n for n in NOTIFICATIONS_LOG if n.get("request_id") == request_id]
        
        return {
            "request": request,
            "related_tickets": related_tickets,
            "notifications": related_notifications,
            "workflow_complete": request.status == "approved"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# =======================
# Development Server
# =======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)