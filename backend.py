from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta

app = FastAPI()

# =======================
# Models
# =======================

class Request(BaseModel):
    request_id: int
    status: str
    created_at: datetime

class Ticket(BaseModel):
    ticket_id: int
    request_id: int
    reason: str
    created_at: datetime

# =======================
# In-memory DB
# =======================

REQUESTS_DB = {
    101: Request(request_id=101, status="approved", created_at=datetime.now() - timedelta(hours=10)),
    102: Request(request_id=102, status="in-progress", created_at=datetime.now() - timedelta(hours=50)),
    103: Request(request_id=103, status="disapproved", created_at=datetime.now() - timedelta(hours=80)),
    104: Request(request_id=104, status="disapproved", created_at=datetime.now() - timedelta(hours=100)),
}

TICKET_QUEUE: List[Ticket] = []

# =======================
# Service Layer
# =======================

def get_request_by_id(request_id: int) -> Request:
    req = REQUESTS_DB.get(request_id)
    if not req:
        raise ValueError("Request ID not found.")
    return req

def create_ticket(request_id: int, reason: str) -> Ticket:
    ticket_id = len(TICKET_QUEUE) + 1
    ticket = Ticket(
        ticket_id=ticket_id,
        request_id=request_id,
        reason=reason,
        created_at=datetime.now()
    )
    TICKET_QUEUE.append(ticket)
    return ticket

def process_drive_access_sop(request_id: int) -> dict:
    request = get_request_by_id(request_id)
    age_hours = (datetime.now() - request.created_at).total_seconds() / 3600

    if request.status == "approved":
        return {"message": "Drive access already approved."}

    if request.status in ["in-progress", "disapproved"]:
        if age_hours <= 72:
            return {"message": "Request is being processed. Please check back later."}
        else:
            ticket = create_ticket(request_id, reason="Drive Access Re-Approval Needed")
            return {
                "message": "Access request delayed. A new approval ticket has been created.",
                "ticket_id": ticket.ticket_id
            }

# =======================
# API Routes
# =======================

@app.get("/")
def root():
    return {"message": "SOP API is running."}

@app.get("/sop/drive-access")
def drive_access_sop(request_id: int = Query(..., description="ID of the access request")):
    try:
        return process_drive_access_sop(request_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/requests/{request_id}")
def get_request(request_id: int):
    try:
        request = get_request_by_id(request_id)
        return request
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/tickets")
def list_tickets():
    return TICKET_QUEUE
