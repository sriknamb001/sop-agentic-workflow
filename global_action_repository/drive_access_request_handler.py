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