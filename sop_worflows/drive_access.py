sop_content = '''
Check if user has provided request ID
Query the request status from the user
If request is approved, inform user
If request is in progress or disapproved and under 72 hours, inform user to wait
If request is over 72 hours old, create reapproval ticket
Notify user about ticket creation
'''