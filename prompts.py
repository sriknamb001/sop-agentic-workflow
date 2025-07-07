def state_llm_prompt(sop_workflow, execution_memory):
    return  f'''
    I want you to act as the action decision agent for the workflow automation task.
    You will be provided with the following information.
    1. Workflow
    2. Execution Memory
    Workflow consists of a logical sequence of actions. Execution Memory consists of the history of
    actions, observations and feedback.
    Your task is to decide the next action based on the workflow and execution memory.
    *** If the execution memory is empty, output the first action from the workflow.
    *** If the feedback for the current entry in execution memory mentions success, output the next
    action as per the logic shown in the workflow.
    *** If the feedback for the current entry in execution memory mentions fail, decide the next action
    as follows.
    – If observation indicates that the user wants to go back to any of the previous actions, perform
    a semantic search in the current execution memory and find the relevant action to help the user.
    Output the action as the next action. Post this you must continue the workflow from where it was
    broken.
    – If the observation clearly indicates that the user has a question or a query, output the action as seek
    external knowledge. If feedback for the seek external knowledge action step is success, output the
    previous valid action from the Execution Memory as the next action.
    *** If the feedback for the last entry in execution memory mentions fail and observation does not
    clearly indicate any of the above scenarios, decide the next action as follows. Carefully evaluate the
    inter-dependence of the current failed action on the previous actions in the execution memory and
    select the most logical previous action that needs to be repeated. Output it as the next action.
    ### Workflow: {sop_workflow} ###
    ### Execution Memory: {execution_memory} ###
    Think step by step and output your thinking as thought.
    Generate all the responses in the JSON format without any deviation. Output JSON should have
    keys "thought, "next_action".
    '''

def action_llm_prompt(action,action_type,action_context):
    return f'''
    I want you to act as the action execution agent for the workflow automation task. You will be
    provided with the following information.
    1. Action in the workflow
    2. Action type
    3. Action context
    Your task is to generate data to execute an action as per the action, action type and action context.
    1. If action type includes ask_user_input, your task is to generate a polite question to the user using
    the action. Output the question as user_interaction.
    2. If action type includes api_call, your task is to extract and assign a correct value to each of the
    required param using the action context. Output the required params and its values.
    3. If action type includes external_knowledge, your task is formulate a short search like query from
    the user’s question/query provided in the action context. Output the search query as search_query.
    4. If action type includes message_to_user, your task is to generate the response to the user as
    shown in the action context. For failure case, inform user that you are retrying the <action>. Output
    the response as user_interaction.
    ### Action: {action}
    ### Action type: {action_type}
    ### Action context: {action_context}
    Think step by step and output your thinking as thought.
    Generate all the responses in the JSON format without any deviation. Output JSON should have
    keys "thought, "user_interaction", "params", "search_query".'''

def user_llm_prompt(question,user_reply,condition):
    return f'''
    I want you to act as the user interaction agent for the workflow automation task.
    You will be provided with the following information.
    1. Question asked to the user
    2. User’s reply
    3. Condition
    Your tasks are as follows.
    1. Verify if the user’s reply satisfies the condition.
    If yes, set input_validation field as success. Otherwise set it as fail.
    2. Extract all the entities from user’s reply and output the slots with key and value per entity. Assign
    a distinctive name to the key as per the question for easy identification.
    3. Generate a response to the user as follows.
    If input_validation is success, provide a one-line acknowledgment message.
    If input_validation field is fail:
    ** If User’s reply clearly shows a question or a query, output the message that you are working on
    it and politely ask user to wait.
    ** If User’s reply is not a question or a query, provide a one-line acknowledgment message.
    ### Question asked to the user: {question}
    ### User’s reply: {user_reply}
    ### Condition: User’s reply which indicates or includes {condition}
    Think step by step and output your thinking as thought.
    Generate all the responses in the JSON format without any deviation. Output JSON should have
    keys "thought, "input_validation", "user_response", "slots".
    '''