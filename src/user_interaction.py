valid_responses = ["yes", "y", "no", "n"]
negative_responses = ["no", "n"]
positive_responses = ["yes", "y"]

invalid_response_id = 0
positive_response_id = 1
negative_response_id = 2
satisfied_response_id = 3
quit_response_id = 4
retry_response_id = 5

def process_user_input(user_input):
    user_input = user_input.lower()
    if user_input in valid_responses:
        if user_input in positive_responses:
            return positive_response_id
        elif user_input in negative_responses:
            return negative_response_id
    else:
        return invalid_response_id

def ask_user_to_try_again_or_quit():
    prompt = "Would you like to try this step again?\n"
    prompt += "If you say no the program will assume you are done and exit. "
    user_input = input(prompt)
    response_id = process_user_input(user_input)
    if response_id == positive_response_id:
        return True
    elif response_id == negative_response_id:
        return False

def check_if_user_satisified():
    prompt = "Are you satisfied with the displayed output"
    prompt += " for this step? "
    response_id = invalid_response_id
    while(response_id == invalid_response_id):
        user_input = input(prompt)
        response_id = process_user_input(user_input)
        if response_id == positive_response_id:
            return True
        elif response_id == negative_response_id:
            return False
        elif response_id == invalid_response_id:
            print("Invalid response try again.")
            print("We expect either yes or no.")

def get_user_feedback_for_node_output():
    user_is_satisfied = check_if_user_satisified()
    if user_is_satisfied:
        return satisfied_response_id
    else:
        user_wants_retry = ask_user_to_try_again_or_quit()
        if user_wants_retry:
            return retry_response_id
        else:
            return quit_response_id
        
