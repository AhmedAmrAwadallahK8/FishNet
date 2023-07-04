"""
Contains various functions useful for user input

Global Variables:
    valid_responses (list): set of valid response strings
    negative_responses (list): set of valid negative response strings
    positive_responses (list): set of valid positive response strings
    invalid_response_id (int): integer associated with an invalid response
    valid_response_id (int): integer associated with a valid response
    negative_response_id (int): integer associated with a negative response
    positive_response_id (int): integer associated with a positive response
    satisfied_response_id (int): integer associated with a satisfied user 
    response
    quit_response_id (int): integer associated with a quit response
    retry_response_id (int): integer associated with a retry response
    forced_quit_response_id (int): integer associated with a force quite response
"""

valid_responses = ["yes", "y", "no", "n"]
negative_responses = ["no", "n"]
positive_responses = ["yes", "y"]

invalid_response_id = -1
valid_response_id = 2
negative_response_id = 0
positive_response_id = 1
satisfied_response_id = 3
quit_response_id = 4
retry_response_id = 5
forced_quit_response_id = 6

def process_user_input(user_input):
    """
    Takes in a user input, verifies its a valid input, then returns a positive
    or negative response ID depending on the input. If its not a valid input
    then returns a invalid response id

    Args:
        user_input (str): string derived from user input

    Returns:
        int: response id code
    """
    user_input = user_input.lower()
    if user_input in valid_responses:
        if user_input in positive_responses:
            return positive_response_id
        elif user_input in negative_responses:
            return negative_response_id
    else:
        return invalid_response_id

def response_within_range(numeric_response, numeric_range):
    """
    Checks to see if a numeric value is in within the given range.

    Args:
        numeric_response (float): Real numeric value
        numeric_range (list): contains a pair of two numeric values where the
        value at index 0 is the minimum value and index 1 is the maximum

    Returns:
        boolean: True numeric_response in range else False
    """
    if (numeric_response >= numeric_range[0]) and (numeric_response <= numeric_range[1]):
        return True
    else:
        return False

def get_numeric_input_in_range(prompt, numeric_range):
    """
    Prompts user for numeric input until they input a valid value in range 
    then returns the valid input

    Args:
        prompt (str): prompt message given to user
        numeric_range (list): pair of values defining the min and max

    Returns:
        float : valid user input converted to a float
    """
    usr_response = invalid_response_id
    while(usr_response == invalid_response_id):
        usr_response = float(input(prompt))
        if response_within_range(usr_response, numeric_range):
            return usr_response
        else:
            print("Invalid input try again.")
            usr_response = invalid_response_id

def get_categorical_input_set_in_range(prompt, categ_set):
    """
    Prompts the user to return a set of categorical values within the given
    categorical set. Keeps looping until user inputs an acceptable response

    Args:
        prompt (str): prompt message given to user
        categ_set (list): set of valid categorical responses

    Returns:
        list: subset of categ_set that the user specified
    """
    usr_response_state = invalid_response_id
    valid_usr_set = []
    while((usr_response_state == invalid_response_id)):
        valid_usr_set = []
        usr_response_state = valid_response_id
        usr_response = input(prompt)
        usr_set = usr_response.split(',')
        for usr_item in usr_set:
            if usr_item in categ_set:
                if usr_item in valid_usr_set:
                    print("Duplicate input try again")
                    usr_response_state = invalid_response_id
                    break
                else:
                    valid_usr_set.append(usr_item)
            else:
                invalid_input = "\"" + usr_item + "\""
                print(f"{invalid_input} is invalid input try again.")
                print("Common error is extra spaces look closesly at the allowed values and replicate them exactly")
                usr_response_state = invalid_response_id
                break
        if len(valid_usr_set) == 0:
            print("Atleast 1 input is required, try again")
            usr_response_state = invalid_response_id
        
    return valid_usr_set
    

def get_categorical_input_in_range(prompt, categ_set):
    """
    Prompts the user to return a categorical value within the given
    categorical set. Keeps looping until user inputs an acceptable response

    Args:
        prompt (str): prompt message given to user
        categ_set (list): set of valid categorical responses

    Returns:
        str: single category within the set
    """
    usr_response = invalid_response_id
    while(usr_response == invalid_response_id):
        usr_response = input(prompt)
        if usr_response in categ_set:
            return usr_response
        else:
            print("Invalid input try again.")
            usr_response = invalid_response_id
        

def ask_if_user_has_replacement_for_requirement(requirement):
    """
    Informs user that a node does not have the proper requirements present
    in FishNet and asks them if they do have a replacement. This is a yes or 
    no question so this function returns a positive_response_id or 
    negative_response_id depending on input

    Args:
        requirement (str): name of the missing requirement

    Returns:
        int: response id associated with the users input
    """
    prompt = f"This process requires {requirement} which does not currently"
    prompt += f" exist. If you dont have a replacement the program will"
    prompt += f" be forced to exit. Do you have a replacement? "
    response_id = ask_user_for_yes_or_no(prompt)
    return response_id

def ask_user_to_try_again_or_quit():
    """
    Asks user if they would like to retry this node or not. This is a yes
    or no question but returns either a retry_response_id or quit_response_id

    Args:
        Nothing

    Returns:
        int: response id that is either retry_response_id or quit_response_id
    """
    prompt = "Would you like to try this step again?\n"
    prompt += "If you say no the program will assume you are done and exit. "
    response_id = ask_user_for_yes_or_no(prompt)
    if response_id == positive_response_id:
        return retry_response_id
    elif response_id == negative_response_id:
        return quit_response_id

def ask_user_for_yes_or_no(prompt):
    """
    Asks user for a yes or no response. If they user does not give a valid
    yes/no response inform them and try again until they do.

    Args:
        prompt (str): prompt that requires a yes/no response

    Returns:
        int: response_id that is either a positive_response_id or 
        negative_response_id
    """
    response_id = invalid_response_id
    while(response_id == invalid_response_id):
        user_input = input(prompt)
        response_id = process_user_input(user_input)
        if response_id == invalid_response_id:
            print("Invalid response try again.")
            print("We expect either yes or no.")
        else:
            return response_id

def check_if_user_satisified():
    """
    Asks user if they are satisfied then returns a boolean associated with the
    response

    Args:
        Nothing

    Returns:
        boolean: True if user is satisfied otherwise False
    """
    prompt = "Are you satisfied with the displayed output"
    prompt += " for this step? "
    response_id = ask_user_for_yes_or_no(prompt)
    if response_id == positive_response_id:
        return True
    elif response_id == negative_response_id:
        return False

def get_user_feedback_for_node_output():
    """
    Ask user if satisfied, if not then ask them if they would like to try
    again or quit.

    Args:
        Nothing

    Returns:
        int: satisfied_response_id if user initial says yes otherwise 
        retry_response_id or quit_response_id is returned depending on input
    """
    user_is_satisfied = check_if_user_satisified()
    if user_is_satisfied:
        return satisfied_response_id
    else:
        response_id = ask_user_to_try_again_or_quit()
        return response_id
        
