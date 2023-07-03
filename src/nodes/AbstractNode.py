import src.user_interaction as usr_int
import src.file_handler as file_handler
import matplotlib.pyplot as plt
import cv2

class AbstractNode:
    """
    AbstractNode functions similarily to what an abstract class would in C++.
    This class is not designed to be directly created it's purpose is to
    define all the general behavior that is expected of any node within the
    context of our pipeline.
   
    Every child of abstractnode has to define process and initialize_node.
    If you have output to save define save_output.
    Its suggested to replace reinitialize_node when a node allows users to retry
    to save computation and setup time.
    Its only necessary to replace plot_output if you allow users to retry
    as they will need some feedback to make this decision

    Global Attributes:
        NODE_SUCCESS_CODE (int): Number that corresponds to a successful
        completion of a node.
        NODE_FAILURE_CODE (int): Number that corresponds to a node that failed
        to complete.

    Attributes:
        output_name (str): Key that corresponds to the output
        requirements (list): List of keys from other node's that a node needs
        in order to function.
        user_can_retry (boolean): If true unlocks a new logic path that handles
        users being able to retry a node
        node_title (str): This is what is displayed when a node is commencing
        requirements_met (boolean): If requirements of a node are met it's
        able to move forward. Initially set to True but only evaluated in the
        check_requirements method.
        node_status_code (int): status of the nodes success/fail state. By 
        default all nodes are assumed to be in a failure state
        output_pack (any): The type of this depends on how a node stores its
        output. It would be the value stored to the ouput_name key.

    Global Function:

    Methods:
        run(): Performs all critical general functions of a node
        check_requirements(): checks to see if the requirements the node has
        are present within FishNet's pipeline_output. If it does not exist
        allows the user to either substitute with their own data. If the user
        is unable to do that then it terminates the node and returns the 
        failure code.
        process(): This performs all the critical functions of a specific node.
        This method is undefined within this abstract class.
        initialize_node(): Performs all initialization needed prior to 
        processing a node. This method is undefined within this abstract class.
        reinitialize_node(): Allows an alterante initialization path when a 
        node is requested to retry its process step. By default it calls the
        initialize_node method.
        plot_output(): Displays output to the user. This method is undefined
        within this abstract class.
        save_output(): Saves node specific output within the output folder.
        This method is undefined within this abstract class.
        get_output_name(): Returns the output_name attribute
        set_node_as_successful(): Sets the node_status_code attribute with the
        success code
        set_node_as_failed(): Sets the node_status_code attribute with the
        failure code
        node_intro_msg(): Prints the introduction message for a node
        give_and_save_node_data(): If the node is succesful save output and
        give output relevant for other nodes to FishNet's pipeline_output
        global variable.
        give_fishnet_output(): Directly calls FishNet's store_output function
        using the key value pair attributes as input.
        save_img(img, img_name): Grabs the output folder path from FishNet
        then saves the given image using the given path.
    """
    NODE_SUCCESS_CODE = 1000
    NODE_FAILURE_CODE = 1001
    def __init__(self,
                 output_name="",
                 requirements=[],
                 user_can_retry=False,
                 node_title="Uninitialized Node Title"):
        self.output_name = output_name
        self.requirements = requirements
        self.user_can_retry = user_can_retry
        self.node_title = node_title
        self.requirements_met = True
        self.node_status_code = AbstractNode.NODE_FAILURE_CODE
        self.output_pack = None

    def run(self):
        """
        Performs all required general functions of a node

        Args:
            Nothing

        Returns: 
            Nothing
        """
        self.node_intro_msg()
        self.check_requirements()
        if self.requirements_met is False:
            return self.node_status_code

        self.initialize_node()
        if self.user_can_retry:
            usr_feedback = usr_int.retry_response_id
            first_pass = True
            while usr_feedback == usr_int.retry_response_id:
                if not first_pass:
                    self.reinitialize_node()

                node_output = self.process()
                self.plot_output()
                usr_feedback = usr_int.get_user_feedback_for_node_output()

                if usr_feedback == usr_int.satisfied_response_id:
                    self.give_and_save_node_data()
                    return self.node_status_code
                elif usr_feedback == usr_int.quit_response_id:
                    return self.node_status_code
                if first_pass:
                    first_pass = False
        else:
            node_output = self.process()
            self.give_and_save_node_data()
            return self.node_status_code

    def check_requirements(self):
        """
        Verifies if the required data for this node is stored within
        FishNet's pipeline_output. If not provide a way for users to input
        their own substitutes. If user is unable then set the requirements_met
        attribute to false.

        Args:
            Nothing

        Returns: 
            Nothing
        """
        from src.fishnet import FishNet
        for requirement in self.requirements:
            if requirement not in FishNet.pipeline_output.keys():
                user_response_id = usr_int.ask_if_user_has_replacement_for_requirement(requirement)
                if user_response_id == usr_int.positive_response_id:
                    loaded_img = file_handler.load_img_file()
                    FishNet.pipeline_output[requirement] = loaded_img
                elif user_response_id == usr_int.negative_response_id:
                    self.requirements_met = False

        
    def process(self):
        """
        Prints a default warning message

        Args:
            Nothing

        Returns: 
            Nothing
        """
        print("This is the default process method that does nothing")

    def initialize_node(self):
        """
        Prints a default warning message

        Args:
            Nothing

        Returns: 
            Nothing
        """
        print("This is the default initialization method that does nothing")

    def reinitialize_node(self):
        """
        Calls initialize_node

        Args:
            Nothing

        Returns: 
            Nothing
        """
        self.initialize_node()

    def plot_output(self):
        """
        Prints a default warning message

        Args:
            Nothing

        Returns: 
            Nothing
        """
        print("This is the default plot method that does nothing")

    def save_output(self):
        """
        Does nothing

        Args:
            Nothing

        Returns: 
            Nothing
        """
        pass

    def get_output_name(self):
        """
        A getter method that returns output_name

        Args:
            Nothing

        Returns: 
            str: The output_name of the node
        """
        return self.output_name

    def set_node_as_successful(self):
        """
        Sets node_status_code with the NODE_SUCCESS_CODE

        Args:
            Nothing

        Returns: 
            Nothing
        """
        self.node_status_code = AbstractNode.NODE_SUCCESS_CODE

    def set_node_as_failed(self):
        """
        Sets node_status_code with the NODE_FAILURE_CODE

        Args:
            Nothing

        Returns: 
            Nothing
        """
        self.node_status_code = AbstractNode.NODE_FAILURE_CODE

    def node_intro_msg(self):
        """
        Prints the introduction of a node

        Args:
            Nothing

        Returns: 
            Nothing
        """
        prompt = f"\n---- Commencing {self.node_title} ----\n"
        print(prompt)

    def give_and_save_node_data(self):
        """
        If the node is succesful it saves the output with the systems files
        and within FishNet's pipeline_output

        Args:
            Nothing

        Returns: 
            Nothing
        """
        if self.node_status_code == AbstractNode.NODE_SUCCESS_CODE:
            self.save_output()
            self.give_fishnet_output()


    def give_fishnet_output(self):
        """
        Directly handles communicating with FishNet's pipeline_output and the
        key value pair (output_name, output_pack)

        Args:
            Nothing

        Returns: 
            Nothing
        """
        from src.fishnet import FishNet
        FishNet.store_output(self.output_pack, self.output_name)

    def save_img(self, img, img_name):
        """
        Combines FishNet's output folder and img_name to create a full file 
        path then saves the image using opencv

        Args:
            img (ndarray): Numpy array derived from image data
            img_name (str): Node relative img path

        Returns: 
            Nothing
        """
        from src.fishnet import FishNet
        folder_name = FishNet.save_folder
        img_file_path = folder_name + img_name
        cv2.imwrite(img_file_path, img)
