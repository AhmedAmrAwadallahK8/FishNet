import src.user_interaction as usr_int

class AbstractNode:
    def __init__(self, output_name="", requirements=[], user_can_retry=False):
        self.requirement_exists = {}
        self.output_name = output_name
        self.requirements = requirements
        self.user_can_retry = user_can_retry
        self.requirements_met = True

    def get_output_name(self):
        return self.output_name

    def process(self):
        print("This is the default process method that does nothing")
        return None

    def initialize_node(self):
        print("This is the default initialization method that does nothing")
        pass

    def reinitialize_node(self):
        self.initialize_node()

    def plot_output(self):
        print("This is the default plot method that does nothing")
        pass

    def ask_user_if_they_have_substitute_for_requirement(self, requirement):
        prompt = "The requirement {requirement} has not been met by a"
        prompt += "step earlier in the pipeline. Do you have a replacement?"
        

    def check_requirements(self):
        from src.fishnet import FishNet
        for requirement in self.requirements:
            if requirement not in FishNet.pipeline_output.keys():
                self.requirement_check[requirement] = False
                self.requirements_met = False
            elif requirement in FishNet.pipeline_output.keys():
                self.requirement_check[requirement] = True

    def run(self):
        self.check_requirements()
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
                # Close output maybe?

                if usr_feedback == usr_int.satisfied_response_id:
                    return node_output
                elif usr_feedback == usr_int.quit_response_id:
                    return None

                if first_pass:
                    first_pass = False
        else:
            node_output = self.process()
            return node_output
