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
        return None

    def ask_user_if_they_have_substitute_for_requirement(requirement):
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
        return self.process()
