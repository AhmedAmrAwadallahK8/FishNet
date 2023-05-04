class AbstractNode:
    def __init__(self):
        self.output_name = ""

    def get_output_name(self):
        return self.output_name

    def process(self):
        return None
