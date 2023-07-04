from src.nodes.AbstractNode import AbstractNode

"""
This file is not used in FishNet
"""

class CellAnalysisNode(AbstractNode):
    def __init__(self):
        from src.fishnet import FishNet
        super().__init__(output_name="",
                         requirements=[],
                         user_can_retry=False,
                         node_title="Cell Analysis Settings Node")


        
    def process(self):
        pass
