import os
import glob
import cv2
import nd2
from multiprocessing import Pool
from src.common import Pipeline
from src.loader import ImageContainer 
from src.preprocessing import *
from src.segmentation import SegmentAnything

import os
import uuid
import time
import nd2
import argparse
from typing import List
import numpy as np
from src.loader import ND2Loader, Node
from src.visualization import ImageViewer

class Pipeline:
    """
    A class representing a node-based pipeline executor.

    Attributes:
    ----------
    nodes (List[Node]): A list of nodes in the pipeline.
    current_node_index (int): The index of the current node being processed.
    """
    def __init__(self, nodes: List[Node]):
        # Initialize the pipeline with a list of nodes and a current node index
        self.nodes = nodes
        self.current_node_index = 0

    def process_node(self, node):
        # Get the current node and its inputs
        inputs = self.get_inputs(node)

        print('inputs for current node:')
        print(inputs)

        # If the inputs are None or the node expects no inputs, continue processing
        if node.check_valid_inputs(inputs):
            start_time = time.time()
            outputs = node.process(inputs)
            print(outputs)
            end_time = time.time()
            node.execution_time = end_time - start_time
            
            # Check the node's outputs and handle them accordingly
            if node.check_valid_outputs(outputs):
                self.handle_outputs(node, outputs)
            else:
                # If the outputs are unexpected, prompt the user for action
                print(f"Node '{node.name}' produced unexpected outputs.")
                self.prompt_user(node)
        else:
            # If the inputs are unexpected, prompt the user for action
            print(f"Node '{node.name}' received unexpected inputs.")
            self.prompt_user(node)
        
    def process(self):
        # Process the pipeline step by step
        while self.current_node_index < len(self.nodes):
            self.process_node(self.nodes[self.current_node_index])
                
    def get_inputs(self, node):
        # Get the inputs for a node from the previous node's outputs or from user input
        if node.valid_inputs is None:
            return None
        
        print('getting inputs')
        if self.current_node_index == 0:
            print(f"Node '{node.name}' requires inputs.")
            return node.get_user_inputs()
        else:
            print('getting prev node')
            prev_node = self.nodes[self.current_node_index - 1]
            print(prev_node)
            print(prev_node.outputs)
            if prev_node.outputs is None:
                print(f"Node '{prev_node.name}' did not produce any outputs.")
                self.prompt_user(prev_node)
                return self.get_inputs(node)
            elif node.inputs != prev_node.outputs.__class__:
                print(f"Node '{prev_node.name}' produced outputs of unexpected type.")
                self.prompt_user(prev_node)
                return self.get_inputs(node)
            else:
                return prev_node.outputs
                
    def handle_outputs(self, node, outputs):
        # Save the node's outputs and prompt the user for action
        node.outputs = outputs
        self.prompt_user(node)
        
    def prompt_user(self, node):
        # Prompt the user for action (advance, repeat, or exit)
        while True:
            user_input = input(f"Node '{node.name}' completed. [A]dvance, [R]epeat, or [E]xit? ")
            if user_input.lower() == "a":
                self.current_node_index += 1
                break
            elif user_input.lower() == "r":
                self.process_node(node)
                break
            elif user_input.lower() == "e":
                self.current_node_index = len(self.nodes)
                break
            else:
                print("Invalid input.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pipeline.')
    parser.add_argument(
        '--stepwise', 
        action='store_true', 
        default=True,
        help='Run the pipeline step by step.'
    )
    args = parser.parse_args()

    filename = r'D:\python_projects\fishnet-old\input\wt.nd2'

    nodes = [
        ND2Loader(name="Loader", filename=filename),
        # ImageViewer(name="Viewer1", colormap="blue"), # current options: None, blue, yellow
        # SegmentAnything(name="Segmenter"), # Not working
    ] 

    pipeline = Pipeline(nodes)

    if args.stepwise:
        pipeline.process()

    # TODO: option to run the pipeline without user input
    # else:
    #     pipeline.run()