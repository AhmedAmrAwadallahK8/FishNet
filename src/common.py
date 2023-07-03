
class TempPipeline:
    """
    Responsible for provide a method for an external class to interact with a
    sequence of nodes

    Global Attributes:
        Nothing

    Global Function: 
        Nothing

    Attributes:
        node_idx (int): The index of the current node
        nodes (list): A list of nodes
        end_node (str): The final "node" in the list

    Methods:
        advance(): Advances the node index by 1 if the current node is not the
        end node
        is_not_finished(): Returns a result that indicates whether the pipeline
        has reached the end node or not
        add_node(node): Not implemented
        run_node(): Runs the current node and returns its exit code
    """

    def __init__(self, nodes):
        self.node_idx = 0
        self.nodes = nodes
        self.end_node = "ENDNODE"
        self.nodes.append(self.end_node)

    def advance(self):
        """
        If the node is not an end node the current node index is advanced by 1
        
        Args:
            Nothing

        Returns:
            boolean: Returns True if it advanced otherised False
        """
        node = self.nodes[self.node_idx]
        if node is not self.end_node:
            self.node_idx += 1
            return True
        else:
            return False

    def is_not_finished(self):
        """
        Returns a boolean that indicates whether the pipeline is finished. A 
        pipeline is considered done when the current index poitns to the end
        node

        Args:
            Nothing

        Returns:
            boolean: Returns False if current node is equivalent to end_node 
            else False
        """
        node = self.nodes[self.node_idx]
        if node == self.end_node:
            return False
        else:
            return True

    def add_node(self, node):
        """
        Not Implemented
        """
        pass

    def run_node(self):
        """
        Selects the current node based on the node_idx and calls its run method.
        The node run method returns some ouput indicating the success or failure
        status of the node which is returned by this call

        Args:
            Nothing
        
        Returns:
            int: Integer error code associated with the node
        """
        node = self.nodes[self.node_idx]
        if node is not self.end_node:
            node_exit_code = node.run()
            return node_exit_code

# Below is old pipeline functionality that is not relevant for this program
## For now these are just wrappers and don't do much. Might delete in the future.
## All functions inherit off of the base Node class below.
## My plan was to save some data on node execution like execution time, success, etc.


# import time

# class Node:
#     def __init__(self):
#         pass
# 
#     # def process(self, image):
#     #     start_time = time.time()
#     #     # Call the implementation-specific processing code
#     #     self._process(image)
#     #     end_time = time.time()
#     #     print(f"Processing took {end_time - start_time} seconds")
# 
# 
#     def process(self, image):
#         # This method should be implemented by the subclasses
#         pass


# class Pipeline:
#     def __init__(self, nodes):
#         self.nodes = nodes
# 
#     def add_node(self, node):
#         self.nodes.append(node)
#         
#     def process(self, image):
#         for node in self.nodes:
#             image = node.process(image)
#         return image

