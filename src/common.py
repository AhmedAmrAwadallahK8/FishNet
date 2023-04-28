import time

## For now these are just wrappers and don't do much. Might delete in the future.
## All functions inherit off of the base Node class below.
## My plan was to save some data on node execution like execution time, success, etc.



class Node:
    def __init__(self):
        pass

    # def process(self, image):
    #     start_time = time.time()
    #     # Call the implementation-specific processing code
    #     self._process(image)
    #     end_time = time.time()
    #     print(f"Processing took {end_time - start_time} seconds")


    def process(self, image):
        # This method should be implemented by the subclasses
        pass


class Pipeline:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)
        
    def process(self, image):
        for node in self.nodes:
            image = node.process(image)
        return image