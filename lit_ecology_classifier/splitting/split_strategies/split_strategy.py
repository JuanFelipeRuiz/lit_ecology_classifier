"""
Abstract class for split strategies. The use of a abstract class is to define
that each split strategy must implement the perform_split method. 
"""

class SplitStrategy:
    def perform_split(self, data):
        raise NotImplementedError("Must override perform_split")
    