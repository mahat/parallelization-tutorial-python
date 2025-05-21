from typing import List

class Machine:
    """
    Abstract representation machine/gpu instance to validate parallization patterns
    Atributes:
        data: matrix or model piece to doing parallel work
        rank: rank of the instance
        instance_list: other instances to communicate if rank == 0 
        stats: statistics of compuations or communications
    methods:

    
    """

    def __init__(self,rank,data,instance_list: List["Machine"]):
        self.rank = rank
        self.data = data
        self.instance_list = instance_list
        self.stats = {}


    def get_data(self):
        """
        returns data GET method
        """
        return self.data
    
    def set_data(self,data):
        """
        POST method
        """
        self.data = data
        # any return for confirmation ? 
    
    def all_reduce(self,reduce_function):
        """
        method for reducing from all instances and broadcast to them
        reduce function must be two variables function i.e. lambda a,b: a + b -> sum reduce
        """
        if self.rank == 0: # only rank 0 machine can do this operations
            for i in self.instance_list:
                self.data = reduce_function(self.data,i.get_data())
            # broad cast
            self.broadcast()
        else:
            raise Exception(f'Only rank 0 machine can do broad all reduce current rank is {self.rank}')

    def broadcast(self):
        """
        send data to all instances
        """
        if self.rank == 0:
            for i in self.instance_list:
                i.set_data(self.data)
        else:
            raise Exception(f'Only rank 0 machine can do broad casting current rank is {self.rank}')
        
    def reduce_scatter(self,reduce_function,shard_function):
        pass



