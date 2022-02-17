from abc import ABC, abstractmethod
from enum import Enum
import hashlib

import wandb

class DataType(Enum):
    num = 0
    string = 1
    image = 2
    tensor = 3
    dict = 4

class Logger(ABC):
    '''
    This is an abstract class for logging data.
    '''

    @abstractmethod
    def log(self, data_type, data, key = None):
        '''
        This function logs the various types of data.

        Keyword arguments:
        data_type -- <DataType> The data type of the data
        data -- The data to be logged
        key -- An optional key/name of the data

        '''
        pass
    
class EmptyLogger(Logger):
    def __init__(self, name, id, config_dict):
        pass

    def log(self, data_type, data, key = None):
        return

class PrintLogger(Logger):
    def __init__(self, name, id, config_dict):
        pass

    def log(self, data_type, data, key = None):
        if key is not None:
            key = str(key)+': '
        else:
            key = ''

        if(data_type == DataType.num):
            print(key+str(data))
        elif(data_type == DataType.string):
            print(key+data)
        elif(data_type == DataType.image):
            print(key+str(data))
        elif(data_type == DataType.tensor):
            print(key+str(data))
        elif(data_type == DataType.dict):
            print(key+str(data))

class WandbLogger(Logger):
    def __init__(self, name, id, config_dict):
        s = name+str(id)
        hash_id = hashlib.md5(s.encode()).hexdigest()[-8:]
        wandb.init(project="iil-cartpole", entity="ruke1ire", name = name, resume = "allow", id = hash_id, config = config_dict)

    def log(self, data_type, data, key):
        if(data_type == DataType.dict):
            dictionary = data
        else:
            dictionary = dict()
            dictionary[key] = data
        wandb.log(dictionary)