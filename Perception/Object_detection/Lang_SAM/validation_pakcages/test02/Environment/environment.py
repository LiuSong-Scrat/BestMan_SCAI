
from .flexiv_experiment_table import *
from .database import *

class environment():

    """
    The general environment class deisgned to carry all sub-environment modules (e.g. flexiv experiment table, container cabinate etc.)
    This should be modified when the permanent environment setting is changed
    """

    def __init__(self):
        
        self.database = DataBase().database
        self.platform1 = flexiv_experiment_table()
        self.env_dict = {'flexiv experiment table': self.platform1}
    
    def add_obj(self, sub_env_name, obj_name, obj_coor, other_info=None, avoid_collision=False):

        self.env_dict[sub_env_name].add_obj(self.database, obj_name, obj_coor, other_info, avoid_collision)

    def remove_obj(self, sub_env_name, obj_name):

        self.env_dict[sub_env_name].remove_obj(obj_name)
    
    def visualize_env(self, sub_env_name):

        self.env_dict[sub_env_name].visualize_env()

    def check_collision(self, sub_env_name, obj_name, obj_coor):

        self.env_dict[sub_env_name].check_collision(self.database, obj_name, obj_coor)
    
    def explore_valid_placing_coordinate(self, sub_env_name, obj_name, region=None, clearance=None):

        return self.env_dict[sub_env_name].explore_valid_placing_coordinate(self.database, obj_name, region, clearance)

    def export_environment(self, sub_env_name):

        return self.env_dict[sub_env_name].export_environment()
