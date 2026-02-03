import numpy as np
import matplotlib.pyplot as plt
import json

class flexiv_experiment_table():

    """
    A 2D environemnt class of the flexiv experiment table. This class includes the following functions:
    add_obj(): add the specified object into the enviornment with any additional informaiton
    remove_obj(): remvoe the specified object
    visualize_env(): plot the bounding box of the objects in the environment
    check_collision(): check whether collision will happen if the object is place at the specified coordinate
    explore_valid_placing_coordinate(): automatically finds the coordinate to place the object in a collision-free manner
    export_environment(): export the environment object list in form of json
    
    """

    def __init__(self):
        
        self.bound = [-0.600, 0.600, 0.000, 0.700]    # [Ymin, Ymax, Xmin, Xmax] in flexiv's coordinate
        self.object_list = {}
        self.grid_resolution = 0.001
        self.grid = np.zeros((int((self.bound[3]-self.bound[2]) / self.grid_resolution),
                              int((self.bound[1]-self.bound[0]) / self.grid_resolution)))

    
    def add_obj(self, database, obj_name, obj_coor, other_info=None, avoid_collision=False):

        """
        Description: 
        Add an obejct into the environment

        Inputs:
        database: the pre-defiend database of the objects
        obj_name (str): the name of the object with index ('beaker 1' etc.).
        obj_coor (List[float]): the coordinate of the center of the object (object bounding box). Input should follow (Y,X) of the real-world base's coordinate
        other_info (dict{str}): the additional information of the object in addition to bounding box and coordinate
        avoid_collision (bool): if True, the object will not be placed if collision happens;
                                if False, the object will still be placed in the environment with warning generated
        
        Outputs:
        None

        IMPORTANT !!!!!!!
        
        The definition of coordinate parameter obj_coor, the format should be [Y, X] where X, Y are both the actual coordinate of the obejct presented in the robot's base frame.
        The reason for doing this is: the definition of the X, Y axis in a matrix is different from that of the robot base's frame (viewing from the robot's base)

        """
        
        if obj_name in self.object_list:
            print("Object with the same name in environment. Please check")
            return

        obj_name_database = obj_name
        while not obj_name_database[-1].isalpha(): obj_name_database = obj_name_database[:-1]
        any_collision = self.check_collision(database, obj_name_database, obj_coor)
        if any_collision: 
            print("WARNING! New object " + obj_name + " collides with exisiting objects. Please check coordinate.")
            if avoid_collision:
                print("Object not added because avoid_collision is turned on.")
                return 
        bbox_size = database[obj_name_database].diameter
        occx, occy = self.calculate_occupancy_bound(bbox_size, obj_coor)
        self.add_occupancy(occx, occy)
        self.object_list[obj_name] = {
                                      "object_coordinate": obj_coor,
                                      "object_bbox_size": bbox_size,
                                      }
        
        if other_info is not None: 
            for key in other_info:
                self.object_list[obj_name][key] = other_info[key]
        print("Successfully added object: " + obj_name)

    
    def remove_obj(self, obj_name):
        """
        Description:
        remove an object from the environment

        Input:
        obj_name (str): object name in the environment with index, e.g., 'beaker 1'

        Output:
        None
        """

        if obj_name not in self.object_list:
            print("Object does no exist")
            return
        obj_coor = self.object_list[obj_name]["object_coordinate"]
        bbox_size = self.object_list[obj_name]["object_bbox_size"]
        occx, occy = self.calculate_occupancy_bound(bbox_size, obj_coor)
        self.clear_occupancy(occx, occy)
        self.object_list.pop(obj_name)

        print("Successfully removed object: " + obj_name)

    def visualize_env(self):
        """
        Description:
        Show the bounding box and layout the the 2D environment

        Input:
        None

        Output:
        None
        """
        
        fig, ax = plt.subplots(figsize=((self.bound[1]-self.bound[0])*10, (self.bound[3]-self.bound[2])*10))
        for obj in self.object_list:
            coor = self.object_list[obj]["object_coordinate"]
            size = self.object_list[obj]["object_bbox_size"]
            xs = [coor[0]-size*0.5, coor[0]+size*0.5, coor[0]+size*0.5, coor[0]-size*0.5, coor[0]-size*0.5]
            ys = [coor[1]-size*0.5, coor[1]-size*0.5, coor[1]+size*0.5, coor[1]+size*0.5, coor[1]-size*0.5]
            ax.plot(xs, ys, color="red")
        ax.set_xlim(self.bound[1], self.bound[0])
        ax.set_ylim(self.bound[2], self.bound[3])
        plt.grid()
        plt.show()

    def check_collision(self, database, obj_name, obj_coor, clearance=0.050):
        """
        Description:
        Check that given an object and a desired place coordinate, whether collision will happen

        Input:
        database: the pre-defiend database of the objects
        obj_name (str): the name of the object with index ('beaker 1' etc.).
        obj_coor (List[float]): the coordinate of the center of the object (object bounding box). Input should follow (Y,X) of the real-world base's coordinate
        clearance (float): minimum distance between objects
        """

        obj_name_database = obj_name
        while not obj_name_database[-1].isalpha(): obj_name_database = obj_name_database[:-1]
        bbox_size = database[obj_name_database].diameter + clearance
        occx, occy = self.calculate_occupancy_bound(bbox_size, obj_coor)
        if (self.grid[occy[0]:occy[1], occx[0]:occx[1]] == 1).any(): return True
        else: return False

    def calculate_occupancy_bound(self, bbox_size, obj_coor):
        """
        Description:
        Calculate the boundary of bound boxes in terms of i,j coordinate in the grid map (matrix)

        Input:
        bbox_size (float): the real world bounding box size in m.
        obj_coor (List[float]): the real world object coordinate. The same [Y,X] format as in add_obj.

        Output:
        occx (List[int]): The X index of start and end of bounding box
        occy (List[int]): The Y index of start and end of bounding box
        """

        occx = [int((obj_coor[0]-bbox_size*0.5-self.bound[0])/self.grid_resolution),
                int((obj_coor[0]+bbox_size*0.5-self.bound[0])/self.grid_resolution)]
        occy = [int((obj_coor[1]-bbox_size*0.5-self.bound[2])/self.grid_resolution),
                int((obj_coor[1]+bbox_size*0.5-self.bound[2])/self.grid_resolution)]
        return occx, occy

    def add_occupancy(self, occx, occy):

        self.grid[occy[0]:occy[1], occx[0]:occx[1]] = 1

    def clear_occupancy(self, occx, occy):

        self.grid[occy[0]:occy[1], occx[0]:occx[1]] = 0

    def explore_valid_placing_coordinate(self, database, obj_name, region=None, clearance=None):
        """
        Description:
        Automatically find the coordinate to place an object in the designated region. 
        The range fo region should follow exactly that in the main file (currently sample_runthrough.py and GPT propt file)

        Input:
        database: the pre-defiend database of the objects
        obj_name (str): the name of the object with index ('beaker 1' etc.)
        region (str): the name of designated region. Check in the following for the exact name of regions
        clearance (float): minimum distance between objects

        Output:
        coor (List[float]): the real-world coordinate of the legal coordinate. This follows the formate of [X,Y], NOT [Y,X].
        """
        
        if region == None: region_bound = [0.650, 0.400, -0.400, -0.100]
        elif region == 'preparation region': region_bound = [0.300, 0.650, 0.500, 0.300]
        elif region == 'experiment region': region_bound = [0.650, 0.500, 0.20, -0.100]
        # elif region == 'experiment region': region_bound = [0.650, 0.500, 0.150, -0.100]
        elif region == 'complete region': region_bound = [0.650, 0.300, -0.500, -0.200]

        if clearance == None: clearance = 0.080
        grids = 60
        gx = np.linspace(region_bound[0], region_bound[1], grids)
        gy = np.linspace(region_bound[2], region_bound[3], grids)
        for i in range(grids):
            for j in range(grids):
                if not self.check_collision(database, obj_name, [gy[grids-1-i], gx[j]], clearance=clearance): return gx[j], gy[grids-1-i]  #Right to Left Collision Avoid
        return None
    
    def export_environment(self):
        """
        Description:
        Export the environment / save the important information (mostly dynamic / time-variant infromation) in json format

        Input:
        None

        Output:
        None
        """

        with open('/home/yan/BestMan_Chemistry_Test_SONG/Environment/flexiv_table_env.json', 'w') as fp:
            json.dump(self.object_list, fp)

