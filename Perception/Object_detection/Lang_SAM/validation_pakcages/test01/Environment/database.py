
import numpy as np
import matplotlib.pyplot as plt

class DataBase():

    def __init__(self):
        """
        The foundamental knowledgebase of objects

        obj_class:    Currently not in use. meant for providing more information to GPT in te furture
        obj_height:   unit: m
        diameter:     Object assumed to be with square bounding box. unit: m
        grasp_depth:  How far the actual grapsing point is from the top of the object. As an early implementation, this serve the same purpose of the grasp_offset. 
                      Can consider remove this inthe future.
        grasp_offset: How far the actual grapsing point is from center of the object.  
                      - center_grasp: gripper grapsing at the top with the gripper center being the object's top center.
                      - side_grasp: gripper grasping from side, typically with inclined grapsing orientation.
                      - edge_grasp: gripper grasp the edge of containers. This usually works with container with large diameter, which is larger than the gripper opening.
        grasp_force:  The grasp for of gipper. ranging from 0-255 according to common gripper setting. It is recommanded to keep this lower than 20 when handling glass.
        """
        
        self.database = {'glass bottle': experiment_obj(name = 'glass bottle', 
                                                        obj_class = 'container',
                                                        obj_height = 0.175, 
                                                        diameter = 0.080, 
                                                        grasp_depth = 0.000, 
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.030]},
                                                        grasp_force = 20),
                         'glass conical flask': experiment_obj(name = 'glass conical flask', 
                                                        obj_class = 'container',
                                                        obj_height = 0.142, 
                                                        diameter = 0.083, 
                                                        grasp_depth = 0.005,
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.013]},
                                                        grasp_force = 10),
                         'glass graduated cylinder': experiment_obj(name = 'glass graduated cylinder', 
                                                        obj_class = 'container',
                                                        obj_height = 0.170, 
                                                        diameter = 0.042, 
                                                        grasp_depth = 0.010,
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.040]},
                                                        grasp_force = 10),
                         'glass beaker': experiment_obj(name = 'glass beaker', 
                                                        obj_class = 'container',
                                                        obj_height = 0.125, 
                                                        diameter = 0.088, 
                                                        grasp_depth = 0.005, 
                                                        grasp_offset = {'edge_grasp': [0.000, -0.044, -0.020]},
                                                        grasp_force = 10),
                         'glass tall beaker': experiment_obj(name = 'glass tall beaker', 
                                                        obj_class = 'container',
                                                        obj_height = 0.130, 
                                                        diameter = 0.07, #SRC 0.63
                                                        grasp_depth = 0.005, 
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.030]},
                                                        grasp_force = 15),
                         'plastic cup': experiment_obj(name = 'plastic cup', 
                                                        obj_class = 'container',
                                                        obj_height = 0.070, 
                                                        diameter = 0.070, 
                                                        grasp_depth = 0.005, 
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.030],
                                                                        'edge_grasp': [0.000, -0.035, 0.000]},
                                                        grasp_force = 0),
                         'tape roll': experiment_obj(name = 'tape roll', 
                                                        obj_class = 'tool',
                                                        obj_height = 0.050, 
                                                        diameter = 0.110, 
                                                        grasp_depth = 0.005, 
                                                        grasp_offset = {'edge_grasp': [0.000, -0.055, 0.000]},
                                                        grasp_force = 50),
                         'glass container with lid': experiment_obj(name = 'glass container with lid', 
                                                        obj_class = 'container',
                                                        obj_height = 0.178, 
                                                        diameter = 0.080, 
                                                        grasp_depth = 0.005, 
                                                        grasp_offset = {'center_grasp': [0.000, 0.000, 0.000],
                                                                        'side_grasp': [0.000, 0.000, -0.010]},
                                                        grasp_force = 10),
                        }


class experiment_obj():

    def __init__(self, name, obj_class, obj_height, diameter, grasp_depth, grasp_offset, grasp_force):
        
        self.name = name
        self.obj_class = obj_class
        self.obj_height = obj_height
        self.diameter = diameter
        self.grasp_depth = grasp_depth
        self.grasp_offset = grasp_offset
        self.grasp_force = grasp_force

