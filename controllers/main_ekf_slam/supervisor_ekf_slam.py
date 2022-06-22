import copy
from controller import Supervisor, Node
from time import sleep
import numpy as np
import math

supervisor = None
robot_node = None
target_node = None

start_translation = None
start_rotation = None
np.set_printoptions(precision=3, suppress=True)


def init_supervisor():
    print(' ')
    print(' ')
    print("Supervisor initiated")
    print(' ')
    print(' ')
    global supervisor, robot_node, target_node, start_translation, start_rotation

    # create the Supervisor instance.
    supervisor = Supervisor()

    # do this once only
    root = supervisor.getRoot()
    root_children_field = root.getField("children")
    robot_node = None
    target_node = None
    for idx in range(root_children_field.getCount()):
        if root_children_field.getMFNode(idx).getTypeName() == "E-puck":
            robot_node = root_children_field.getMFNode(idx)

    start_translation = copy.copy(robot_node.getField("translation").getSFVec3f())
    start_rotation = copy.copy(robot_node.getField("rotation").getSFRotation())


def supervisor_reset_to_home():
    global robot_node
    pos_field = robot_node.getField("translation")
    pos_field.setSFVec3f(start_translation)
    pos_field = robot_node.getField("rotation")
    pos_field.setSFRotation(start_rotation)
    print(' ')
    print(' ')
    print("*****Supervisor Reset robot to start position*****")
    print(' ')
    print(' ')


def supervisor_get_obstacle_positions():
    global supervisor
    coords_list = []

    root_children_field = supervisor.getRoot().getField("children")
    for idx in range(root_children_field.getCount()):
        if root_children_field.getMFNode(idx).getTypeName() == "CardboardBox":
            box_node = root_children_field.getMFNode(idx)
            box_coords = box_node.getField("translation").getSFVec3f()
            coords_list.append(np.array([box_coords[0], 1 - box_coords[2]]))

    return coords_list




def supervisor_get_robot_pose():

    robot_position = np.array(robot_node.getField("translation").getSFVec3f())
    robot_pose = np.array(
        [robot_position[0], 1. - robot_position[2], robot_node.getField("rotation").getSFRotation()[3] + math.pi / 2])

    return robot_pose
