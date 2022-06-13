import math
import random
import copy
from controller import Robot, Motor, DistanceSensor, Lidar, Keyboard
import numpy as np
import collections
from time import sleep
import matplotlib.pyplot as plt

state = "sense" # Drive along the course
USE_ODOMETRY = False # False for ground truth pose information, True for real odometry

'''# create the Robot instance.
SLAM_controller_supervisor.init_supervisor()
robot = SLAM_controller_supervisor.supervisor'''

# Map Variables
MAP_BOUNDS = [1.,1.]
CELL_RESOLUTIONS = np.array([0.1, 0.1]) # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS,NUM_X_CELLS])

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
GROUND_SENSOR_THRESHOLD = 600 # Light intensity units
LIDAR_SENSOR_MAX_RANGE = 5.#3 # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians

#RANSAC values
MAX_TRIALS = 1000 # Max times to run algorithm
MAX_SAMPLE = 10 # Randomly select X points
MIN_LINE_POINTS = 6 # If less than 5 points left, stop algorithm
RANSAC_TOLERANCE = 0.15 # If point is within 20 cm of line, it is part of the line
RANSAC_CONSENSUS = 6 # At least 5 points required to determine if a line

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

# Constants to help with the Odometry update
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1

# GAIN Values
theta_gain = 1.0
distance_gain = 0.3

MAX_VEL_REDUCTION = 0.25
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION # m/s
EPUCK_AXLE_DIAMETER = 0.053
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2
WHEEL_FORWARD = 1
WHEEL_STOPPED = 0
WHEEL_BACKWARD = -1


def populate_map(m):
    obs_list = SLAM_controller_supervisor.supervisor_get_obstacle_positions()
    obs_size = 0.06  # 6cm boxes
    for obs in obs_list:
        obs_coords_lower = obs - obs_size / 2.
        obs_coords_upper = obs + obs_size / 2.
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1
        obs_coords_lower = [obs[0] - obs_size / 2, obs[1] + obs_size / 2.]
        obs_coords_upper = [obs[0] + obs_size / 2., obs[1] - obs_size / 2.]
        obs_coords = np.linspace(obs_coords_lower, obs_coords_upper, 10)
        for coord in obs_coords:
            m[transform_world_coord_to_map_coord(coord)] = 1


def get_bounded_theta(theta):
    '''
    Returns theta bounded in [-PI, PI]
    '''
    while theta > math.pi: theta -= 2. * math.pi
    while theta < -math.pi: theta += 2. * math.pi
    return theta


def convert_lidar_reading_to_world_coord(lidar_bin, lidar_distance):
    """
    @param lidar_bin: The beam index that provided this measurement
    @param lidar_distance: The distance measurement from the sensor for that beam
    @return world_point: List containing the corresponding (x,y) point in the world frame of reference
    """

    # YOUR CODE HERE
    # print("Dist:", lidar_distance, "Ang:", lidar_bin)

    # No detection
    if (lidar_distance > LIDAR_SENSOR_MAX_RANGE):  # or lidar_distance > math.sqrt(2)):
        return None

    # Lidar centered at robot 0,0 so no translation needed
    # Convert lidar -> robot adding math.pi/2 to fix direction
    bQ_x = math.sin(lidar_bin + math.pi / 2) * lidar_distance
    bQ_y = math.cos(lidar_bin + math.pi / 2) * lidar_distance
    # print(bQ_x, bQ_y)
    # convert robot -> world
    x = math.cos(pose_theta) * bQ_x - math.sin(pose_theta) * bQ_y + pose_x
    y = math.sin(pose_theta) * bQ_x + math.cos(pose_theta) * bQ_y + pose_y
    # print(x,y)
    return [x, y]


# From Lab 5
def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


# From Lab 5
def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return np.array([(col + 0.5) * CELL_RESOLUTIONS[1], (row + 0.5) * CELL_RESOLUTIONS[0]])



def run_robot(robot):
    # get the time step of the current world.
    SIM_TIMESTEP = int(robot.getBasicTimeStep())
    max_speed = 6.28
    max_speed = 2
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # get and enable lidar
    lidar = robot.getDevice("LDS-01")
    lidar.enable(SIM_TIMESTEP)
    # lidar.enablePointCloud()

    # Initialize lidar motors
    lidar_main_motor = robot.getDevice('LDS-01_main_motor')
    lidar_secondary_motor = robot.getDevice('LDS-01_secondary_motor')
    lidar_main_motor.setPosition(float('inf'))
    lidar_secondary_motor.setPosition(float('inf'))
    lidar_main_motor.setVelocity(30.0)
    lidar_secondary_motor.setVelocity(60.0)

    lidar_data = np.zeros(LIDAR_ANGLE_BINS)
    print(lidar_data)

    step = LIDAR_ANGLE_RANGE / LIDAR_ANGLE_BINS

    lidar_offsets = np.linspace(step * (LIDAR_ANGLE_BINS // 2), -step * (LIDAR_ANGLE_BINS // 2), LIDAR_ANGLE_BINS)
    # print(lidar_offsets)

    # lidar_data = lidar.getRangeImage()
    # print('getRangeImage', lidar_data)
    # y = lidar_data
    # x = np.linspace(math.pi, 0, np.size(y))
    # plt.polar(x, y)
    # plt.pause(0.0000001)
    # plt.clf()

    # convert lidar to world locations

    # klawiatura
    keyboard = Keyboard()
    keyboard.enable(SIM_TIMESTEP)

    motor_cmd = {
        ord("W"): (max_speed, max_speed),
        ord("S"): (-max_speed, -max_speed),
        ord("A"): (-max_speed, max_speed),
        ord("D"): (max_speed, -max_speed),
    }

    def steruj(cmd):
        left_motor.setVelocity(cmd[0])
        right_motor.setVelocity(cmd[1])

    while robot.step(SIM_TIMESTEP) != -1:

        # sterowanie klawiatura
        key = keyboard.getKey()
        if key in motor_cmd.keys():
            steruj(motor_cmd[key])
        else:
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)



        lidar_data = lidar.getRangeImage()
        # print("FOV", lidar.getFov())
        # print("FOV2", lidar.getVerticalFov())
        # print('min', lidar.getMinRange())
        # print('max', lidar.getMaxRange())
        # print('getRangeImage', lidar_data)
        # y = lidar_data
        # x = np.linspace(math.pi * 1.5666666, 0, np.size(y))
        # plt.polar(x, y)
        # plt.pause(0.0000001)
        # plt.clf()

        lidar_readings = []
        for i in range(21):
            lidar_found_loc =convert_lidar_reading_to_world_coord(i, lidar_data[i])
            if lidar_found_loc is not None:
                lidar_readings.append(lidar_found_loc)
        print(lidar_readings)
        # X = [i[0] for i in lidar_readings]
        # Y = [i[1] for i in lidar_readings]
        # # sleep(0.25)
        # plt.scatter(X, Y)
        # plt.pause(0.0000001)
        # plt.clf()


if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)
    plt.show()
