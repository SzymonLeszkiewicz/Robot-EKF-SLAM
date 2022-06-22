import math
import random
import copy
from controller import Robot, Motor, DistanceSensor, Lidar
import numpy as np
import collections
from time import sleep
import matplotlib.pyplot as plt

LIDAR_SENSOR_MAX_RANGE = 1  # 3 # Meters
LIDAR_SENSOR_MAX_RANGE = 4

LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees, 1.5708 radians
LIDAR_ANGLE_RANGE = 6.2831  # 360 degrees

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
lw_kierunek = 0
rw_kierunek = 0


def perform_least_squares_line_estimate(lidar_world_coords, selected_points):
    """
    @param lidar_world_coords: List of world coordinates read from lidar data (tuples of the form [x, y])
    @param selected_points: Indicies of the points selected for this least squares line estimation
    @return m, b: Slope and intercept for line estimated from data - y = mx + b
    """
    sum_y = sum_yy = sum_x = sum_xx = sum_yx = 0  # Sums of y coordinates, y^2 for each coordinate, x coordinates, x^2 for each coordinate, and y*x for each point

    for point in selected_points:
        world_coord = lidar_world_coords[point]

        sum_y += world_coord[1]
        sum_yy += world_coord[1] ** 2
        sum_x += world_coord[0]
        sum_xx += world_coord[0] ** 2
        sum_yx += world_coord[0] * world_coord[1]

    num_points = len(selected_points)
    b = (sum_y * sum_xx - sum_x * sum_yx) / (num_points * sum_xx - sum_x ** 2)
    m = (num_points * sum_yx - sum_x * sum_y) / (num_points * sum_xx - sum_x ** 2)

    return m, b


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


def run_robot(robot):
    # get the time step of the current world.
    SIM_TIMESTEP = int(robot.getBasicTimeStep())
    max_speed = 6.28
    max_speed = 0
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # get and enable lidar
    lidar = robot.getLidar("LDS-01")
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

    while robot.step(SIM_TIMESTEP) != -1:
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
            lidar_found_loc = convert_lidar_reading_to_world_coord(i, lidar_data[i])
            if lidar_found_loc is not None:
                lidar_readings.append(lidar_found_loc)
        X = [i[0] for i in lidar_readings]
        Y = [i[1] for i in lidar_readings]
        # sleep(0.25)
        plt.scatter(X, Y)
        plt.pause(0.0000001)
        plt.clf()
        left_motor.setVelocity(max_speed)
        right_motor.setVelocity(max_speed)


if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)
    plt.show()
