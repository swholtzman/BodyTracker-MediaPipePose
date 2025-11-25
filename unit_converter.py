import numpy as np

def polar_to_cartesian(distance, angle):
    """
    Takes the polar coordinates provided by the camera and converts to cartesian coordinates.
    :param distance: distance in meters
    :param angle: angle in degrees
    :return: (x,y) in meters
    """
    #convert angle from degrees to rads
    theta = angle * np.pi / 180

    #Horizontal value, positive is to the right
    x = distance * np.sin(theta)
    #Vertical value, distance away from the camera
    y = distance * np.cos(theta)

    return x, y