import cv2
import numpy as np

def generate_markers():
    # Use the 4x4 Dictionary (Robust and simple)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Generate Marker ID 0 and ID 1
    # 1000 pixels ensures high quality printing
    marker_0 = cv2.aruco.generateImageMarker(dictionary, 0, 1000)
    marker_1 = cv2.aruco.generateImageMarker(dictionary, 1, 1000)

    cv2.imwrite("marker_0.png", marker_0)
    cv2.imwrite("marker_1.png", marker_1)

    print("Markers saved as marker_0.png and marker_1.png")
    print("OPEN THEM IN WORD/DOCS AND RESIZE TO EXACTLY 10cm x 10cm BEFORE PRINTING.")


if __name__ == "__main__":
    generate_markers()