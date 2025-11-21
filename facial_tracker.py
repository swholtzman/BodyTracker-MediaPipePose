import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

# focal_length
# pixel_width
# distance
# object_width

class FacialTracker(object):
    def __init__(self):
        pass

# cap = cv2.VideoCapture(0)
# detector = FaceMeshDetector(maxFaces=1)
#
# while True:
#     success, img = cap.read()
#     img, faces = detector.findFaceMesh(img,draw=True) # img,draw=False to remove points
#
#     if faces:
#         face = faces[0]
#         pointLeft = face[145]
#         pointRight = face[374]
#
#         # DRAWING THE FACE MESH
#         cv2.line(img, pointLeft, pointRight, (0, 255, 0), 3)
#
#         cv2.circle(img, pointLeft, 10, (255, 0, 255), cv2.FILLED)
#         cv2.circle(img, pointRight, 10, (255, 0, 255), cv2.FILLED)
#
#         pixel_width, _ = detector.findDistance(pointLeft, pointRight)
#
#         #  FINDING FOCAL WIDTH
#         object_width = 6.3 # current static measurement between eyes
#             # can use a calibration process to determine this per person (hold a measure stick on your head)
#         distance = 50 # also make this dynamic
#         focal_length = (pixel_width * distance) / object_width
#         print(focal_length)
#
#     #     FINDING THE DISTANCE
#         focal_length = 840 # will need to calculate this during calibration to adapt to any camera
#         distance = (object_width * focal_length) / pixel_width
#         print(distance)
#
#         cvzone.putTextRect(img, f'Depth: {int(distance)}cm', (face[10][0]-100, face[10][1]-50), scale = 2, colorT=(0, 255, 0), colorR=None)
#
#     cv2.imshow('Image', img)
#     cv2.waitKey(1)
#
# # use cm as the base measurement units
# # use pixels for camera width measurement unit
#
# # FIND SUBJECT DISTANCE
# distance_to_subject = (subject_width * focal_length)