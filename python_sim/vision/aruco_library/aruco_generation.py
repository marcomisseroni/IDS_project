import cv2

# generate the 9 aruco marker that identify the three limo
# limo 0: target 0,1,2
# limo 1: target 3,4,5
# limo 2: target 6,7,8
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

for i in range(9):
    marker = cv2.aruco.generateImageMarker(dictionary, i, 200)
    cv2.imwrite(f"marker_{i}.png", marker)