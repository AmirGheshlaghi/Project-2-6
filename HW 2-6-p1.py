import cv2
import numpy as np

img = cv2.imread("image 2.webp")
img = cv2.resize(img, (675, 500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 350, 700)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2.5, 50, maxRadius=35, minRadius=23)

i = 0
if circles is not None:
	circles = circles[0].astype(np.uint32)

	for circle in circles:
		cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
		i += 1

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()