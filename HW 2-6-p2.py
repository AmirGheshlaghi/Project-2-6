import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/yolov8x")

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 200, 450)

# line detection
lines_list = []
lines = cv2.HoughLines(edges, 1, np.pi/180, 200, min_theta=np.pi/3.9)

# Identify the lines in the image
for r_theta in lines:

	# Determine r and theta for the lines
	arr = np.array(r_theta[0], dtype=np.float64)
	r, theta = arr

	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*r
	y0 = b*r
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# sport ball detection
results = model.predict(source=img)

# Identify the sport ball in the image
for result in results:
	for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):

		# Determine the coordinates of the sport ball in the image
		if obj_cls == 32:
			xx1 = obj_xyxy[0].item()
			yy1 = obj_xyxy[1].item()
			xx2 = obj_xyxy[2].item()
			yy2 = obj_xyxy[3].item()

			cv2.rectangle(img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0, 0, 255), 2)

cv2.imshow("Frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()