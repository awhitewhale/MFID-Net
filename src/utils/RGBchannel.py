import cv2
import sys
import numpy as np

image = cv2.imread('8289_0.95_0.1.jpg', cv2.IMREAD_COLOR)
b,g,r=cv2.split(image)

cv2.imwrite("b.png",b)
cv2.imwrite("g.png",g)
cv2.imwrite("r.png",r)