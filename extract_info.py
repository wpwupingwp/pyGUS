from sys import argv
import cv2

i = argv[1]
o = argv[1] + '-extracted.png'
img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
alpha = img[:, :, 3]
cv2.imwrite(o, alpha)
