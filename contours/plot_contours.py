import numpy as np
import cv2


e = np.load('regus_live_cam2_P.npz', allow_pickle=True)
print(e.files)


img = cv2.imread('cam2.jpg')
for fname in e.files:
    print(fname)
    contour = np.int32(e[fname] * 200)
    img = cv2.drawContours(img, [contour], 0, (0,0,255), 2)
cv2.imwrite('cam2contours.jpg', img)
print(e.files)
