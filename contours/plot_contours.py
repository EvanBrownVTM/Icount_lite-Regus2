import numpy as np
import cv2


e = np.load('regus_lab_cam0_nt.npz', allow_pickle=True)
print(e.files)


img = cv2.imread('/home/evan/Desktop/Icount_lite/archive/33be5103-9055-4c1e-a07d-7435f593d9de/cam0/31.jpg')
for fname in e.files:
    print(fname)
    contour = np.int32(e[fname] * 416)
    img = cv2.drawContours(img, [contour], 0, (0,0,255), 2)
cv2.imwrite('regus_lab_cam0_nt.jpg', img)
print(e.files)
