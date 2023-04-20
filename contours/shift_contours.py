import numpy as np
import cv2
import os

e = np.load('cam2_Regus_hallway_02222023.npz', allow_pickle=True)
output_contour_path = 'cam2_Regus_hallway_02222023.npz'
print(e.files)

shelf_dict = {}
# img = cv2.imread('/home/evan/Desktop/Icount_lite/archive/a6727b46-6757-4234-bc33-2c5429e03c48/cam1/31.jpg')

#adjust contours
for shelf in e.files:
    contour = e[shelf] * 640/416
    shelf_dict[shelf] = contour
    # img = cv2.drawContours(img, [np.int32(contour*416)], 0, (0,0,255), 2)


# #display contours
# cv2.imshow(os.path.normpath(os.path.basename(output_contour_path)), img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#save contours
print(shelf_dict)
np.savez(output_contour_path, **shelf_dict)