##### NETWORK #####
IP_ADDRESS_LOCAL="192.168.1.155"
IP_ADDRESS_PI="192.168.1.65"
IP_ADDRESS_NANO="192.168.1.105"
FaceRec = True

##### Device specific #####
user = 'evan'
log_path = "/home/{}/Desktop/".format(user)
base_path = "/home/{}/Desktop/Icount_lite/".format(user)

##### Location specific #####
camera_map = {"cam0": 23881566, "cam1": 23881567, "cam2": 23875565} # Format{"cam0": 23***, "cam1": 23***, "cam2":23***}
activate_arch = True
sms_alert = False
thresh_cv_time = 70
machine_location = 'Regus Liberty Station'
cls_dict = {0: 'Coca Cola 20z', 1: 'Gatorade 12oz', 2: 'Monster Regular', 3: 'Diet Coke (SF)', 4: 'Dasani Water 16 oz'} 

cam0_zone = 'contours/regus_lab_cam0_nt.npz'#'contours/cam0_Regus_office.npz' #
cam1_zone = 'contours/regus_lab_cam1_nt.npz'#'contours/cam1_Regus_office.npz' #
cam2_zone = 'contours/regus_live_cam2.npz'#'contours/cam2_Regus_office.npz' #

##### Software setting #####
archive_flag = True
maxCamerasToUse = 3
archive_size = 416
save_size = 200
display_mode = True
pika_flag = True
icount_mode = False
show_contours=True
model_name = "yolov4x-mish-416"