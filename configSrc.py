##### NETWORK #####
IP_ADDRESS_LOCAL="192.168.1.155"
IP_ADDRESS_PI="192.168.1.65"
IP_ADDRESS_NANO="192.168.1.105"
FaceRec = True

##### Device specific #####
user = 'cvnx'
log_path = "/home/{}/Desktop/".format(user)
base_path = "/home/{}/Desktop/Icount_lite/".format(user)

##### Location specific #####
camera_map = {"cam0": 23881566, "cam1": 23881567, "cam2": 23875565} # Format{"cam0": 23***, "cam1": 23***, "cam2":23***}
activate_arch = True
sms_alert = False
thresh_cv_time = 70
machine_location = 'Regus Liberty Station'
cls_dict = {0: 'Miss Vickies', 1: 'Coca Cola 20z', 2: 'Gatorade 12oz', 3: 'sun_chips', 4: 'Dorito Nacho', 5: 'lays', 6: 'Monster Regular', 7: 'gold_peak', 8: 'Diet Coke (SF)', 9: 'sprite', 10: 'Dasani Water 16 oz'} #11 prod

cam0_zone = 'utils_lite/regus_lab_cam0_nt.npz'
cam1_zone = 'utils_lite/regus_lab_cam1_nt.npz'
cam2_zone = 'utils_lite/regus_live_cam2.npz'

##### Software setting #####
archive_flag = True
maxCamerasToUse = 3
archive_size = 416
save_size = 200
display_mode = False
pika_flag = True
