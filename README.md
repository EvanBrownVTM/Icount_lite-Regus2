# Icount_lite
SETUP
1) cd plugins && sudo make clean && make
2) mkdir archive
3) download yolov4-416-tiny.trt in yolo folder
4) create configSrc.py from template below and adjust IPs and camera serial #'s 

configSrc.py example:

IP_ADDRESS_LOCAL="192.168.1.155"
IP_ADDRESS_PI="192.168.1.65"
IP_ADDRESS_NANO="192.168.1.138"
#IP_ADDRESS_PI="localhost"
base_path = "/home/cvnx/Desktop/Icount_lite/"
log_path = "/home/cvnx/Desktop/"
camera_map = {"cam0": 23881566, "cam1":23881567, "cam2":23875565}
#camera_map = {"cam0": 23601413, "cam1":23797017, "cam2":23601565, 'cam0_1':23881564, 'cam0_2':23881564}
activate_arch = True
cls_dict = {0: 'vickies', 1: 'coke', 2: 'gatorade', 3: 'sun_chips', 4: 'doritos', 5: 'lays', 6: 'monster', 7: 'gold_peak', 8: 'diet_coke', 9: 'sprite'}
#{0: 'Vickies', 1: 'Coke', 2: 'Gatorade', 3: 'Sun_chips', 4: 'Doritos', 5: 'Lays', 6: 'Monster', 7: 'Gold_peak', 8: 'Diet_coke', 9: 'Sprite'}














