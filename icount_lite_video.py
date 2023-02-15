############################################
'''
Date: Nov 29, 2022
author: CV Team

Icount realtime application
	- Run detection across 3 cameras
	- Postprocessing upload of video
	- Capture and write operate synchronously

TO-DO
	- Activate/Deactivate cameras
	- Clean state
'''
###########################################

import os
import sys
import time
import logging
import pika
import copy
import numpy as np
import cv2
import json
import pycuda.autoinit  # This is needed for initializing CUDA driver
import configSrc as cfg
import tensorflow as tf
from PIL import Image
import requests
import traceback
import argparse

from pypylon import pylon
from collections import deque, Counter, defaultdict
from utils.yolo_with_plugins import TrtYOLO
from utils.display import open_window, set_display, show_fps
from utils.visualization_ic import BBoxVisualization
from utils_lite.tracker import AVT
from utils_lite.front_cam_solver import FrontCam
from utils_lite.side_cam_solver import SideCam
from  utils_lite.utils import descale_contour
from datetime import datetime
from scipy.optimize import linear_sum_assignment

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(filename='{}logs/Icount.log'.format(cfg.log_path), level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
logging.disable(logging.DEBUG)
logger=logging.getLogger()
logger.info("")
sys.stderr.write=logger.error

#Setting
maxCamerasToUse = cfg.maxCamerasToUse
archive_size = cfg.archive_size
save_size = cfg.save_size
display_mode = cfg.display_mode
pika_flag = cfg.pika_flag
cls_dict = cfg.cls_dict

#Initialization
tsv_url = 'http://192.168.1.140:8085/tsv/flashapi'
timestamp_format = "%Y%m%d-%H_%M_%S"
fps = 0.0
conf_th = 0.7

def getFrames(camera_dirs):
	frames0 = iter([Image.open(os.path.join(camera_dirs[0],f)) for f in sorted(os.listdir(camera_dirs[0]))])
	frames1 = iter([Image.open(os.path.join(camera_dirs[1],f)) for f in sorted(os.listdir(camera_dirs[1]))])
	frames2 = iter([Image.open(os.path.join(camera_dirs[2],f)) for f in sorted(os.listdir(camera_dirs[2]))])
	return frames0, frames1, frames2

def init():
	logger.info('Loading TensoRT model...')
	# build the class (index/name) dictionary from labelmap file
	trt_yolo = TrtYOLO("yolov4x-mish-416", (416, 416), 4, False, path_folder = 'yolo/')

	#print('\tRunning warmup detection')
	dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
	_, _, _ = trt_yolo.detect(dummy_img, 0.6)
	logger.info('Model loaded and ready for detection')

	return trt_yolo

def sms_text(tsv_url, post_time):
	sms_response = requests.post(url= tsv_url, data='["CreateSMSText", "CV FRAUD ALERT: ({}): Transaction time threshold exceeded / {}sec {}"]'.format(cfg.machine_location, post_time, datetime.now().strftime("%c"))).json()
	if sms_response['resultCode'] == "SUCCESS":
		logger.info("   CV sms alert succesfully sent")
	else:
		logger.info("   CV sms alert: Failed")

#RabbitMQ Initialization
def initializeChannel():
	#Initialize queue for door signal
	credentials = pika.PlainCredentials('guest','guest')
	parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials, blocked_connection_timeout=3000)
	connection = pika.BlockingConnection(parameters)
	channel = connection.channel()
	channel.queue_declare(queue='cvIcount',durable = True)
	channel2 = connection.channel()
	channel2.queue_declare(queue='cvPost',durable = True)

	#Clear queue for pre-existing messages
	channel.queue_purge(queue='cvIcount')
	channel2.queue_purge(queue='cvPost')

	logger.info("Rabbitmq connections initialized ")
	return channel, channel2, connection


def trt_detect(frame, trt_yolo, conf_th, vis):
	if frame is not None:
		boxes, confs, clss = trt_yolo.detect(frame, conf_th)
		if display_mode:
			frame = vis.draw_bboxes(frame, boxes, confs, clss)
		return frame, clss, boxes, confs


def update_logic(new_boxes, clss, frame, cam_solver, avt, frame_id,frame_draw):
	cents = []
	cent2bbox = {}
	cent2cls = {}
	id2active_zone = {}
	for i in range(len(new_boxes)):
		bbox = new_boxes[i]
		cls = clss[i]
		cents.append([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])
		cent2bbox["{}_{}".format((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)] = bbox
		cent2cls["{}_{}".format((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)] = cls

	objects, disappeared = avt.update(cents)

	for (objectID, centroid) in objects.items():
		cent_symbol = "{}_{}".format(centroid[0], centroid[1])
		if cent_symbol not in cent2bbox:
			continue
		cam_solver.update_tracks(cent2cls[cent_symbol], cent2bbox[cent_symbol], objectID, frame_id, frame)
		id2active_zone[objectID] = cam_solver._tracks[objectID]._active_zone

		if display_mode:
			text = "ID {}, {}".format(str(objectID), cam_solver._tracks[objectID]._active_zone)
			cv2.putText(frame_draw, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

	return id2active_zone

def solver_infer(cam_solver, logger, cv_activities, cv_pick_cam, cv_ret_cam, idle_flag = False):
	cam_solver.interact_tracks(logger, cv_activities, cv_pick_cam, cv_ret_cam, idle_flag)

def merge_cart(cam0_solver, cam1_solver, cam2_solver):
	cart = defaultdict(int)

	#Running infer only on cam2 - start
	#for cl in cam0_solver.cart:
	#	cart[cl] += cam0_solver.cart[cl]
	#for cl in cam1_solver.cart:
	#	cart[cl] += cam1_solver.cart[cl]
	# - end
	for cl in cam2_solver.cart:
		cart[cl] += cam2_solver.cart[cl]

	return cart

def displayCart(det_frame, cart):
	#cv2.putText(det_frame, 'Cart:', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cnt = 0
	for prod_ind in sorted(cart):
		if cart[prod_ind] != 0:
			cv2.putText(det_frame, "{}:{}".format(cls_dict[prod_ind], cart[prod_ind]), (0, 50  + 30 * cnt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			cnt += 1
def infer_engine(timestr, frame0, frame1, frame2, frame_cnt0, frame_cnt1, frame_cnt2, cv_activities_cam0, cv_activities_cam1, cv_activities_cam2, cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2):
	frame0_copy = frame0.copy()
	frame1_copy = frame1.copy()
	frame2_copy = frame2.copy()

	det_frame0, clss0, new_boxes0, confs0 = trt_detect(frame0, trt_yolo, conf_th, vis)
	det_frame1, clss1, new_boxes1, confs1 = trt_detect(frame1, trt_yolo, conf_th, vis)
	det_frame2, clss2, new_boxes2, confs2 = trt_detect(frame2, trt_yolo, conf_th, vis)

	file2info = {}
	file2info['bboxes'] = np.asarray(np.asarray(new_boxes0, dtype=np.int32) / archive_size * save_size, dtype = np.int32).tolist()
	file2info['classes'] = np.asarray(clss0, dtype = np.int32).tolist()
	file2info['scores'] = np.asarray(confs0).tolist()
	if not os.path.exists("{}archive/{}/cam0/prod".format(cfg.base_path, transid)):
		os.makedirs("{}archive/{}/cam0/prod".format(cfg.base_path, transid))

	file2info1 = {}
	file2info1['bboxes'] = np.asarray(np.asarray(new_boxes1, dtype=np.int32) / archive_size * save_size, dtype = np.int32).tolist()
	file2info1['classes'] = np.asarray(clss1, dtype = np.int32).tolist()
	file2info1['scores'] = np.asarray(confs1).tolist()
	if not os.path.exists("{}archive/{}/cam1/prod".format(cfg.base_path, transid)):
		os.makedirs("{}archive/{}/cam1/prod".format(cfg.base_path, transid))

	file2info2 = {}
	file2info2['bboxes'] = np.asarray(np.asarray(new_boxes2, dtype=np.int32) / archive_size * save_size, dtype = np.int32).tolist()
	file2info2['classes'] = np.asarray(clss2, dtype = np.int32).tolist()
	file2info2['scores'] = np.asarray(confs2).tolist()
	if not os.path.exists("{}archive/{}/cam2/prod".format(cfg.base_path, transid)):
		os.makedirs("{}archive/{}/cam2/prod".format(cfg.base_path, transid))
	f_name = "%s_%05d"%(timestr, int(frame_cnt0))
	f_name1 = "%s_%05d"%(timestr, int(frame_cnt1))
	f_name2 = "%s_%05d"%(timestr, int(frame_cnt2))
	json.dump(file2info, open('{}archive/{}/cam0/prod/{}.json'.format(cfg.base_path, transid, f_name), 'w'))
	json.dump(file2info1, open('{}archive/{}/cam1/prod/{}.json'.format(cfg.base_path, transid, f_name1), 'w'))
	json.dump(file2info2, open('{}archive/{}/cam2/prod/{}.json'.format(cfg.base_path, transid, f_name2), 'w'))

	id2active_zone0 = update_logic(new_boxes0, clss0, frame0_copy, cam0_solver, avt0, frame_cnt0 - 1, frame0)
	id2active_zone1 = update_logic(new_boxes1, clss1, frame1_copy, cam1_solver, avt1, frame_cnt1 - 1, frame1)
	id2active_zone2 = update_logic(new_boxes2, clss2, frame2_copy, cam2_solver, avt2, frame_cnt2 - 1, frame2)

	solver_infer(cam0_solver, logger, cv_activities_cam0, cv_pick_cam0, cv_ret_cam0)
	solver_infer(cam1_solver, logger, cv_activities_cam1, cv_pick_cam1, cv_ret_cam1)
	solver_infer(cam2_solver, logger, cv_activities_cam2, cv_pick_cam2, cv_ret_cam2)

	cart = merge_cart(cam0_solver, cam1_solver, cam2_solver)

	return det_frame0, det_frame1, det_frame2, cart


def distance(item1, item2):
    if item1[0] != item2[0]:
        return 10
    time1 = datetime.strptime(item1[2], "%Y-%m-%d:%H:%M:%S")
    time2 = datetime.strptime(item2[2], "%Y-%m-%d:%H:%M:%S")
    return abs((time1 - time2).total_seconds())
def match(set1, set2, thresh=None):
    """
        e.g.
            set1: [{"class_id":1, "action":"PICK", "timestamp": '2022-07-25:11:15:50'}, {"class_id":1, "action":"RETURN", "timestamp": '2022-07-25:11:15:57'}]
            set1: [{"class_id":1, "action":"RETURN", "timestamp": '2022-07-25:11:16:00'}]
    """
    not_matched1 = []
    not_matched2 = []

    n = len(set1)
    m = len(set2)

    cost_matrix = np.zeros((n, m))

    for i, item1 in enumerate(set1):
        for j, item2 in enumerate(set2):
            score = distance(item1, item2)
            cost_matrix[i, j] = score

    rows, cols = linear_sum_assignment(cost_matrix)
    not_matched1 += set(rows).symmetric_difference(range(n))
    not_matched2 += set(cols).symmetric_difference(range(m))
    matches = []
    for row, col in zip(rows, cols):
        if (thresh and cost_matrix[row, col].sum() < thresh) or not thresh:
            matches.append( (row, col) )
        else:
            not_matched1.append(row)
            not_matched2.append(col)

    return matches, not_matched1, not_matched2

def process_actions(cv_act_cam0, cv_act_cam2, timeout = 3, m_flag = False):
	m0_act, m1_act, m2_act = match(cv_act_cam0, cv_act_cam2, timeout)
	cam0_act_index = []
	cam2_act_index = []
	ret_acts = []
	for x,y in m0_act:
		ret_acts.append(cv_act_cam0[x])
		cam0_act_index.append(x)
		cam2_act_index.append(y)
	if not m_flag:
		for ind in m1_act:
			time1 = datetime.strptime(cv_act_cam0[ind][-1], "%Y-%m-%d:%H:%M:%S")
			curr_time = datetime.strptime(time.strftime(timestamp_format), timestamp_format)
			if abs((time1 - curr_time).total_seconds()) > timeout:
				cam0_act_index.append(ind)
				ret_acts.append(cv_act_cam0[ind])
	if not m_flag:
		for ind in m2_act:
			time2 = datetime.strptime(cv_act_cam2[ind][-1], "%Y-%m-%d:%H:%M:%S")
			curr_time = datetime.strptime(time.strftime(timestamp_format), timestamp_format)
			if abs((time2 - curr_time).total_seconds()) > timeout:
				cam2_act_index.append(ind)
				ret_acts.append(cv_act_cam2[ind])

	cv_act_cam0 = [i for j, i in enumerate(cv_act_cam0) if j not in cam0_act_index]
	cv_act_cam2 = [i for j, i in enumerate(cv_act_cam2) if j not in cam2_act_index]
	return ret_acts, cv_act_cam0, cv_act_cam2

def fuse_cam01_02_activities(cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2, matched_pick_cam01, matched_return_cam01, matched_pick_cam02, matched_return_cam02):
	cv_pick_cam0_copy = copy.deepcopy(cv_pick_cam0)
	cv_ret_cam0_copy = copy.deepcopy(cv_ret_cam0)
	act_picks01, cv_pick_cam0, cv_pick_cam1 = process_actions(cv_pick_cam0, cv_pick_cam1, timeout = 3)
	act_returns01, cv_ret_cam0, cv_ret_cam1 = process_actions(cv_ret_cam0, cv_ret_cam1, timeout = 3)

	act_picks02, cv_pick_cam0_copy, cv_pick_cam2 = process_actions(cv_pick_cam0_copy, cv_pick_cam2, timeout = 3)
	act_returns02, cv_ret_cam0_copy, cv_ret_cam2 = process_actions(cv_ret_cam0_copy, cv_ret_cam2, timeout = 3)

	cv_pick_cam0_fuse = []
	for item in cv_pick_cam0:
		if item in cv_pick_cam0_copy:
			cv_pick_cam0_fuse.append(item)
	cv_ret_cam0_fuse = []
	for item in cv_ret_cam0:
		if item in cv_ret_cam0_copy:
			cv_ret_cam0_fuse.append(item)

	cv_pick_cam0 = cv_pick_cam0_fuse
	cv_ret_cam0 = cv_ret_cam0_fuse

	if len(act_picks01) > 0:
		for act_pick in act_picks01:
			matched_pick_cam01.append(act_pick)
	if len(act_returns01) > 0:
		for act_return in act_returns01:
			matched_return_cam01.append(act_return)
	if len(act_picks02) > 0:
		for act_pick in act_picks02:
			matched_pick_cam02.append(act_pick)
	if len(act_returns02) > 0:
		for act_return in act_returns02:
			matched_return_cam02.append(act_return)
	return cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2

def fuse_cam12_activities(cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2, cv_activities_fused):
	act_picks12, cv_pick_cam1, cv_pick_cam2 = process_actions(cv_pick_cam1, cv_pick_cam2, timeout = 3, m_flag = True)
	act_returns12, cv_ret_cam1, cv_ret_cam2 = process_actions(cv_ret_cam1, cv_ret_cam2, timeout = 3, m_flag = True)
	if len(act_picks12) > 0:
		for act_pick in act_picks12:
			cv_activities_fused.append({'class_id': act_pick[0], 'action': act_pick[1], 'timestamp': act_pick[2]})
			logger.info("   fused action: {} {} @ {}".format(act_pick[1], act_pick[0], act_pick[2]))

	if len(act_returns12) > 0:
		for act_return in act_returns12:
			cv_activities_fused.append({'class_id': act_return[0], 'action': act_return[1], 'timestamp': act_return[2]})
			logger.info("   fused action: {} {} @ {}".format(act_return[1], act_return[0], act_return[2]))

	return cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2

def fuse_all_cams_activities(matched_pick_cam01, matched_pick_cam02, matched_return_cam01, matched_return_cam02, cv_activities_fused):
	matched_act_picks012, matched_pick_cam01, matched_pick_cam02 = process_actions(matched_pick_cam01, matched_pick_cam02, timeout = 3)
	matched_act_returns012, matched_return_cam01, matched_return_cam02 = process_actions(matched_return_cam01, matched_return_cam02, timeout = 3)

	if len(matched_act_picks012) > 0:
		for act_pick in matched_act_picks012:
			cv_activities_fused.append({'class_id': act_pick[0], 'action': act_pick[1], 'timestamp': act_pick[2]})
			logger.info("   fused action: {} {} @ {}".format(act_pick[1], act_pick[0], act_pick[2]))
	if len(matched_act_returns012) > 0:
		for act_return in matched_act_returns012:
			cv_activities_fused.append({'class_id': act_return[0], 'action': act_return[1], 'timestamp': act_return[2]})
			logger.info("   fused action: {} {} @ {}".format(act_return[1], act_return[0], act_return[2]))

	return matched_pick_cam01, matched_pick_cam02, matched_return_cam01, matched_return_cam02

#convert raw image to bytes
def _bytes_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#compress image bytes
def img2jpeg(image):
	is_success, im_buf_arr = cv2.imencode(".jpg", image)
	byte_im = im_buf_arr.tobytes()
	return byte_im

_, channel2, connection = initializeChannel()

#initialize solvers
avt0 = AVT()
avt1 = AVT()
avt2 = AVT()

trt_yolo = init()
vis = BBoxVisualization(cls_dict)

cam0_solver = FrontCam('cam0', cfg.cam0_zone)
cam1_solver = SideCam('cam1', cfg.cam1_zone)
cam2_solver = SideCam('cam2', cfg.cam2_zone)

#intialize variables
tic = time.time()

cameraContextValue = 0
frame_cnt0 = 0
frame_cnt1 = 0
frame_cnt2 = 0

transid = 'trans_init'
cv_activities = []
check_list = [ False for i in range(maxCamerasToUse)]

def main(transid):
	#load frames
	camera_dirs = [os.path.join(base_path, 'archive', x) for x in ['cam0', 'cam1', 'cam2']]
	frames0, frames1, frames2 = getFrames(camera_dirs)

	#initialize variables
	door_state = "DoorOpened"
	duration_time = 0
	frame_cnt0 = 0
	frame_cnt1 = 0
	frame_cnt2 = 0
	cv_activities_cam0 = []
	cv_activities_cam1 = []
	cv_activities_cam2 = []
	cv_pick_cam0 = []
	cv_ret_cam0 = []
	cv_pick_cam1 = []
	cv_ret_cam1 = []
	cv_pick_cam2 = []
	cv_ret_cam2 = []
	matched_pick_cam01 = []
	matched_return_cam01 = []
	matched_pick_cam02 = []
	matched_return_cam02 = []

	cv_activities = []
	ls_activities = []
					
	clear_flag = 1

	#************Run model on video************
	logger.info('Running detections on stored video')
	while True:
		try:
			if cameraContextValue == 0:
				frame = next(frames0)
				frame_cnt0 += 1
			elif cameraContextValue == 1:
				frame = next(frames1)
				frame_cnt1 += 1
			else:
				frame = next(frames2)
				frame_cnt2 += 1
		except StopIteration:
			door_state = 'DoorLocked'
			logger.info('Reached end of video')	
			break

		if cameraContextValue == 0:
			frame_cnt0 += 1
			frame0 = cv2.resize(np.uint8(frame), (archive_size, archive_size))
			check_list[0] = True
		elif cameraContextValue == 1:
			frame_cnt1 += 1
			frame1 = cv2.resize(np.uint8(frame), (archive_size, archive_size))
			frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
			check_list[1] = True
		else:
			frame_cnt2 += 1
			frame2 = cv2.resize(np.uint8(frame), (archive_size, archive_size))
			frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
			check_list[2] = True
						
		if cameraContextValue == 0:
			cameraContextValue = 1
		elif cameraContextValue == 1:
			cameraContextValue = 2
		else:
			cameraContextValue = 0	

		if all(check_list):

			timestr = time.strftime(timestamp_format)
			check_list = np.logical_not(check_list)

			if icount_mode:
				det_frame0, det_frame1, det_frame2, cart = infer_engine(timestr, frame0, frame1, frame2, frame_cnt0, frame_cnt1, frame_cnt2, cv_activities_cam0, cv_activities_cam1, cv_activities_cam2, cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2)
				#Performing simple inference / cam2 only
				cv_activities = cv_activities_cam2 + cv_activities_cam0 + cv_activities_cam1

				#cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2 = fuse_cam01_02_activities(cv_pick_cam0, cv_ret_cam0, cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2, \
				#																									matched_pick_cam01, matched_return_cam01, matched_pick_cam02, matched_return_cam02)
				#cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2 = fuse_cam12_activities(cv_pick_cam1, cv_ret_cam1, cv_pick_cam2, cv_ret_cam2, cv_activities)matched_return_cam01,
				#matched_pick_cam01, matched_pick_cam02, matched_return_cam01, matched_return_cam02 = fuse_all_cams_activities(matched_pick_cam01, matched_pick_cam02, matched_return_cam01, matched_return_cam02, cv_activities)

			if display_mode:
				img_hstack = det_frame0
				img_hstack = np.hstack((img_hstack, det_frame1))
				img_hstack = np.hstack((img_hstack, det_frame2))
				img_hstack = show_fps(img_hstack, fps)
				displayCart(img_hstack, cart)

				cv2.imshow('Yo', img_hstack[:,:,::-1])

				key = cv2.waitKey(1)
				if key == 27:  # ESC key: quit program
					break

			toc = time.time()
			curr_fps = 1.0 / (toc- tic)
			fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
			tic = toc
			if frame_cnt0 % 20 == 0:
				print(fps)
		
		#************Upload detections***********
		logger.info
		data = {"cmd": "Done", "transid": transid, "timestamp": time.strftime("%Y%m%d-%H_%M_%S"), "cv_activities": cv_activities, "ls_activities": ls_activities}
		mess = json.dumps(data)
		channel2.basic_publish(exchange='',
					routing_key="cvPost",
					body=mess)

		logger.info('CV_activities:')
		logger.info(cv_activities)
		logger.info('LS_activities:')
		logger.info(ls_activities)
		if (len(cv_activities) > 0) or (len(ls_activities) > 0): #only send signal to postprocess if we have either a cv_activity or a ls_activity
			if len(cv_activities) > 0:
				cv_activities = sorted(cv_activities, key=lambda d: d['timestamp']) 
			data = {"cmd": "Done", "transid": transid, "timestamp": time.strftime("%Y%m%d-%H_%M_%S"), "cv_activities": cv_activities, "ls_activities": ls_activities}
			mess = json.dumps(data)
			channel2.basic_publish(exchange='',
							routing_key="cvPost",
							body=mess)
			if icount_mode:
				logger.info("Sent cvPost signal (Icount mode)\n")
			else:
				logger.info("Sent cvPost signal (Recording-only mode)\n")
		else:
			logger.info("No cvPost signal sent - no CV or LS activities")
		door_state = 'initialize'
		ls_activities = ""

if __name__ == '__main__':
	#get user input transid
	parser = argparse.ArgumentParser()
	parser.add_argument('--transid')
	args = parser.parse_known_args()
	print(args)
	transid = args.transid
	print(transid)

	main(transid)
