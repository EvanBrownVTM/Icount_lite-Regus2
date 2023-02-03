<<<<<<< HEAD
##################################################
'''
Date: Nov 29, 2022
author: CV teamm

To run asynchronously run batch upload

Observe output in logs

'''
##################################################

=======
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
import os
import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("/usr/local/cuda-10.2/bin")
sys.path.append("/usr/local/cuda-10.2/lib64")
import time
import logging
import pika
import copy
import numpy as np
import shutil
import cv2
import json
import random
import traceback
import pycuda.autoinit  # This is needed for initializing CUDA driver
import multiprocessing
<<<<<<< HEAD
import utils_lite.configSrc as cfg
=======
import configSrc as cfg
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
import tensorflow as tf
import requests
import moviepy.video.io.ImageSequenceClip
from datetime import datetime
<<<<<<< HEAD
=======
from scipy.optimize import linear_sum_assignment
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad


logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
<<<<<<< HEAD
logging.basicConfig(filename='{}logs/Post.log'.format(cfg.log_path), level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
=======
logging.basicConfig(filename='/home/cvnx/Desktop/logs/Post.log', level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
logging.disable(logging.DEBUG)
logger=logging.getLogger()
logger.info("")
sys.stderr.write=logger.error

vicki_app = "http://192.168.1.140:8085/tsv/flashapi"
archive_size = 200
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
<<<<<<< HEAD

=======
cls_dict = {0:'vickies', 1:'coke', 2:'gatorade', 3:'sun_chips', 4:'doritos', 5:'lays', 6:'monster', 7:'gold_peak', 8:'diet_coke', 9:'sprite'}

prod_json = requests.post(url = vicki_app, data = "['fetchProduct']").json()
name2id = {}
for prod in prod_json:
	name2id[prod['productName']] = prod['description']

#print(name2id)

#sys.exit(0)
#name2id = {"Peanut MMs":"b9d644d0-40d4-4d65-82a0-c94d474f13b3", "Sour Patch Kids": "1fef14b0-f36e-4ed1-9686-90e099183829"}
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
def get_custom_machine_setting(custom_machine_setting):
	while True:
		try:
			ret = requests.post(url=vicki_app, data="[\"FetchCustomMachineSetting\", \"{}\"]".format(custom_machine_setting)).json()["value"]
<<<<<<< HEAD
			return ret
		except Exception as e:
			continue

=======
			return ret   
		except Exception as e:
			continue
			
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
def login_as_machine(url, id, token, api_key):
        try:
                headers = {"Content-Type": "application/json",
                                   "grant_type": "client_credentials",
                                   "apikey": api_key,
                                   "machine_token": token}
                response = requests.post("{}/loyalty/machines/{}/login".format(url, id), headers=headers)
                if response.status_code is 200:
                        logger.info("Login successuful")
                        return response.json()['access_token']
                else:
                        logger.info("Login fail")
                        logger.info(response)
                        return -1
        except Exception as e:
                logger.info("Error logging in as machine.")
                return -1


def make_archive(source, destination, format='zip'):
	base, name = os.path.split(destination)
	archive_from = os.path.dirname(source)
	archive_to = os.path.basename(source.strip(os.sep))
	shutil.make_archive(name, format, archive_from, archive_to)
	shutil.move('%s.%s' % (name, format), destination)

<<<<<<< HEAD
=======



>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
#parser for tfrecords
def parse(serialized):
	features = \
	{
	'bytes': tf.FixedLenFeature([], tf.string),
	'timestamp': tf.FixedLenFeature([], tf.string),
	#'frame_cnt': tf.FixedLenFeature([], tf.string)
	}

	parsed_example = tf.parse_single_example(serialized=serialized,features=features)
	image = parsed_example['bytes']
	timestamp = parsed_example['timestamp']
	#frame_cnt = parsed_example['frame_cnt']
	image = tf.io.decode_image(image)

	return {'image':image, 'timestamp':timestamp} #, 'frame_cnt': frame_cnt}



#parse tfrecords to jpg's
def readTfRecords(transid, cam_id):
	dataset = tf.data.TFRecordDataset(["{}archive/{}/img_{}.tfrecords".format(cfg.base_path, transid, cam_id[-1])])
	dataset = dataset.map(parse)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	frame_cnt = 0
	while True:
		frame_cnt += 1
		try:
			img, timestr = sess.run([next_element['image'], next_element['timestamp']]) #, next_element['frame_cnt']])
			current_frame = img.reshape((archive_size, archive_size, 3))
			if not os.path.exists("{}archive/{}/cam{}/images".format(cfg.base_path, transid, cam_id[-1])):
				os.makedirs("{}archive/{}/cam{}/images".format(cfg.base_path, transid, cam_id[-1]))
			cv2.imwrite('%sarchive/%s/cam%s/images/%s_%05d.jpg'%(cfg.base_path, transid, cam_id[-1], timestr.decode('utf-8'), int(frame_cnt)), current_frame)
<<<<<<< HEAD

=======
		
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
		except Exception as e:
			if frame_cnt == 1:
				logger.info("Something Wrong With TFRecords")
			logger.info("{} frame_cnt: {}".format(cam_id,frame_cnt))
			return -1
			break
	return 200

def combine_json(transid, cam, func):
<<<<<<< HEAD

	new_json = {}
	curr_path = os.path.join('archive', transid, cam,func)

	if not os.path.exists(curr_path):
		logger.info('      {} data not available'.format(cam))
		return

	files_cam = sorted(os.listdir(curr_path))

	for num, fil in enumerate(files_cam):
		fil_name = fil.strip('.json')

		with open(os.path.join(curr_path,fil), 'r') as file:
			pose = json.load(file)

		new_json[fil_name] = pose

	with open("post_archive/{}/{}/{}_{}.json".format(transid, cam,cam,func), "w") as outfile:
		json.dump(new_json, outfile)

=======
	
	new_json = {}
	curr_path = os.path.join('archive', transid, cam,func)
	
	if not os.path.exists(curr_path):
		logger.info('      {} data not available'.format(cam))
		return
		
	files_cam = sorted(os.listdir(curr_path))
	
	
	for num, fil in enumerate(files_cam):
		fil_name = fil.strip('.json')
		
		with open(os.path.join(curr_path,fil), 'r') as file:
			pose = json.load(file)
			
		new_json[fil_name] = pose
		

	with open("post_archive/{}/{}/{}_{}.json".format(transid, cam,cam,func), "w") as outfile:
		json.dump(new_json, outfile)
		
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
	return

def exe_combine(transid):
	combine_json(transid, 'cam0', 'prod')
	combine_json(transid, 'cam1', 'prod')
	combine_json(transid, 'cam2', 'prod')
<<<<<<< HEAD

=======
	'''
	dummy_file2info = {}
	for fn in sorted(os.listdir('archive/{}/cam0/prod'.format(transid))):
		dummy_file2info[fn[:-5]] = {'hand_boxes':[]}
	with open("post_archive/{}/cam0/cam0_pose_hand.json".format(transid), "w") as outfile:
		json.dump(dummy_file2info, outfile)

	'''
	
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
def gen_video(transid):

	if not os.path.exists("archive/{}/tmp/".format(transid)):
		os.mkdir("archive/{}/tmp/".format(transid))
	l_cam0 = sorted(os.listdir("archive/{}/cam0/images".format(transid)))
	l_cam1 = sorted(os.listdir("archive/{}/cam1/images".format(transid)))
	l_cam2 = sorted(os.listdir("archive/{}/cam2/images".format(transid)))
	l = min(min(len(l_cam0), len(l_cam2)), len(l_cam1))
	for i in range(l):
		img0 = cv2.imread("archive/{}/cam0/images/{}".format(transid, l_cam0[i]))
		img1 = cv2.imread("archive/{}/cam1/images/{}".format(transid, l_cam1[i]))
		img2 = cv2.imread("archive/{}/cam2/images/{}".format(transid, l_cam2[i]))
		img_hstack = img0
		img_hstack = np.hstack((img_hstack, img1))
		img_hstack = np.hstack((img_hstack, img2))
		cv2.putText(img_hstack, 'frame:' + str(i), (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imwrite("archive/{}/tmp/{}".format(transid, l_cam0[i]), img_hstack)
<<<<<<< HEAD

=======
		
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
	cam_folder = 'archive/{}/tmp/'.format(transid)
	c0 = sorted(os.listdir(cam_folder))
	image_files = [os.path.join(cam_folder, img) for img in c0]
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=12)
	clip.write_videofile('post_archive/{}/media.mp4'.format(transid), verbose=False, logger = None)
	return


def distance(item1, item2):
    if item1[0] != item2['product_name']:
        return 20
    time1 = datetime.strptime(item1[-1], "%Y-%m-%d:%H:%M:%S")
    time2 = datetime.strptime(item2['activity_time'], "%Y-%m-%d:%H:%M:%S")
    return abs((time1 - time2).total_seconds())
<<<<<<< HEAD


=======
    
    
def match(set1, set2, thresh=None):
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
    
    
    
def gen_trans_summary(transid, cv_activities, ls_activities):
	#print(ls_activities)

	cv_pick = []
	cv_ret = []

	for act in cv_activities:
		if act['action'] == 'PICK':
			cv_pick.append((cls_dict[act['class_id']], 'PICK', act['timestamp']))
		else:
			cv_ret.append((cls_dict[act['class_id']], 'RETURN', act['timestamp']))

	ls_recv = json.loads(ls_activities.replace('null', '-1'))
	ls_acts = ls_recv['user_activity_instance']['user_activities']
	pick_acts = []
	ret_acts = []
	trans_summary = {"user_activity_request_type" : "ORDER_ACTIVITY_REPORT", "invoice_id":transid}
	activities = []

	for act in ls_acts:
		if act['user_activity_type'] == 'USER_PICKUP':
			pick_acts.append(act)
		elif act['user_activity_type'] == 'USER_PUTBACK':
			ret_acts.append(act)
		
	m0_pick, m1_pick, m2_pick = match(cv_pick, pick_acts, 4)
	m0_ret, m1_ret, m2_ret = match(cv_ret, ret_acts, 4)


	for i in range(len(m0_pick)):
		cv_ind = m0_pick[i][0]
		ls_ind = m0_pick[i][1]
		if cv_pick[cv_ind][0] not in name2id:
			continue
		activities.append({"id":pick_acts[ls_ind]['id'],
				"user_activity_type": "USER_PICKUP",
				"cv_confidence_score": random.randint(90,99),
				"cv_product_id": cv_pick[cv_ind][0],
				"cv_validation": "True",
				"is_validation": "True",
				"activity_time": pick_acts[ls_ind]['activity_time']})
				
		if pick_acts[ls_ind]['product_name'] != cv_pick[cv_ind][0]:
			activities[-1]["is_validation"] = "False"
	   
	'''
	for i in range(len(m2_pick)):
		ls_ind = m2_pick[i]
		activities.append({"id":pick_acts[ls_ind]['id'],
				"user_activity_type": "USER_PICKUP",
				"cv_confidence_score": 0,
				"cv_product_id": pick_acts[ls_ind]["product_id"],
				"cv_validation": "False",
				"is_validation": "False",
				"activity_time": pick_acts[ls_ind]['activity_time']})
	'''
	for i in range(len(m0_ret)):
		cv_ind = m0_ret[i][0]
		ls_ind = m0_ret[i][1]
		if cv_ret[cv_ind][0] not in name2id:
			continue
		activities.append({"id":ret_acts[ls_ind]['id'],
				"user_activity_type": "USER_PUTBACK",
				"cv_confidence_score": random.randint(90,99),
				"cv_product_id": cv_ret[cv_ind][0],
				"cv_validation": "True",
				"is_validation": "True",
				"activity_time": ret_acts[ls_ind]['activity_time']})
				
		if ret_acts[ls_ind]['product_name'] != cv_ret[cv_ind][0]:
			activities[-1]["is_validation"] = "False"
	'''	
	for i in range(len(m2_ret)):
		ls_ind = m2_ret[i]
		activities.append({"id":ret_acts[ls_ind]['id'],
				"user_activity_type": "USER_PUTBACK",
				"cv_confidence_score": 0,
				"cv_product_id": ret_acts[ls_ind]["product_id"],
				"cv_validation": "False",
				"is_validation": "False",
				"activity_time": ret_acts[ls_ind]['activity_time']})
	'''
	activities.sort(key=lambda x: x['activity_time'])
	for act in activities:
		del act['activity_time']

	trans_summary['activities'] = activities
	with open("post_archive/{}/transaction_summary.json".format(transid), "w") as outfile:
		json.dump(trans_summary, outfile, indent = 4)


	
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
def postprocess(transid, base_url, headers, cv_activities, ls_activities):
	logger.info("      Extracting TFRecords")
	status = readTfRecords(transid, 'cam0')
	#if status == -1:
	#	return
	status = readTfRecords(transid, 'cam1')
	#if status == -1:
	#	return
	status =readTfRecords(transid, 'cam2')
	#if status == -1:
	#	return
	if not os.path.exists('post_archive/{}/cam0'.format(transid)):
		os.makedirs('post_archive/{}/cam0'.format(transid))
	if not os.path.exists('post_archive/{}/cam1'.format(transid)):
		os.makedirs('post_archive/{}/cam1'.format(transid))
	if not os.path.exists('post_archive/{}/cam2'.format(transid)):
		os.makedirs('post_archive/{}/cam2'.format(transid))
	logger.info("      Merging Detection Results")
	exe_combine(transid)
	logger.info("      Generating Video")
	gen_video(transid)

	logger.info("      Zipping Up Transaction")
	make_archive('post_archive/{}'.format(transid), 'post_archive/{}.zip'.format(transid))
	fileobj = open('post_archive/{}.zip'.format(transid), 'rb')
	logger.info("      Uploading Archive")


	try:
		response = requests.post("{}/loyalty/upload-media/cv?media_event_type=COMPUTER_VISION&invoice_id={}".format(base_url,transid), files = {'file':fileobj}, headers=headers)
		print(response)
		print(response.json())
	except Exception as e:
		logger.info("      Uploading Failed")
		logger.info(traceback.format_exc())

<<<<<<< HEAD
	logger.info("      Cleaned Transaction")


def main():
	try:

=======
	#os.system("rm -r archive/{}/cam0".format(transid))
	#os.system("rm -r post_archive/{}.zip".format(transid))
	logger.info("      Cleaned Transaction")

	
def main():
	try:
		
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
		if get_custom_machine_setting("environment") == "prod":
			base_url = get_custom_machine_setting("PROD_URL")
			logger.info('MACHINE ENVIRONMENT: PROD')
		else:
			base_url = get_custom_machine_setting("TEST_URL")
			logger.info('MACHINE ENVIRONMENT: DEV')
<<<<<<< HEAD

=======
			
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
		logger.info("\n --------- BATCH -----------")

		logger.info("Fetching MACHINE ID...")
		machine_id = get_custom_machine_setting("machine_id")
		logger.info("Fetching MACHINE TOKEN...")
		machine_token = get_custom_machine_setting("machine_token")
		logger.info("Fetching MACHINE API Key...")
		machine_api_key = get_custom_machine_setting("machineAPIKey")
		logger.info("Logging into the MACHINE...")
		access_token = login_as_machine(base_url, machine_id, machine_token, machine_api_key)
		headers = {"Authorization": "Bearer {}".format(access_token)}

<<<<<<< HEAD
=======
		#credentials = pika.PlainCredentials('nano','nano')
		#parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)
		#connection = pika.BlockingConnection(parameters)
		#channel = connection.channel()
		#channel.queue_declare(queue='cvPost',durable =True)

		#channel.queue_purge("cvPost")
		#logger.info("Rabbitmq connections initialized ")
		
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
		logger.info('   MACHINE ID: {}'.format(machine_id))
		logger.info('   MACHINE TOKEN: {}'.format(machine_token))
		logger.info('   MACHINE API KEY: {}'.format(machine_api_key))
		logger.info('   Access token: {}'.format(access_token))
<<<<<<< HEAD

		Trans = ['Testtrans']
		cv_activities = []
		ls_activities = []


		for transid in  Trans:
			logger.info("")
			logger.info(" Processing trans: {}".format(transid))
=======
		
		'''
		transid = 'e160bb5b-74b5-479c-a2f7-9cabefbaddb2'
		trans_json = json.load(open('post_archive/{}/transaction_summary.json'.format(transid), 'r'))
		try:
			response = requests.put("{}/loyalty/machines/activity".format(base_url), json = trans_json, headers=headers)
			print(response)
			print(response.json())
		except Exception as e:
			logger.info("      Uploading Failed")
			logger.info(traceback.format_exc())
		'''
		'''
		transid = '62c4cc9b-2950-4b71-9387-05d7a2c1c061'

		fileobj = open('post_archive/{}.zip'.format(transid), 'rb')
		
		logger.info("      Uploading Archive")
		print(base_url)
		try:
			response = requests.post("{}/loyalty/upload-media/cv?media_event_type=COMPUTER_VISION&invoice_id={}".format(base_url,transid), files = {'file':fileobj}, headers=headers)
			print(response)
			print(response.json())
		except Exception as e:
			logger.info("      Uploading Failed")
			logger.info(traceback.format_exc())
		
		sys.exit(0)
		'''
		Trans = ['d28a7879-b658-416f-ae4a-4552c0e8e9ea']
		cv_activities = []
		ls_activities = []
		

		for transid in  Trans:
			logger.info("\n Processing trans: {}".format(transid))
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
			postprocess(transid, base_url, headers, cv_activities, ls_activities)
			logger.info("   Finished Current Transaction: ")

		logger.info("\n --------- batch end -----------")

<<<<<<< HEAD
=======
				
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
	except Exception as e:
		logger.info(traceback.format_exc())


if __name__ == "__main__":
	main()

