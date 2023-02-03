#!/usr/bin/env python
import pika
import json
<<<<<<< HEAD

credentials = pika.PlainCredentials('nano','nano')
=======
import sys
sys.path.insert(0, '../')
import configSrc as cfg

credentials = pika.PlainCredentials(cfg.pika_name,cfg.pika_name)
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

<<<<<<< HEAD
channel.queue_declare(queue="cvRequest",durable = True)

data = '{\n "cmd": "DoorOpened", \n "parm1":"Testtrans:True"\n}'
=======
channel.queue_declare(queue="cvIcount",durable = True)

data = '{\n "cmd": "DoorOpened", \n "parm1":"trans702:True"\n}'
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
<<<<<<< HEAD
                        routing_key="cvRequest",
=======
                        routing_key="cvIcount",
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
<<<<<<< HEAD
=======


'''
credentials = pika.PlainCredentials(cfg.pika_name,cfg.pika_name)
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvArchive1",durable = True)

data = '{\n "cmd": "DoorOpened", \n "parm1":"trans1"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvArchive1",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
'''
>>>>>>> 829de47c8188bdfc7dc5d3253d63e97a9bc70cad
