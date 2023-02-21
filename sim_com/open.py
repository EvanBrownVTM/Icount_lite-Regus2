#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('guest','guest')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvIcount",durable = True)

data = '{\n "cmd": "DoorOpened", \n "parm1":"trans702:True"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvIcount",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
