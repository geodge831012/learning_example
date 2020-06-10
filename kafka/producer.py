from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')   #连接kafka

msg = "this is a new msg".encode('utf-8')
producer.send('test1', key='cd'.encode('utf-8'), value=msg, partition=10)

producer.close()
