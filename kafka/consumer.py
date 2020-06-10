from kafka import KafkaConsumer,TopicPartition

#consumer = KafkaConsumer('test', bootstrap_servers=['localhost:9092'])
consumer = KafkaConsumer('test1', auto_offset_reset='earliest', bootstrap_servers=['localhost:9092'])

#consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
#consumer.assign([TopicPartition(topic='test', partition=0)])
#consumer.seek(TopicPartition(topic='test', partition=0), offset=3)

for msg in consumer:
    recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
    print(recv)
