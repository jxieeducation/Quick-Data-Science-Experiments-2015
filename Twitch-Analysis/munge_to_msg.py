from pyspark import SparkContext, SparkConf
import xmltodict
import json

sc = SparkContext(appName="twitch_data_munging")
twitch = sc.textFile('hdfs://localhost:9000/data/twitch.xml')
# twitch = twitch.sample(False, 0.0001, 1337)

def parseXML(msg):
	try:
		return xmltodict.parse('<root>' + msg + '</root>')['root']['msg']
	except:
		return ''


messages = twitch.map(parseXML).filter(lambda msgDict: msgDict is not '')
messages.saveAsTextFile('hdfs://localhost:9000/data/twitch_msg')
