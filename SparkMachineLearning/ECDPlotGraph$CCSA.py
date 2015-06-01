__author__ = 'agaram'
import os
import sys

os.environ['SPARK_HOME'] = '/usr/local/spark/spark-1.3.0-bin-hadoop2.4'


#    Add PySpark to the Python Path manually before invoking the MLlib functions
sys.path.append('~/anaconda/lib/python2.7/site-packages')
sys.path.append('/usr/local/spark/spark-1.3.0-bin-hadoop2.4/python')
sys.path.append('/usr/local/spark/spark-1.3.0-bin-hadoop2.4/python/lib/py4j-0.8.2.1-src.zip')


# Try and import the PySpark classes
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.classification import LogisticRegressionWithSGD
    from pyspark.mllib.classification import LabeledPoint
    from pyspark.mllib.util import MLUtils

    print("Successfully loaded Spark and MLlib classes...")

except ImportError as e:
    print("Error importing spark modules", e)
    sys.exit(1)


from numpy import array

conf = SparkConf().setAppName("LoadDataAsLibSVM").setMaster("local")

sc = SparkContext(conf=conf)

data = sc.textFile("/Users/agaram/development/DataScienceExperiments/econometricsPoc/ECDTotalData.csv")
print('data = {0}'.format(data))
training, test = data.randomSplit([0.3, 0.7], 11)

parsedData = training.map(lambda line: [float(x) for x in line.split(',')])
print('Collecting Data...')

pointData = parsedData.collect()
print(pointData)

import matplotlib.pyplot as plt

f3 = [pointDataItem[2] for pointDataItem in pointData]
f5 = [pointDataItem[4] for pointDataItem in pointData]


plt.plot(f3, 'r-', label='CCSA')
plt.plot(f5, 'k-', label='Recession Prediction')
plt.legend()


plt.show()


