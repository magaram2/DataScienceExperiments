import os
import sys

os.environ['SPARK_HOME'] = '/usr/local/spark/spark-1.2.0-bin-hadoop2.4'


#    Add PySpark to the Python Path manually before invoking the MLlib functions
sys.path.append('~/anaconda/lib/python2.7/site-packages')
sys.path.append('/usr/local/spark/spark-1.2.0-bin-hadoop2.4/python')
sys.path.append('/usr/local/spark/spark-1.2.0-bin-hadoop2.4/python/lib/py4j-0.8.2.1-src.zip')


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

conf = SparkConf().setAppName("RecessionPredictionModel").setMaster("local")

sc = SparkContext(conf=conf)

data = sc.textFile("/Users/agaram/development/DataScienceExperiments/econometricsPoc/EconometricsDataSlope.csv/Sheet1-Table1.csv")

parsedData = data.map(lambda line: LabeledPoint([float(x) for x in line.split(',')[1:8]][6],
                                                array([float(x) for x in line.split(',')[1:8]])))

MLUtils.saveAsLibSVMFile(parsedData, "/Users/agaram/development/DataScienceExperiments/econometricsPoc/svmDataSlope")

