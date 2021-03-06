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

    print("Successfully loaded Spark and MLlib classes...")

except ImportError as e:
    print("Error importing spark modules", e)
    sys.exit(1)


from numpy import array

conf = SparkConf().setAppName("RecessionPredictionModel").setMaster("local")

sc = SparkContext(conf=conf)

##################################################################
################  Train the machine ##############################
##################################################################

data = sc.textFile("/Users/agaram/development/DataScienceExperiments/econometricsPoc/ECDTraining.csv")
print('data = {0}'.format(data))

parsedData = data.map(lambda line: LabeledPoint([float(x) for x in line.split(',')][4],
                                                array([float(x) for x in line.split(',')])))
print('parsedData = {0}'.format(parsedData))

model = LogisticRegressionWithSGD.train(parsedData)

labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))

print('labelsAndPreds = {0}'.format(labelsAndPreds))

trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

trainAccuracy = labelsAndPreds.filter(lambda (v, p): v == p).count() / float(parsedData.count())

print('Training Accuracy = ', trainAccuracy*100)
print('Training Error = ', trainErr*100)

######################################################
############ Testing the Machine #####################
######################################################

data1 = sc.textFile("/Users/agaram/development/DataScienceExperiments/econometricsPoc/ECDTestingData.csv")

parsedData1 = data1.map(lambda line: LabeledPoint([float(x) for x in line.split(',')][4], array([float(x) for x in line.split(',')])))

model1 = LogisticRegressionWithSGD.train(parsedData1)

labelsAndPreds1 = parsedData1.map(lambda p: (p.label, model.predict(p.features)))

print(labelsAndPreds1)

trainErr1 = labelsAndPreds1.filter(lambda (v, p): v != p).count() / float(parsedData1.count())

countTruePositive = labelsAndPreds1.filter(lambda (v, p): v == p and v == 1 and p == 1).count()
countTrueNegative = labelsAndPreds1.filter(lambda (v, p): v == p and v == 0 and p == 0).count()
countFalsePositive = labelsAndPreds1.filter(lambda (v, p): v != p and v == 0 and p == 1).count()
countFalseNegative = labelsAndPreds1.filter(lambda (v, p): v != p and v == 1 and p == 0).count()

precision = countTruePositive / float(countTruePositive + countFalsePositive)
recall = countTruePositive / float(countTruePositive + countFalseNegative)
f1score = (2*precision*recall)/float(precision + recall)
print('************* Model metrics validation data ***************************')
print('**     Count of True Positives = {0}'.format(countTruePositive))
print('**     Count of True Negatives = {0}'.format(countTrueNegative))
print('**     Count of False Positives = {0}'.format(countFalsePositive))
print('**     Count of False Negatives = {0}'.format(countFalseNegative))
print('***********************************************************************')

print('Precision = {0:.2f}'.format(precision))
print('Recall = {0:.2f}'.format(recall))
print('F1Score = {0:.2f}'.format(f1score))


trainAccuracy1 = labelsAndPreds1.filter(lambda (v, p): v == p).count() / float(parsedData1.count())


print('Testing Accuracy = ', trainAccuracy1*100)
print('Testing Error = ', trainErr1*100)







