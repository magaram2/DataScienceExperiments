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
    from pyspark.mllib.classification import SVMWithSGD
#    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.mllib.classification import LabeledPoint
    from pyspark.mllib.util import MLUtils

    print("Successfully loaded Spark and MLlib classes...")

except ImportError as e:
    print("Error importing spark modules", e)
    sys.exit(1)


from numpy import array

conf = SparkConf().setAppName("RecessionPredictionModel").setMaster("local")

sc = SparkContext(conf=conf)

parsedData = MLUtils.loadLibSVMFile(sc, "/Users/agaram/development/DataScienceExperiments/econometricsPoc/svm")

print('Number of samples = {0}'.format(parsedData.count()))

training, test = parsedData.randomSplit([0.3, 0.7], 11)

numIterations = 100

model = SVMWithSGD.train(training, numIterations)

model.clearThreshold()

scoreAndLabelsRaw = test.map(lambda point: (model.predict(point.features), point.label))

print('Printing RAW Scores and Labels')

labeledScoresRaw = scoreAndLabelsRaw.collect()


model.setThreshold(1)

scoreAndLabels = test.map(lambda point: (model.predict(point.features), point.label))

print('Printing Scores and Labels [Threshold = 1]')

labeledScores = scoreAndLabels.collect()

for labeledScore in labeledScores:
    print(labeledScore)

countTruePositive = scoreAndLabels.filter(lambda (p, v): (p == 1) and (v == 1)).count()
countTrueNegative = scoreAndLabels.filter(lambda (p, v): (p == 0) and (v == 0)).count()
countFalsePositive = scoreAndLabels.filter(lambda (p, v): (p == 1) and (v == 0)).count()
countFalseNegative = scoreAndLabels.filter(lambda (p, v): (p == 0) and (v == 1)).count()

testAccuracy = (countTruePositive + countTrueNegative)/float(scoreAndLabels.count())
testError = (countFalsePositive + countFalseNegative)/float(scoreAndLabels.count())

precision = countTruePositive / float(countTruePositive + countFalsePositive)
recall = countTruePositive / float(countTruePositive + countFalseNegative)
f1score = (2*precision*recall)/float(precision + recall)

print('Number of Score & Labels = {0}'.format(scoreAndLabels.count()))
print('True positive = {0}'.format(countTruePositive))
print('True negative = {0}'.format(countTrueNegative))
print('False positive = {0}'.format(countFalsePositive))
print('False negative = {0}'.format(countFalseNegative))

print('Precision = {0}'.format(precision))
print('Recall = {0}'.format(recall))
print('FScore = {0}'.format(f1score))
print('Test Accuracy = {0}'.format(testAccuracy))
print('Test Error = {0}'.format(testError))


for labeledScoreRaw in labeledScoresRaw:
    print(labeledScoreRaw)

import matplotlib.pyplot as plt

print('Plotting Raw Labels and Scores..')
plt.plot(labeledScoresRaw)
plt.show()




