# Python code for the article 
# Logistic Regression using Apache Spark by Leonard Giura
# See http://technobium.com/logistic-regression-using-apache-spark/

# First import packages and classes that we will need throughout

# In[182]:

import numpy as np

from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionWithSGD


# Make sure that we have the files in the right place

# In[183]:

get_ipython().run_cell_magic(u'sh', u'', u'ls playground/')


# Peek at the data set

# In[184]:

get_ipython().run_cell_magic(u'sh', u'', u'head playground/Qualitative_Bankruptcy.data.txt')


# Load the data set into a Spark RDD

# In[206]:

data = sc.textFile('playground/Qualitative_Bankruptcy.data.txt')


# Check to makes sure the data set is loaded correctly

# In[207]:

print data.count()

assert data.count() == 250


# Read few lines

# In[208]:

data.take(2)


# The dictionary `getDoubleValue` map the categorical features into numerial representation
# 
# The function `line_parser()` transform each line into a `LabeledPoint` object
# 
# Finally perform try the function on some examples

# In[209]:

getDoubleValue = { 'P' : 3.0, 'A' : 2.0, 'N' : 1.0, 'NB': 1.0, 'B': 0.0 }

def line_parser(line):
    tokens = line.split(',')
    label = getDoubleValue[tokens[-1]]
    features = map(lambda t: getDoubleValue[t], tokens[:-1])
    return LabeledPoint(label, features)

lp = line_parser(example_line)
print lp

assert lp.label == 1.0
assert np.allclose(lp.features, [3.0, 3.0, 2.0, 2.0, 2.0, 3.0])


# Map the data set into a data set of `LabeledPoint`s

# In[210]:

parsedData = data.map(line_parser)

print parsedData.take(1)[0]

# Integrity check
assert parsedData.filter(lambda lp: lp.label != 1.0 and lp.label != 0.0).isEmpty()


# Split the data into training and test (we're missing the validation set)

# In[211]:

trainingData, testData = parsedData.randomSplit([0.6, 0.4], seed = 434)


# Train two logistic regression models with two different optimizers (LBFGS and SGD).

# In[212]:

model1 = LogisticRegressionWithLBFGS.train(trainingData, iterations = 100, intercept = True, numClasses = 2)
model2 = LogisticRegressionWithSGD.train(trainingData, iterations = 100, intercept = True)

# Print the model parameters
print model1
print model2


# Compare the models on few random samples

# In[213]:

samples = trainingData.sample(False, 10.0 / 250.0).collect()
for point in samples:
    print point, model1.predict(point.features), model2.predict(point.features)


# Evaluate the training and test errors

# In[217]:

trainingLabelAndPreds1 = trainingData.map(lambda point: (point.label, model1.predict(point.features)))
trainingError1 = trainingLabelAndPreds1.map(lambda (r1, r2): float(r1 != r2)).mean()
print 'LBFGS training error =', trainingError1

testLabelAndPreds1 = testData.map(lambda point: (point.label, model1.predict(point.features)))
testError1 = testLabelAndPreds1.map(lambda (r1, r2): float(r1 != r2)).mean()
print 'LBFGS test error =',testError1

trainingLabelAndPreds2 = trainingData.map(lambda point: (point.label, model2.predict(point.features)))
trainingError2 = trainingLabelAndPreds2.map(lambda (r1, r2): float(r1 != r2)).mean()
print 'SGD training error =', trainingError2

testLabelAndPreds2 = testData.map(lambda point: (point.label, model2.predict(point.features)))
testError2 = testLabelAndPreds2.map(lambda (r1, r2): float(r1 != r2)).mean()
print 'SGD test error =', testError2
