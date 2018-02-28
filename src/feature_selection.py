import numpy 
import math

'''
Given the dataset (typically training is sent), this returns the best features based on the selected configuration for feature selection

Params:
X -> numpy array of input (only numeric)
y -> numpy array of corresponding output.
config -> dictionary of configuration. 

Return: 
selected feature indices. Data is not changed.

'''

def select_features(data, config):
	method = config["method"]
	num_features = config["num_features"]
	X = data[:,:-1]
	y = data[:,-1]

	if method == "info_gain":
		return list(information_gain(X, y)[0: num_features])

'''
Computes information gain of all features and returns the sorted indices with highest information gain
'''

def information_gain(X, y):

	cols = X.shape[1]
	info_gains = numpy.zeros(cols)
	for i in range(cols):
		values = X[:,i]
		info_gains[i] = calculate_info_gain(values, y)

	return numpy.argsort(info_gains)[::-1]

'''
Calculates information gain for a given feature
'''

def calculate_info_gain(values, y):

	ig = entropy(count(y))
	distinct = set(values)
	for val in distinct:
		indices = numpy.argwhere(values==1).flatten()
		ig -= entropy(count(y[indices]))

	return ig

'''
Creates a dictionary of counts for a give set of values.
'''
def count(values):
	unique, counts = numpy.unique(values, return_counts=True)
	return dict(zip(unique, counts))


'''
Calculates entropy
'''
def entropy(category_value_dict):
	total = sum(category_value_dict.values())

	ent = 0.0
	for key in category_value_dict:
		prob = float(category_value_dict[key])/float(total)
		ent += (-prob * math.log(prob))

	return ent