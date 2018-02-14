import sys
import numpy as np
import math
from collections import defaultdict
from sklearn.model_selection import KFold

# Simple elegant method for phishing url classification.
# Uses Naive Bayes method.

def read_data(filename):
    data = np.array(map(lambda x: x.split(","), open(filename).readlines()[1:]))
    return data

def split_train_test(data, ratio_train=0.8):
	np.random.shuffle(data)
	X = data[:,:-1]
	y = data[:,-1]
	total_size = X.shape[0]
	train_size = int(total_size * ratio_train)
	return X[:train_size,:], y[:train_size], X[train_size:,:], y[train_size:]

def train(X, y):

	N = X.shape[0]
	class_prob = defaultdict(float)
	class_feature_value_count = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

	for i,row in enumerate(X):
		class_prob[y[i]] += 1.0

		for j,col in enumerate(row):
			class_feature_value_count[y[i]][j][col] += 1


	for cls in class_prob:
		for j in class_feature_value_count[cls]:
			for val in class_feature_value_count[cls][j]:
				class_feature_value_count[cls][j][val] = class_feature_value_count[cls][j][val] / class_prob[cls]


		class_prob[cls] /= float(N)

	return class_prob, class_feature_value_count

def predict(X_test, y_test, class_prob, class_feature_value_count):
	correct = 0
	for i,row in enumerate(X_test):
		true_class = y_test[i]
		predicton = None
		max_val = float("-inf")

		for cls in class_prob:
			val = math.log(class_prob[cls])
			for feature in class_feature_value_count[cls]:
				val += class_feature_value_count[cls][feature][row[feature]]

			if val > max_val:
				max_val = val
				predicton = cls

		if true_class == predicton:
			correct += 1

	accuracy = 100*float(correct)/X_test.shape[0]
	
	return accuracy


print "Phishing URL predictor - Naive Bayes approach"

training_file = sys.argv[1]
option = "random"
if len(sys.argv) > 2:
	option = sys.argv[2]

if option == "random":
	#Random shuffle run
	split = 0.8
	if len(sys.argv) == 4:
		split = float(sys.argv[3])

	data = read_data(training_file)
	X_train, y_train, X_test, y_test = split_train_test(data, split)
	class_prob, class_feature_value_count = train(X_train, y_train)
	accuracy = predict(X_test, y_test, class_prob, class_feature_value_count)
	print "\n"
	print "\n"
	print "Random test-train split. Training ratio = "+str(split)+"."
	print "\n"
	print "Accuracy = "+str(accuracy)+" %."
	print "\n"

elif option == "cv":
	k = 5
	if len(sys.argv) == 3:
		k = int(sys.argv[2])

	kf = KFold(n_splits=k)
	data = read_data(training_file)
	np.random.shuffle(data)
	X = data[:,:-1]
	y = data[:,-1]
	accuracy = 0.0
	for train_idx, test_idx in kf.split(data):
		class_prob, class_feature_value_count = train(X[train_idx],y[train_idx])
		accuracy += predict(X[test_idx], y[test_idx], class_prob, class_feature_value_count)
	print "\n"
	print "\n"
	print "Cross-validated. Folds = "+str(k)+"."
	print "Average Accuracy over different folds = "+str(accuracy/k)
	print "\n"

else:
	print "Illegal options set. Use either 'random' or 'cv'"

