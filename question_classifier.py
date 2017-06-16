import spacy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

from Tkinter import Tk
from tkFileDialog import askopenfilename


nlp = spacy.load('en')

def read_data(filename = 'labelleddata.txt'):
	"""
		function to read data from given filename. 
		return : list of tuples of question and class.
	"""
	f = open(filename, 'r')
	question_class = []
	for line in f:
		parts = line.split(',,,')
		question_class.append( (parts[0].strip(), parts[1].strip()) )
	return question_class


# first word, first word's pos tag, second word, second word's pos tag, first wh word if none => -1.
def get_feature(text):
	"""
		given a question it returns the feature vector for the question which is later used for classification.
		returns feature vector of given text
	"""
	s = []
	for t in text:
		if ord(t) < 128:
			s.append(t)
	text = ''.join(s)
	doc = nlp(u'' + text)
	i = 0
	feature_vector = [-1] * 5
	for word in doc:
		if word.tag_[0] == 'W' and feature_vector[4] == -1:
			feature_vector[4] = word.lemma_
		if i == 0:
			feature_vector[0] = word.lemma_
			feature_vector[1] =  word.tag_
		if i == 1:
			feature_vector[2] = word.lemma_
			feature_vector[3] = word.tag_

		i += 1

	return feature_vector
	
def get_mapping(value_list):
	"""
		maps list of string to integer.

		return: returns abovemapping
	"""
	res = {}
	i = 0
	for val in value_list:
		res[str(val)] = i
		i += 1

	return res

def map_data_to_int(all_data, label):
	"""
		function to convert column named label in all_data with string entries to integer entries. 

		return : mapping used, data modified
	"""
	mapping = get_mapping(all_data[label].unique())
	all_data.replace({label: mapping}, inplace = True)

	return mapping, all_data

def main():
	"""
		1) we first read the train data. 
		2) Features considered are  'First Word', 'First Word PosTag', 'Second Word', 'Second Word Pos Tag', first 'Wh- Word'  present (question word) 
		3) Next we create a pandas data frame with mapping all the strings to integer.
		4) Next we train model and use it for predcition.
	
	"""
	print "---------------------- Choose training data file (Choose : data/labelleddata.txt) --------------------"
	Tk().withdraw() 
	train_filename = askopenfilename() # pop up for selection
	
	all_data = read_data(train_filename)
	labels = ['FirstWord', 'FirstWordTag', 'SecondWord', 'SecondWordTag', 'WhWord', 'Class']

	all_data_vectors = []

	for data in all_data:
		vector = get_feature(data[0])
		vector.append(data[1])
		all_data_vectors.append(vector)

	# constructing pandas data frame for better data manipulation
	all_data = pd.DataFrame.from_records(all_data_vectors, columns = labels)


	first_word_map, all_data = map_data_to_int(all_data, 'FirstWord')
	first_word_tag_map, all_data = map_data_to_int(all_data, 'FirstWordTag')
	second_word_map, all_data = map_data_to_int(all_data, 'SecondWord')
	second_word_tag_map, all_data = map_data_to_int(all_data, 'SecondWordTag')
	wh_word_map, all_data = map_data_to_int(all_data, 'WhWord')

	class_map = {'unknown': 0, 'what': 1, 'who': 2, 'when': 3, 'affirmation':4}

	all_data = all_data.replace({'Class': class_map})

	#train, test = train_test_split(all_data, test_size = 0.2)

	train = all_data
	y_train = train['Class']
	#y_test = test['Class']
	x_train = train.drop('Class', axis=1)
	#x_test = test.drop('Class', axis=1)
	print "---------------------- Trainig the model, please wait --------------------"
	clf = RandomForestClassifier(n_estimators=500,max_features=3,min_samples_split=5,oob_score=True)
	clf.fit(x_train, y_train)

	print "---------------------- Choose testing data file --------------------"
	Tk().withdraw()
	input_filename = askopenfilename() 
	input_file = open(input_filename, 'r')

	output_file = open('test_output.txt', 'w')

	for line in input_file:
		text_feature = get_feature(line)
		if text_feature[0] in first_word_map:
			text_feature[0] = first_word_map[text_feature[0]]
		else:
			text_feature[0] = -1

		if text_feature[1] in first_word_tag_map:
			text_feature[1] = first_word_tag_map[text_feature[1]]
		else:
			text_feature[1] = -1

		if text_feature[2] in second_word_map:
			text_feature[2] = second_word_map[text_feature[2]]
		else:
			text_feature[2] = -1

		if text_feature[3] in second_word_tag_map:
			text_feature[3] = second_word_tag_map[text_feature[3]]
		else:
			text_feature[3] = -1

		if text_feature[4] in wh_word_map:
			text_feature[4] = wh_word_map[text_feature[4]]
		else:
			text_feature[4] = -1

		reverse_class_map = {}
		for (key, value) in class_map.iteritems():
			reverse_class_map[value] = key

		res = clf.predict_proba([text_feature])
		j = 0
		ans = 'unknown'
		for val in res[0]:
			if val > 0.7:
				ans = reverse_class_map[j]
			j += 1
		output_file.write(line.strip() + ' ,,,, ' + ans+'\n')

	output_file.close()
	print "output written in test_output.txt"

if __name__ == '__main__':
	main()
