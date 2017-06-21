import tensorflow as tf
import numpy as np
#from tf2 import create_feature_sets_and_labels
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("D:\Work\TF", one_hot = True)
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_line = 1000000

def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos,neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:hm_line]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
	
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w]>50:
			l2.append(w)
	
	print(len(l2))
	return l2 
	
def sample_handling(sample, lexicon, classification):
	featureset = []
	
	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_line]:
			current_word = word_tokenize(l.lower())
			current_word = [lemmatizer.lemmatize(i) for i in current_word]
			feature = np.zeros(len(lexicon))
			for words in current_word:
				if words.lower() in lexicon:
					index_value = lexicon.index(words.lower())
					feature[index_value] += 1
					
				feature = list(feature)
				featureset.append([feature, classification])
	return featureset

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features)
	
	features = np.array(features)
	testing_size = int(test_size*len(features))
	
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	
	return train_x,train_y,test_x,test_y
	
if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

print(len(train_x))
x = tf.placeholder('float',[None,len(train_x)])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x),n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	#(input * weights) + biases
	
	l1 = tf.add(tf.matmul(data , hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1 , hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2 , hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.matmul(l3 , output_layer['weights']) + output_layer['biases']
	
	return output
	
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate = 0.001
	
	hm_epochs = 10
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x,y: batch_y})
				epoch_loss += c
				i += batch_size
				
			print('Epoch', epoch+1, 'Completed out of ',hm_epochs,'loss :',epoch_loss )
			
		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy',accuracy.eval({x : test_x, y : test_y}))
			
train_neural_network(x)