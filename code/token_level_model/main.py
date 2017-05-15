import matplotlib.pyplot as plt
from nltk import word_tokenize
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import pickle
import utilities as datasets
import utilities
import solver
import tensorflow as tf

# Set seed for reproducability
tf.set_random_seed(1)


debug_mode = True
all_lengths = []

class PreProcessing:

	def __init__(self, params):
		self.unknown_word = "UNKNOWN" # not used
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()
		self.word_index = {}
		self.index_word = {} # populated at end of all the processing
		self.word_index_counter = 0

		self.word_index[self.pad_word] = self.word_index_counter
		self.word_index_counter+=1
		self.word_index[self.sent_start] = self.word_index_counter
		self.word_index_counter+=1
		self.word_index[self.sent_end] = self.word_index_counter
		self.word_index_counter+=1

		self.params = params

	def pad_sequences_my(self, sequences, maxlen, padding='post', truncating='post'):
		ret=[]
		# truncating 'pre' not supported as of now
		pad_word_idx = self.word_index[self.pad_word]
		for sequence in sequences:
			if len(sequence)>=maxlen:
				sequence=sequence[:maxlen]
			else:
				if padding=='post':
					sequence = sequence + [pad_word_idx]*(maxlen - len(sequence))
				else:
					sequence = [pad_word_idx]*(maxlen - len(sequence)) + sequence
			ret.append(sequence)
		return np.array(ret)

	
	def fromIdxSeqToVocabSeq(self, seq):
		return [self.index_word[x] for x in seq]

	def tokenizer(self, texts):
		global all_lengths
		ret = [ word_tokenize(text.lower()) for text in texts ]
		for text in ret:
			for token in text:
				if token not in self.word_index:
					self.word_index[token] = self.word_index_counter
					self.word_index_counter+=1
			all_lengths.append(len(text))
		ret = [ [ self.word_index[t] for t in text ] for text in ret ]
		return ret

	def char_tokenizer(self, texts, special_tokens={'<unk>':'U'}):
		global all_lengths
		print texts[7]
		for k,v in special_tokens.items():
			texts = [  text.replace(k,v) for text in texts ]
		print texts[7]
		for text in texts:
			for ch in text:
				if ch not in self.word_index:
					self.word_index[ch] = self.word_index_counter
					self.word_index_counter+=1
			all_lengths.append(len(text))
		ret = [ [ self.word_index[t] for t in text ] for text in texts ]
		return ret

	def loadData(self, split='train', char_or_word="word"):   
		print "-----------------loadData()--------- split= ",split
		data_src = config.data_src + "." + split + ".txt"
		texts=[]
		if debug_mode:
			with open(data_src,"r") as fr:
				ctr=0
				for line in fr:
					texts.append(line)
					ctr+=1
					if ctr>200:
						break
		else:
			texts = open(data_src,"r").readlines()
		if char_or_word=="word":
			texts = [self.sent_start + " " + text + " " + self.sent_end for text in texts]
			sequences = self.tokenizer(texts)
			word_index = self.word_index
			#TO DO: add bucketing...
		else: # character
			sequences = self.char_tokenizer(texts)
			word_index = self.word_index			
			sequences = [ [self.word_index[self.sent_start]] + sequence + [self.word_index[self.sent_end]] for sequence in sequences]
			#TO DO: add bucketing...
		sequences = self.pad_sequences_my(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)	
		#word_index[self.unknown_word]=0
		print sequences[0]
		print texts[0]
		print "-----------------Done loadData()---------"
		return sequences


	def prepareLMdata(self, sequences):
		data = np.array( [ sequence[:-1] for sequence in sequences ] )
		if self.params['use_tf']:
			labels = np.array( [sequence[1:] for sequence in sequences ] )
		else:
			labels = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in sequences ] )

		if debug_mode:
			data = data[:192]
			labels = labels[:192]
		else:
			batch_sz = config.batch_size
			lim = batch_sz * (  len(data)/batch_sz ) # TO DO: Add support for incomplete batch in the model or in solver
			data = data[:lim]
			labels = labels[:lim]
		return data,labels

def getPretrainedEmbeddings(src):
	ret={}
	vals = open(src,"r").readlines()
	vals = [val.strip().split('\t') for val in vals]
	for val in vals:
		#print val[0]
		ret[val[0]] = [float(v) for v in val[1:]]
		assert len( ret[val[0]] ) == 50
	ret['sentstart'] = ret['SENT_START']
	ret['sentend'] = ret['SENT_END']
	return ret


def main():

	# buckets
	buckets = {  0:{'max_input_seq_length': config.MAX_SEQUENCE_LENGTH-1 } } # 40-1=39
	print buckets
	print "==================================================="

	# get model
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['max_input_seq_length'] = config.MAX_SEQUENCE_LENGTH - 1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = config.batch_size
	params['pretrained_embeddings']=False
	params['char_or_word'] = config.char_or_word
	params['use_tf'] = True
	params['keep_prob'] = 1.0 - config.dropout_val
	print params

	preprocessing = PreProcessing(params)

	train_sequences = preprocessing.loadData(split='train', char_or_word=params['char_or_word'] )		
	val_sequences = preprocessing.loadData(split='valid', char_or_word=params['char_or_word'] )
	test_sequences = preprocessing.loadData(split='test', char_or_word=params['char_or_word'] )	
	print preprocessing.word_index[preprocessing.sent_start]
	print preprocessing.word_index[preprocessing.sent_end]
	index_word = {i:w for w,i in preprocessing.word_index.items()}
	preprocessing.index_word = index_word
	train = preprocessing.prepareLMdata(train_sequences)
	val = preprocessing.prepareLMdata(val_sequences)
	test = preprocessing.prepareLMdata(test_sequences)
	#train_weights = preprocessing.loadTeacherProbValuesForTrain() # the function will pick the default path from config
	
	trainx, trainy = train

	# seq. length analysis
	global all_lengths
	all_lengths = np.array(all_lengths)
	import scipy.stats
	print " *** ", scipy.stats.describe(all_lengths)

	params['vocab_size'] =  preprocessing.word_index_counter
	print "params['vocab_size'] --------> ",params['vocab_size']

	data,labels = train
	print "--------- SAMPLE & shape..."
	print "train data.shape ",data.shape
	print "train labels.shape ",labels.shape
	print "sameple:", data[0]
	print "sample outputs: ",labels[0]
	print "-------------------"
	
	if params['use_tf']:
		# model
		mode=  ["inference", "train", "sample"][2]
		print "mode = ",mode
		if mode=='train':
			train_buckets = {}
			train = data, labels
			for bucket,_ in enumerate(buckets):
				train_buckets[bucket] = train

			rnn_model = solver.Solver(buckets)
			_ = rnn_model.getModel(params, mode='train',reuse=False, buckets=buckets)
			rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=val, reverse_vocab=preprocessing.index_word, do_init=True)
		
		else: # inference, sample
			rnn_model = solver.Solver(buckets=None, mode='inference')
			params['max_inp_seq_length'] = 39
			params['saved_model_path'] = config.saved_model_path
			_ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)

			if mode == 'inference':
				data_typ=["train","val"][1]
				if data_typ=="val":
					valx, valy = val
				else:
					valx, valy = train
				dump_seq_prob=True
				dump_seq_prob_path="./tmp/" + data_typ + "_groundtruth_probs.txt"
				rnn_model.solveAll(params, valx, valy, preprocessing.index_word, sess=None, dump_seq_prob=dump_seq_prob, dump_seq_prob_path=dump_seq_prob_path)
			else:
				sample_output_path="./tmp/samples.txt"
				rnn_model.sample(params, preprocessing.index_word, sample_output_path, None)

	else: #Keras
		print "KERAS MODEL"

		params['inp_length'] = 	params['max_input_seq_length']
		rnn_model = keras_models.RNNModel()
		model = rnn_model.getModel(params)
		
		#x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test
		x_train, y_train = train
		x_val, y_val = val
		x_test, y_test = test

		#call backs
		perplexity_callback = keras_models.PerplexityCalculator()
		weight_save_callback = ModelCheckpoint( 'tmp/keras_checkpoints/' + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=0, save_best_only=False, mode='auto')

		# train
		keras_models.validation_data = (x_val, y_val)
		model.fit(x_train, y_train, validation_data=(x_val, y_val),
				  nb_epoch=3, batch_size= params['batch_size'], callbacks=[weight_save_callback, perplexity_callback] ) #config.num_epochs
		#saveEmbeddings(model, preprocessing.word_index)
				  
		#evaluate
		scores = model.evaluate(x_test, y_test, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))

		print "--- Sampling few sequences.. "
		for i in range(5):
			print ""
			pred = utilities.generateSentence(model, preprocessing.word_index, preprocessing.sent_start, 
				preprocessing.sent_end, preprocessing.unknown_word)
			sent = [preprocessing.index_word[i] for i in pred]
			if params['char_or_word'] == "char":
				print ''.join(sent)
			else:
				print ' '.join(sent)


if __name__ == "__main__":
	main()
