import matplotlib.pyplot as plt
from nltk import word_tokenize
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import models
import pickle
import utilities as datasets
import utilities
import solver

class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNKNOWN" # not used
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()
		self.word_index = {}
		self.index_word = {} # populated at end of all the processing
		self.word_index_counter = 0

		self.word_index[self.pad_word] = self.word_index_counter
		self.word_index_counter+=1
		self.word_index[self.sentstart] = self.word_index_counter
		self.word_index_counter+=1
		self.word_index[self.sentend] = self.word_index_counter
		self.word_index_counter+=1

	def pad_sequences_my(sequences, maxlen, padding='post', truncating='post'):
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

	def tokenzier(self, texts):
		ret = [ word_tokenize(text.lower()) for text in texts ]
		for text in ret:
			for token in text:
				if token not in self.word_index:
					self.word_index[token] = self.word_index_counter
					self.word_index_counter+=1
		return ret

	def loadData(self, split='train'):   
		print "loading " ,split," data..."
		data_src = config.data_src + "." + split + ".txt"
		texts = open(data_src,"r").readlines()
		texts = [self.sent_start + " " + text + " " + self.sent_end for text in texts]
		
		sequences = tokenizer(texts)

		word_index = self.word_index
		#TO DO: add bucketing...
		sequences = pad_sequences_my(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
		
		a=sequences
		word_index[self.unknown_word]=0
		print sequences[0]
		return sequences

	def prepareLMdata(self,seed=123):

		data = np.array( [ sequence[:-1] for sequence in self.sequences ] )
		labels = np.array( [sequence[1:] for sequence in self.sequences ] )

		# Shuffle
		indices = np.arange(data.shape[0])
		np.random.seed(seed)
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]

		# Splits. TO DO: Shuffle once and save the splits
		nb_validation_samples = int(config.VALIDATION_SPLIT * data.shape[0])
		nb_test_samples = int(config.TEST_SPLIT * data.shape[0])
		print "nb_test_samples=",nb_test_samples
		self.x_train = data[0:-nb_test_samples-nb_validation_samples]
		self.y_train = labels[0:-nb_test_samples-nb_validation_samples]
		self.x_val = data[-nb_test_samples-nb_validation_samples:-nb_test_samples]
		self.y_val = labels[-nb_test_samples-nb_validation_samples:-nb_test_samples]
		self.x_test = data[-nb_test_samples:]
		self.y_test = labels[-nb_test_samples:]
		print self.x_train.shape, " ", self.y_train.shape
		print self.x_val.shape
		print self.x_test.shape

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
	preprocessing = PreProcessing()

	train_sequences = preprocessing.loadData(split='train')		
	val_sequences = preprocessing.loadData(split='val')		
	test_sequences = preprocessing.loadData(split='test')		
	print self.word_index[self.sentstart]
	print self.word_index[self.sentend]
	index_word = {i:w for w,i in word_index.items()}
	self.index_word = index_word
	train = preprocessing.prepareLMdata(train_sequences)
	val = preprocessing.prepareLMdata(val_sequences)
	test = preprocessing.prepareLMdata(test_sequences)
	#return
	
	# get model
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['vocab_size'] =  preprocessing.vocab_size
	print "params['vocab_size'] --------> ",params['vocab_size']
	params['max_input_seq_length'] = config.MAX_SEQUENCE_LENGTH - 1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = 20
	params['pretrained_embeddings']=False
	
	#return
	print params
	buckets = {  0:{'max_input_seq_length':40, 'max_output_seq_length':19} }
	#,1:{'max_input_seq_length':40,'max_output_seq_length':19}, 2:{'max_input_seq_length':40, 'max_output_seq_length':19} }
	print buckets
	
	# train
	lim=params['batch_size'] * ( len(train[0])/params['batch_size'] )
	if lim!=-1:
		train_encoder_inputs, train_decoder_inputs, train_decoder_outputs = train
		train_encoder_inputs = train_encoder_inputs[:lim]
		train_decoder_inputs = train_decoder_inputs[:lim]
		train_decoder_outputs = train_decoder_outputs[:lim]
		train = train_encoder_inputs, train_decoder_inputs, train_decoder_outputs
	if params['pretrained_embeddings']:
		pretrained_embeddings = getPretrainedEmbeddings(src="pretrained_embeddings.txt")
		char_to_idx = preprocessing.word_index
		encoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		decoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		for char,idx in char_to_idx.items():
			if char in pretrained_embeddings:
				encoder_embedding_matrix[idx]=pretrained_embeddings[char]
				decoder_embedding_matrix[idx]=pretrained_embeddings[char]
			else:
				print "No pretrained embedding for ",char
		params['encoder_embeddings_matrix'] = encoder_embedding_matrix 
		params['decoder_embeddings_matrix'] = decoder_embedding_matrix 


	# TO DO.. Below code is for MT.. change it for LM
	mode=''
	if mode=='train':
		train_buckets = {}
		for bucket,_ in enumerate(buckets):
			train_buckets[bucket] = train

		rnn_model = solver.Solver(buckets)
		_ = rnn_model.getModel(params, mode='train',reuse=False, buckets=buckets)
		rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=None, reverse_vocab=preprocessing.index_word, do_init=True)
	
	else:
		val_encoder_inputs, val_decoder_inputs, val_decoder_outputs = val
		print "val_encoder_inputs = ",val_encoder_inputs

		if len(val_decoder_outputs.shape)==3:
			val_decoder_outputs=np.reshape(val_decoder_outputs, (val_decoder_outputs.shape[0], val_decoder_outputs.shape[1]))

		rnn_model = solver.Solver(buckets=None, mode='inference')
		_ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
		print "----Running inference-----"
		#rnn_model.runInference(params, val_encoder_inputs[:params['batch_size']], val_decoder_outputs[:params['batch_size']], preprocessing.index_word)
		rnn_model.solveAll(params, val_encoder_inputs, val_decoder_outputs, preprocessing.index_word)

if __name__ == "__main__":
	main()
