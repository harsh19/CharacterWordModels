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

debug_mode = True

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
		self.word_index[self.sent_start] = self.word_index_counter
		self.word_index_counter+=1
		self.word_index[self.sent_end] = self.word_index_counter
		self.word_index_counter+=1

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
		ret = [ word_tokenize(text.lower()) for text in texts ]
		for text in ret:
			for token in text:
				if token not in self.word_index:
					self.word_index[token] = self.word_index_counter
					self.word_index_counter+=1
		ret = [ [ self.word_index[t] for t in text ] for text in ret ]
		return ret

	def loadData(self, split='train'):   
		print "loading " ,split," data..."
		data_src = config.data_src + "." + split + ".txt"
		texts = open(data_src,"r").readlines()
		texts = [self.sent_start + " " + text + " " + self.sent_end for text in texts]
		
		sequences = self.tokenizer(texts)

		word_index = self.word_index
		#TO DO: add bucketing...
		sequences = self.pad_sequences_my(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
		
		a=sequences
		word_index[self.unknown_word]=0
		print sequences[0]
		return sequences

	def prepareLMdata(self, sequences):
		data = np.array( [ sequence[:-1] for sequence in sequences ] )
		labels = np.array( [sequence[1:] for sequence in sequences ] )
		if debug_mode:
			data = data[:200], labels[:200]
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
	preprocessing = PreProcessing()

	train_sequences = preprocessing.loadData(split='train')		
	val_sequences = preprocessing.loadData(split='valid')		
	test_sequences = preprocessing.loadData(split='test')		
	print preprocessing.word_index[preprocessing.sent_start]
	print preprocessing.word_index[preprocessing.sent_end]
	index_word = {i:w for w,i in preprocessing.word_index.items()}
	preprocessing.index_word = index_word
	train = preprocessing.prepareLMdata(train_sequences)
	val = preprocessing.prepareLMdata(val_sequences)
	test = preprocessing.prepareLMdata(test_sequences)
	#return
	
	# get model
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['vocab_size'] =  preprocessing.word_index_counter
	print "params['vocab_size'] --------> ",params['vocab_size']
	params['max_input_seq_length'] = config.MAX_SEQUENCE_LENGTH - 1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = 20
	params['pretrained_embeddings']=False
	
	#return
	print params
	buckets = {  0:{'max_input_seq_length':40, 'max_output_seq_length':19} }
	#,1:{'max_input_seq_length':40,'max_output_seq_length':19}, 2:{'max_input_seq_length':40, 'max_output_seq_length':19} }
	print buckets

	return
	
	# model
	mode='train'
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
