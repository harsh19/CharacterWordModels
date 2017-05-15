import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import OutputSentence, TopN
import utilities
	

class RNNModel:


	def __init__(self, buckets_dict, mode='training', max_input_seq_length_inference=39):
		print "========== INIT ============= "
		if mode=='training':
			self.token_input_sequences_placeholder_list = []
			self.masker_list = []
			self.token_output_sequences_placeholder_list = []

			for bucket_num, bucket in buckets_dict.items():
				max_input_seq_length = bucket['max_input_seq_length']
				self.token_input_sequences_placeholder_list.append( tf.placeholder("int32", [None, max_input_seq_length], name="token_input_sequences"+str(bucket_num))  )# token_lookup_sequences
				self.masker_list.append( tf.placeholder("float32", [None, max_input_seq_length], name="masker"+str(bucket_num)) )
				self.token_output_sequences_placeholder_list.append( tf.placeholder("int32", [None, max_input_seq_length], name="token_output_sequences_placeholder"+str(bucket_num)) )
		else:
			self.token_input_sequences_placeholder_inference = tf.placeholder("int32", [None, max_input_seq_length_inference], name="token_input_sequences_inference")		
		print "========== INIT OVER ============= "


	def _getInitialState(self,cell, batch_size):
		return cell.zero_state(batch_size, tf.float32)

	def _runDecoderStep(self, lstm_cell, cur_inputs, state, reuse=False):
		inputs = cur_inputs
		return lstm_cell(inputs, state=state)

	def _initDecoderOutputVariables(self,lstm_cell_size, token_vocab_size):
		with tf.variable_scope('decoder_output', reuse=None) as scope:
			w_out = tf.get_variable('w_out', shape=[lstm_cell_size, token_vocab_size], initializer=tf.random_normal_initializer(-1.0,1.0))
			b_out = tf.get_variable('b_out', shape=[token_vocab_size]) # , initializer=tf.random_normal())
			scope.reuse_variables()

	def _getDecoderOutputVariables(self):
		with tf.variable_scope('decoder_output', reuse=True) as scope:
			w_out = tf.get_variable('w_out')
			b_out = tf.get_variable('b_out')
			return w_out, b_out

	def _getDecoderOutput(self, cell_output, w_out, b_out ): # outputs_list: list of tensor(batch_size, cell_size) with time_steps number of items
		pred = tf.matmul(cell_output, w_out) + b_out  #(N,vocab_size)
		return pred

	def _initEmbeddings(self, emb_scope, token_vocab_size, embeddings_dim, reuse=False, pretrained_embeddings=None):
		with tf.variable_scope(emb_scope):
			if pretrained_embeddings!=None:
				print " [LOG]: Using pretrained embeddings ... "
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float', initializer=tf.constant_initializer(np.array(pretrained_embeddings)) )
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
			else:
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float')
				# 0-mask
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
		return token_emb_mat

	def _decoderRNN(self, x, params, reuse=False, mode='training', keep_prob=1.0):

		lstm_cell = params['lstm_cell']
		token_vocab_size = params['vocab_size']
		lstm_cell_size = params['lstm_cell_size']
		batch_size = params['batch_size']
		embeddings_dim = params['embeddings_dim']
		batch_time_steps = params['max_inp_seq_length']
		if 'token_emb_mat' in params:
			token_emb_mat = params['token_emb_mat']
		else:
			token_emb_mat = None

		num_steps = batch_time_steps #should be equal to x.shape[1]
		inputs = x

		self.lstm_cell = cell = lstm_cell

		# inital state
		initial_state = self._getInitialState(cell, batch_size)

		#decoder output variable
		self._initDecoderOutputVariables(lstm_cell_size, token_vocab_size)
		w_out, b_out = self._getDecoderOutputVariables()
		self.w_out = w_out
		self.b_out = b_out

		#unrolled lstm 
		outputs = [] # h variablealu		emb_scope appendt each time step
		state = initial_state
		cell_output = state[1]
		#print "mode, keep_prob = ", mode, keep_prob
		with tf.variable_scope("RNN"):
			pred = []
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				inputs_current_time_step = inputs[:, time_step, :]
				inputs_current_time_step = tf.nn.dropout(inputs_current_time_step, keep_prob)
				#print "state = ",state
				(cell_output, state) = self._runDecoderStep(lstm_cell=lstm_cell, cur_inputs=inputs_current_time_step, reuse=(time_step!=0), state=state)
				outputs.append(cell_output)
				cur_pred = self._getDecoderOutput(cell_output, w_out, b_out)
				pred.append(cur_pred)
			pred = tf.stack(pred)
			tf.get_variable_scope().reuse_variables()
		return pred


	#################################################################################################################

	def getSampler(self, config, is_first_step=True, reuse=True):
		if is_first_step:
			batch_size = config['batch_size']
			cell = self.lstm_cell
			init_state = self._getInitialState(cell, batch_size )
			return init_state
		else:
			batch_size = config['batch_size']
			cell_size = config['lstm_cell_size'] # should be same as used during model building
			cell = self.lstm_cell
			token_emb_mat = self.decoder_token_emb_mat
			token_input_sequences_placeholder_sampler = tf.placeholder("int32", [None, 1], name="token_input_sequences_placeholder_sampler")		
			c_prev = tf.placeholder("float32", [None, cell_size], name="c_prev")		
			h_prev = tf.placeholder("float32", [None, cell_size], name="h_prev")
			inp = tf.nn.embedding_lookup(token_emb_mat, token_input_sequences_placeholder_sampler)  # None, 1, embedding_dim
			w_out, b_out = self.w_out, self.b_out
			## inp = tf.squeeze( inp, axis=[1] ) # N?one, embedding_dim
			inp = inp[:, 0, :] # None, embedding_dim
			print "inp = ",inp
			scope_name = "decoder/RNN"
			state = (c_prev, h_prev)
			with tf.variable_scope(scope_name,reuse=reuse):
				(h_next, c_next) = self._runDecoderStep(lstm_cell=cell, cur_inputs=inp, state=state)
			cur_pred = self._getDecoderOutput(h_next, w_out, b_out) # None, vocab_size
			cur_pred = tf.nn.softmax( cur_pred )
			return c_prev, h_prev, token_input_sequences_placeholder_sampler, c_next, h_next, cur_pred


	def getDecoderModel(self, config, is_training=False, mode='training', reuse=False, bucket_num=0 ):

		print "==========================================================="

		token_vocab_size = config['vocab_size']
		max_sentence_length = config['max_inp_seq_length']
		embeddings_dim = config['embeddings_dim']
		lstm_cell_size = config['lstm_cell_size']
		keep_prob = config['keep_prob']

		#placeholders
		if mode=='training':
			token_input_sequences_placeholder = self.token_input_sequences_placeholder_list[bucket_num]
			masker = self.masker_list[bucket_num]
			token_output_sequences_placeholder = self.token_output_sequences_placeholder_list[bucket_num]
		else:
			token_input_sequences_placeholder = self.token_input_sequences_placeholder_inference

		#embeddings
		share_embeddings=False
		emb_scope='emb'
		if reuse:
			token_emb_mat = self.decoder_token_emb_mat
		else:
			pretrained_embeddings=None
			if config['pretrained_embeddings']:
				pretrained_embeddings = config['decoder_embeddings_matrix']
			self.decoder_token_emb_mat = token_emb_mat = self._initEmbeddings(emb_scope, token_vocab_size, embeddings_dim, reuse=reuse, pretrained_embeddings=pretrained_embeddings)

		with tf.variable_scope('decoder',reuse=reuse):
				
			# lstm 
			lstm_cell = rnn.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)

			if mode=='inference':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				inp = tf.nn.embedding_lookup(token_emb_mat, token_input_sequences_placeholder) 
				pred = self._decoderRNN(inp, params, mode='inference', keep_prob=keep_prob)  # timesteps, N, vocab_size
				pred = tf.unstack(pred)
				pred = tf.stack( [ tf.nn.softmax(vals) for vals in pred ] ) # timesteps, N, vocab_size
				return pred
			elif mode=='training':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				inp = tf.nn.embedding_lookup(token_emb_mat, token_input_sequences_placeholder) 
				pred = self._decoderRNN(inp, params, mode='training', keep_prob=keep_prob)  # timesteps, N, vocab_size
				pred_for_loss = pred # since sparse_softmax_cross_entropy_with_logits takes softmax on its own as well
				pred = tf.unstack(pred)
				pred = tf.stack( [ tf.nn.softmax(vals) for vals in pred ] ) # timesteps, N, vocab_size

				if is_training:
					pred_masked = pred_for_loss 
					#tf.multiply( tf.expand_dims(tf.transpose(masker),2), pred)  # after transpose and expand, masker becomes (4,20,1) : timesteps,N,1. so pred_masked is timesteps,N,vocab_size
					
					#print "pred_masked .shape : ",pred_masked.shape
					pred_masked = tf.transpose( pred_masked , [1,0,2] ) # N, timesteps, vocabsize
					cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_masked, labels=token_output_sequences_placeholder) # token_output_sequences_placeholder is N,timesteps. cost will be N, timesteps
					cost = tf.multiply(cost, masker)  # both masker and cost is N,timesteps. 

					#masker = tf.reshape(masks, (-1))
					#cost = losses * masks
					#print "cost.shape: " ,cost.shape
					cost = tf.reduce_sum(cost) # N
					masker_sum = tf.reduce_sum(masker) # N
					cost = tf.divide(cost, masker_sum) # N
					self.cost = cost

				return pred #[ tf.nn.softmax(vals) for vals in pred]
			else:
				print "ERROR: UNSUPPORTED MODE TYPE"

	###################################################################################
