import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import OutputSentence, TopN
import utilities
	

class RNNModel:


	def __init__(self, buckets_dict, mode='training'):
		print "========== INIT ============= "
		self.use_reverse_encoder = False
		if mode=='training':
			self.token_lookup_sequences_decoder_placeholder_list = []
			self.masker_list = []
			self.token_output_sequences_decoder_placeholder_list = []
			self.token_lookup_sequences_placeholder_list = []

			for bucket_num, bucket in buckets_dict.items():
				max_sentence_length = bucket['max_input_seq_length']
				self.token_lookup_sequences_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_lookup_sequences"+str(bucket_num))  )# token_lookup_sequences
				
				max_sentence_length = bucket['max_output_seq_length']
				self.masker_list.append( tf.placeholder("float32", [None, max_sentence_length], name="masker"+str(bucket_num)) )
				self.token_output_sequences_decoder_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_output_sequences_decoder_placeholder"+str(bucket_num)) )
				self.token_lookup_sequences_decoder_placeholder_list.append( tf.placeholder("int32", [None, max_sentence_length], name="token_lookup_sequences_decoder_placeholder"+str(bucket_num)) ) # token_lookup_sequences
		print "========== INIT OVER ============= "


	def _getEncoderInitialState(self,cell, batch_size):
		return cell.zero_state(batch_size, tf.float32)

	def runDecoderStep(self, lstm_cell, cur_inputs, prev_cell_output, state, encoder_outputs, reuse=False):
		print "cur_inputs = ",cur_inputs
		inputs = tf.concat([ cur_inputs, context ], axis=1)
		print "inputs = ",inputs
		print "conterxt = ",context
		return lstm_cell(inputs, state=state), context, alpha


	def initDecoderOutputVariables(self,lstm_cell_size, token_vocab_size):
		with tf.variable_scope('decoder_output', reuse=None) as scope:
			w_out = tf.get_variable('w_out', shape=[lstm_cell_size, token_vocab_size], initializer=tf.random_normal_initializer(-1.0,1.0))
			w_out = tf.get_variable('b_out', shape=[token_vocab_size]) # , initializer=tf.random_normal())
			w_context_out = tf.get_variable('w_context_out', shape=[lstm_cell_size, token_vocab_size], initializer=tf.random_normal_initializer(-1.0,1.0))
			b_context_out = tf.get_variable('b_context_out', initializer=tf.random_normal([token_vocab_size]) )
			scope.reuse_variables()

	def getDecoderOutputVariables(self):
		with tf.variable_scope('decoder_output', reuse=True) as scope:
			w_out = tf.get_variable('w_out')
			b_out = tf.get_variable('b_out')
			w_context_out = tf.get_variable('w_context_out')
			b_context_out = tf.get_variable('b_context_out')
			return w_out, b_out, w_context_out, b_context_out

	def getDecoderOutput(self, output, lstm_cell_size, token_vocab_size, w_out, b_out, w_context_out, b_context_out, context=None ): # outputs_list: list of tensor(batch_size, cell_size) with time_steps number of items
		pred = tf.matmul(output, w_out) #+ b_out  #(N,vocab_size)
		pred += (tf.matmul(context, w_context_out) + b_context_out)  #(N,vocab_size)
		return pred

	def initEmbeddings(self, emb_scope, token_vocab_size, embeddings_dim, reuse=False, pretrained_embeddings=None):
		with tf.variable_scope(emb_scope):
			if pretrained_embeddings!=None:
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float', initializer=tf.constant_initializer(np.array(pretrained_embeddings)) )
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
			else:
				token_emb_mat = tf.get_variable("emb_mat", shape=[token_vocab_size, embeddings_dim], dtype='float')
				# 0-mask
				token_emb_mat = tf.concat( [tf.zeros([1, embeddings_dim]), tf.slice(token_emb_mat, [1,0],[-1,-1]) ], axis=0 )	
				#print "token_emb_mat = ",token_emb_mat
		return token_emb_mat

	def greedyInferenceModel(self, params ):
		lstm_cell = params['lstm_cell']
		encoder_outputs = params['encoder_outputs']
		token_vocab_size = params['vocab_size']
		lstm_cell_size = params['lstm_cell_size']
		batch_size = params['batch_size']
		embeddings_dim = params['embeddings_dim']
		batch_time_steps = params['max_output_seq_length']
		token_emb_mat = params['token_emb_mat']
		w_out, b_out, w_context_out, b_context_out = params['output_vars']
		encoder_outputs = params['encoder_outputs']
		cell_output, state = params['cell_state']

		num_steps = batch_time_steps
		outputs = []

		for time_step in range(num_steps):
			if time_step==0:
				inp = tf.ones([batch_size,1], dtype=tf.int32) #start symbol index  #TO DO: get start index from config
				#outputs.append( tf.reshape(inp,[batch_size]) )
			inputs_current_time_step = tf.reshape( tf.nn.embedding_lookup(token_emb_mat, inp) , [-1, embeddings_dim] )
			if time_step > 0: tf.get_variable_scope().reuse_variables()
			(cell_output, state), context, alpha = self.runDecoderStep(lstm_cell=lstm_cell, cur_inputs=inputs_current_time_step, encoder_outputs=encoder_outputs, prev_cell_output=cell_output, reuse=(time_step!=0), state=state)
			# cell_output: (N,cell_size)
			cur_outputs = self.getDecoderOutput(cell_output, lstm_cell_size, token_vocab_size, w_out, b_out, w_context_out, b_context_out, context)
			assert cur_outputs.shape[1]==token_vocab_size
			word_predictions = tf.argmax(cur_outputs,axis=1)
			outputs.append(word_predictions)
			inp = word_predictions
		return outputs


	def decoderRNN(self, x, params, reuse=False, mode='training'):

		lstm_cell = params['lstm_cell']
		encoder_outputs = params['encoder_outputs']
		token_vocab_size = params['vocab_size']
		lstm_cell_size = params['lstm_cell_size']
		batch_size = params['batch_size']
		embeddings_dim = params['embeddings_dim']
		batch_time_steps = params['max_output_seq_length']
		if 'token_emb_mat' in params:
			token_emb_mat = params['token_emb_mat']
		else:
			token_emb_mat = None

		#with tf.variable_scope('decoder'):
		num_steps = None
		if batch_time_steps:
			num_steps = batch_time_steps #should be equal to x.shape[1]
		else:
			#print "x.shape: ",x.shape
			num_steps = x.shape[1]
		inputs = x

		self.decoder_cell = cell = lstm_cell

		# inital state
		decoder_initial_state = self.getInitialState(encoder_outputs, lstm_cell_size, reuse=reuse)

		#decoder output variable
		self.initDecoderOutputVariables(lstm_cell_size,token_vocab_size)
		w_out, b_out, w_context_out, b_context_out = self.getDecoderOutputVariables()

		#unrolled lstm 
		outputs = [] # h values at each time step
		state = decoder_initial_state
		cell_output = state[1]
		encoder_outputs = tf.stack(encoder_outputs) # timesteps, N, cellsize
		encoder_outputs = tf.transpose(encoder_outputs,[1,0,2]) # N, timesteps, cellsize 
		with tf.variable_scope("RNN"):
			if mode=='training':
				pred = []
				for time_step in range(num_steps):
					if time_step > 0: tf.get_variable_scope().reuse_variables()
					inputs_current_time_step = inputs[:, time_step, :]
					#print "inputs_current_time_step.shape: ",inputs_current_time_step.shape
					print "state = ",state
					(cell_output, state), context, alpha = self.runDecoderStep(lstm_cell=lstm_cell, cur_inputs=inputs_current_time_step, encoder_outputs=encoder_outputs, prev_cell_output=cell_output, reuse=(time_step!=0), state=state)
					#print(cell_output.shape)
					outputs.append(cell_output)
					cur_pred = self.getDecoderOutput(cell_output, lstm_cell_size, token_vocab_size, w_out, b_out, w_context_out, b_context_out, context)
					pred.append(cur_pred)
				pred = tf.stack(pred)
				outputs_tensor = tf.stack(outputs) 
				outputs = tf.unstack(outputs_tensor)
				#pred = self.getDecoderOutput(outputs, lstm_cell_size, token_vocab_size, w_out, b_out, w_context_out, b_context_out, context)
				#print "--->>>>>> ",type(pred)
				#print pred[0].shape
				tf.get_variable_scope().reuse_variables()

			elif mode=='inference':

				#Greedy
				params['output_vars'] = w_out, b_out, w_context_out, b_context_out
				params['cell_output'] = cell_output
				params['encoder_outputs'] = encoder_outputs
				params['cell_state'] = cell_output, state
				params['beam_size'] = 20
				outputs =  self.greedyInferenceModel(params) #self.beamSearchInference(params)  #self.greedyInferenceModel(params)
				ret_encoder_outputs = tf.transpose(encoder_outputs,[1,0,2]) # N, timesteps, cellsize 
				pred = outputs, ret_encoder_outputs

				#Beam search
		return pred


	#################################################################################################################


	def getDecoderModel(self, config, encoder_outputs, is_training=False, mode='training', reuse=False, bucket_num=0 ):

		print "==========================================================="
		if mode=='inference' and is_training:
			print "ERROR. INCONSISTENT PARAMETERS"
		assert mode=='inference' or mode=='training'
		print " IN DECODER MODEL :: ",encoder_outputs[0].shape

		token_vocab_size = config['vocab_size']
		max_sentence_length = config['max_output_seq_length']
		embeddings_dim = config['embeddings_dim']
		lstm_cell_size = config['lstm_cell_size']

		#placeholders
		if mode=='training':
			token_lookup_sequences_decoder_placeholder =self.token_lookup_sequences_decoder_placeholder_list[bucket_num]
			masker = self.masker_list[bucket_num]
			token_output_sequences_placeholder = self.token_output_sequences_decoder_placeholder_list[bucket_num]

		#embeddings
		share_embeddings=False
		emb_scope = 'emb_decoder'
		if share_embeddings:
			emb_scope='emb'
		if reuse:
			token_emb_mat = self.decoder_token_emb_mat
		else:
			pretrained_embeddings=None
			if config['pretrained_embeddings']:
				pretrained_embeddings = config['decoder_embeddings_matrix']
			self.decoder_token_emb_mat = token_emb_mat = self.initEmbeddings(emb_scope, token_vocab_size, embeddings_dim, reuse=reuse, pretrained_embeddings=pretrained_embeddings)

		with tf.variable_scope('decoder',reuse=reuse):
				
			# lstm 
			lstm_cell = rnn.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, state_is_tuple=True, reuse=reuse)

			if mode=='inference':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				params['encoder_outputs'] = encoder_outputs
				params['token_emb_mat'] = token_emb_mat
				inp= None #tf.nn.embedding_lookup(token_emb_mat, token_lookup_sequences_decoder_placeholder)
				pred = self.decoderRNN(inp, params, mode='inference')
			elif mode=='training':
				params={k:v for k,v in config.items()}
				params['lstm_cell'] = lstm_cell 
				params['encoder_outputs'] = encoder_outputs
				#params['token_emb_mat'] = None
				inp = tf.nn.embedding_lookup(token_emb_mat, token_lookup_sequences_decoder_placeholder) 
				pred = self.decoderRNN(inp, params, mode='training')  # timesteps, N, vocab_size
				pred_for_loss = pred # since sparse_softmax_cross_entropy_with_logits takes softmax on its own as well
				pred = tf.unstack(pred)
				pred = tf.stack( [ tf.nn.softmax(vals) for vals in pred ] )

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

	###################################################################################
