import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import OutputSentence, TopN
import utilities
import utilities as utils
import model
	
class Solver:

	def __init__(self, buckets=None, mode='training'):
		if mode=='training':
			self.model_obj = model.RNNModel(buckets, mode=mode)
		else:
			self.model_obj = model.RNNModel(buckets_dict=None, mode=mode)

	def getModel(self, config, buckets, mode='train', reuse=False ):

		self.buckets = buckets 
		self.preds = []
		self.cost_list = []
		self.mask_list = []
		self.token_input_sequences_placeholder_list = []
		self.token_output_sequences_placeholder_list = []

		if mode=='train':
			#########################
			print "==================================================="
			for bucket_num, bucket_dct in self.buckets.items():
				config['max_inp_seq_length'] = bucket_dct['max_input_seq_length']
				print ""
				print "------------------------------------------------------------------------------------------------------------------------------------------- "
				pred = self.model_obj.getDecoderModel(config, is_training=True, mode='training', reuse=reuse, bucket_num=bucket_num)
				self.preds.append(pred)
				self.cost_list.append( self.model_obj.cost )
				reuse=True
			
			decoder_bucket_number = 0
			decoder_outputs_preds = self.model_obj.getDecoderModel(config, is_training=False, mode='training', reuse=True, bucket_num=decoder_bucket_number)	
			self.decoder_outputs_preds = decoder_outputs_preds

			self.token_input_sequences_placeholder_list  = self.model_obj.token_input_sequences_placeholder_list
			self.token_output_sequences_placeholder_list = self.model_obj.token_output_sequences_placeholder_list
			self.mask_list = self.model_obj.masker_list
		else:
			#config['batch_size'] = 5
			encoder_outputs = self.model_obj.getEncoderModel(config, mode='inference', reuse=reuse)
			self.decoder_outputs_inference = self.model_obj.getDecoderModel(config, is_training=False, 	mode='inference', reuse=False)	

	def trainModel(self, config, train_feed_dict, val_feed_dct, reverse_vocab, do_init=True):
		
		# Initializing the variables
		if do_init:
			init = tf.global_variables_initializer()
			sess = tf.Session()
			sess.run(init)
			self.sess= sess

		saver = tf.train.Saver()

		print("============== \n Printing all trainainble variables")
		for v in tf.trainable_variables():
			print(v)
		print("==================")


		for bucket_num,bucket in enumerate(self.buckets):
			input_sequences, output_sequences = train_feed_dict[bucket_num]
			print input_sequences[0]
			print output_sequences[0]
			#cost = self.model_obj.cost

			# if y is passed as (N, seq_length, 1): change it to (N,seq_length)
			if len(output_sequences.shape)==3:
				output_sequences=np.reshape(output_sequences, (output_sequences.shape[0], output_sequences.shape[1]))

			#create temporary feed dictionary
			token_input_sequences_placeholder = self.token_input_sequences_placeholder_list[bucket_num]
			token_output_sequences_placeholder = self.token_output_sequences_placeholder_list[bucket_num]
			feed_dct={token_input_sequences_placeholder:input_sequences, token_output_sequences_placeholder:output_sequences }

			#print "token_lookup_sequences_placeholder,  = ",token_lookup_sequences_placeholder, "\n token_output_sequences_decoder_placeholder = ",token_output_sequences_decoder_placeholder,"token_lookup_sequences_decoder_placeholder=",token_lookup_sequences_decoder_placeholder
			#print "\n encoder_inputs = ",encoder_inputs.shape, "\ndecoder_outputs =  ",decoder_outputs.shape, "\n decoder_inputs =  ", decoder_inputs.shape

			pred = self.preds[bucket_num]
			masker = self.mask_list[bucket_num]
			cost = self.cost_list[bucket_num]

			# Gradient descent
			learning_rate=0.1
			batch_size=config['batch_size']
			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

			sess = self.sess

			training_iters=50
			display_step=2
			sample_step=2
			save_step = 39
			n = feed_dct[token_input_sequences_placeholder].shape[0]
			# Launch the graph
			step = 1
			#preds = np.array( sess.run(self.pred, feed_dict= feed_dct) )
			#print preds
			#with tf.Session() as sess:
			while step < training_iters:
				#num_of_batches =  n/batch_size #(n+batch_size-1)/batch_size
				num_of_batches =  (n+batch_size-1)/batch_size
				for j in range(num_of_batches):
					#print "j= ",j
					feed_dict_cur = {}
					for k,v in feed_dct.items():
						feed_dict_cur[k] = v[j*batch_size:min(n,(j+1)*batch_size)]
						#print feed_dict_cur[k].shape
					cur_out = feed_dict_cur[token_output_sequences_placeholder]
					x,y = np.nonzero(cur_out)
					mask = np.zeros(cur_out.shape, dtype=np.float)
					mask[x,y]=1
					feed_dict_cur[masker]=mask

					sess.run(optimizer, feed_dict=feed_dict_cur )
					if step % display_step == 0:
						if j<10:
						#print " j = ",j
							loss = sess.run(cost, feed_dict= feed_dict_cur)
							print "step ",step," : ",loss
					if step % sample_step == 0:
						if j==0:
		  					self.runInference( config, input_sequences[:batch_size], output_sequences[:batch_size], reverse_vocab, sess )
							'''pred_cur = np.array( sess.run(pred, feed_dict= feed_dict_cur) )
							print pred_cur.shape
							print pred_cur[0].shape
							print np.sum(pred_cur[0],axis=1)
							'''
				if step%save_step==0:
					save_path = saver.save(sess, "./tmp/model"+str(step)+".ckpt")
	  				print "Model saved in file: ",save_path
				step += 1

		self.saver = saver


	###################################################################################

	def runInference(self, config, input_sequences, output_sequences, reverse_vocab, sess=None, print_all=True): # sampling
		print " Inference STEP ...... ============================================================"
		if sess==None:
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
	  		saver.restore(sess, "./tmp/model39.ckpt")
		typ = "greedy" #config['inference_type']
		model_obj = self.model_obj
		bucket_num = 0
		end_index = 2 # TO_DO load this from config
		feed_dct={model_obj.token_input_sequences_placeholder_list[bucket_num]:input_sequences}
		batch_size = config['batch_size'] #x_test.shape[0]
		preds =  self.decoder_outputs_preds
		if typ=="greedy":
			preds = np.array( sess.run(preds, feed_dict= feed_dct) ) # timesteps, N, vocab_size
			timesteps, N, vocab_size = preds.shape
			print "preds.shape = ",preds.shape
			assert N==batch_size
			probs = np.zeros(batch_size)
			for i in range(batch_size):
				t=0 
				while t<timesteps and output_sequences[i][t]!=0:
					next_word = output_sequences[i][t]
					next_word_prob = preds[t][i][next_word]
					probs[i] += np.log(next_word_prob)
					t+=1
			print "log likelihoods are as follows : "
			print probs

	###################################################################################

	def solveAll(self, config, encoder_inputs, decoder_ground_truth_outputs, reverse_vocab, sess=None): # sampling
		print " SolveAll ...... ============================================================"
		batch_size = config['batch_size']
		num_batches = ( len(encoder_inputs) + batch_size - 1)/ batch_size 
		print "num_batches = ",num_batches
		print "batch_size = ",batch_size
		print "len(encoder_inputs) = ",len(encoder_inputs)
		decoder_outputs_inference = []
		for i in range(num_batches):
			print "i= ",i
			encoder_inputs_cur = encoder_inputs[i*batch_size:(i+1)*batch_size]
			decoder_gt_outputs_cur = decoder_ground_truth_outputs[i*batch_size:(i+1)*batch_size]
			lim = len(encoder_inputs_cur)
			if len(encoder_inputs_cur)<batch_size:
				gap = batch_size - len(encoder_inputs_cur)
				for j in range(gap):
					encoder_inputs_cur = np.vstack( (encoder_inputs_cur,encoder_inputs[0]) )
					decoder_gt_outputs_cur = np.vstack( (decoder_gt_outputs_cur,decoder_ground_truth_outputs[0]) )
					#decoder_gt_outputs_cur.extend(decoder_ground_truth_outputs[0]*gap)
			decoder_outputs_inference_cur = self.runInference(config, encoder_inputs_cur, decoder_gt_outputs_cur, reverse_vocab, sess=None, print_all=False)
			decoder_outputs_inference.extend( decoder_outputs_inference_cur[:lim] )
		print len(encoder_inputs)
		print len(decoder_outputs_inference)
		print decoder_outputs_inference[0], decoder_ground_truth_outputs[0]
		print utils.getScores(decoder_outputs_inference, decoder_ground_truth_outputs)


########################################################################################
