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
			for bucket_num, bucket_dct in self.buckets.items():
				config['max_inp_seq_length'] = bucket_dct['max_input_seq_length']
				print ""
				print "------------------------------------------------------------------------------------------------------------------------------------------- "
				pred = self.model_obj.getDecoderModel(config, is_training=True, mode='training', reuse=reuse, bucket_num=bucket_num)
				self.preds.append(pred)
				self.cost_list.append( self.model_obj.cost )
				reuse=True
			self.token_input_sequences_placeholder_list  = self.model_obj.token_input_sequences_placeholder_list
			self.token_output_sequences_placeholder_list = self.model_obj.token_output_sequences_placeholder_list
			self.mask_list = self.model_obj.masker_list
			
			# for predictions / sampler
			bucket_num_for_preds = 0 # TO DO: creat separate input placeholder for this
			config['keep_prob'] = 1.0 # keep 1.0 during test time
			decoder_outputs_preds = self.model_obj.getDecoderModel(config, is_training=False, mode='training', reuse=True, bucket_num=bucket_num_for_preds)	
			self.decoder_inputs_preds = self.token_input_sequences_placeholder_list[bucket_num_for_preds]
			self.decoder_outputs_preds = decoder_outputs_preds
		else:
			config['keep_prob'] = 1.0 # keep 1.0 during test time
			decoder_outputs_preds = self.model_obj.getDecoderModel(config, is_training=False, mode='inference', reuse=reuse)	
			self.decoder_inputs_preds = self.model_obj.token_input_sequences_placeholder_inference
			self.decoder_outputs_preds = decoder_outputs_preds
			
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


		for bucket_num,bucket in enumerate(self.buckets): # move training_iters loop before this
			input_sequences, output_sequences = train_feed_dict[bucket_num]
			print "input_sequences[0] = ", input_sequences[0]
			print "output_sequences[0] = ", output_sequences[0]
			#cost = self.model_obj.cost

			# if y is passed as (N, seq_length, 1): change it to (N,seq_length)
			if len(output_sequences.shape)==3:
				output_sequences=np.reshape(output_sequences, (output_sequences.shape[0], output_sequences.shape[1]))

			#create temporary feed dictionary
			token_input_sequences_placeholder = self.token_input_sequences_placeholder_list[bucket_num]
			token_output_sequences_placeholder = self.token_output_sequences_placeholder_list[bucket_num]
			feed_dct={token_input_sequences_placeholder:input_sequences, token_output_sequences_placeholder:output_sequences }

			pred = self.preds[bucket_num]
			masker = self.mask_list[bucket_num]
			cost = self.cost_list[bucket_num]

			# Gradient descent
			learning_rate=0.001
			beta1=0.9
			beta2=0.999
			epsilon=1e-08
			use_locking=False
			name='Adam'
			batch_size=config['batch_size']
			optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon,use_locking=use_locking,name=name).minimize(cost)

			sess = self.sess

			training_iters=50
			display_step=1
			sample_step=2
			save_step = 1
			n = feed_dct[token_input_sequences_placeholder].shape[0]

			# Launch the graph
			step = 1
			#preds = np.array( sess.run(self.pred, feed_dict= feed_dct) )
			#print preds
			#with tf.Session() as sess:
			while step < training_iters:
				print "STEP = ",step
				#num_of_batches =  n/batch_size #(n+batch_size-1)/batch_size
				num_of_batches =  (n+batch_size-1)/batch_size
				for j in range(num_of_batches):
					if j%100==0:
						print "batch= ",j
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
	  				val_x,val_y = val_feed_dct
					self.getLoss(config, val_x, val_y, token_input_sequences_placeholder, token_output_sequences_placeholder, masker, cost, sess)
	  				self.solveAll(config, val_x, val_y, reverse_vocab, sess)

				if step % sample_step == 0:
  					self.runInference( config, input_sequences[:batch_size], output_sequences[:batch_size], reverse_vocab, sess )
					'''pred_cur = np.array( sess.run(pred, feed_dict= feed_dict_cur) )
					print pred_cur.shape
					print pred_cur[0].shape
					print np.sum(pred_cur[0],axis=1)
					'''
				if step%save_step==0:
					save_path = saver.save(sess, "./tmp/tf/model"+str(step)+".ckpt")
	  				print "Model saved in file: ",save_path

				step += 1

		self.saver = saver


	###################################################################################

	def runInference(self, config, input_sequences, output_sequences, reverse_vocab, sess=None, print_all=True): # Probabilties
		if print_all:
			print " Inference STEP ...... ============================================================"
		if sess==None:
			print "sess is None.. LOAD?ING SAVED MODEL"
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
	  		saved_model_path = config['saved_model_path']
	  		print "Loading saved model from : ",saved_model_path
	  		saver.restore(sess, saved_model_path)
		model_obj = self.model_obj
		end_index = 2 # TO_DO load this from config
		feed_dct={self.decoder_inputs_preds:input_sequences}
		batch_size = config['batch_size'] #x_test.shape[0]
		preds =  self.decoder_outputs_preds
		preds = np.array( sess.run(preds, feed_dict= feed_dct) ) # timesteps, N, vocab_size
		timesteps, N, vocab_size = preds.shape
		assert N==batch_size
		probs = np.zeros(batch_size)
		for i in range(batch_size):
			t=0 
			while t<timesteps and output_sequences[i][t]!=0:
				next_word = output_sequences[i][t]
				next_word_prob = preds[t][i][next_word]
				probs[i] += np.log(next_word_prob)
				t+=1
		if print_all:
			print "log likelihoods are as follows : "
			print probs
		return probs

	###################################################################################

	def solveAll(self, config, input_sequences, output_sequences, reverse_vocab, sess=None, dump_seq_prob=False, dump_seq_prob_path=None): # Probabilities
		print " SolveAll ...... ============================================================"
		if sess==None:
			print "sess is None.. LOAD?ING SAVED MODEL"
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
	  		saved_model_path = config['saved_model_path']
	  		print "Loading saved model from : ",saved_model_path
	  		saver.restore(sess, saved_model_path)
		batch_size = config['batch_size']
		num_batches = ( len(input_sequences) + batch_size - 1)/ batch_size 
		probs = []
		for i in range(num_batches):
			#print "i= ",i
			cur_input_sequences = input_sequences[i*batch_size:(i+1)*batch_size]
			cur_output_sequences = output_sequences[i*batch_size:(i+1)*batch_size]
			lim = len(cur_input_sequences)
			if len(cur_input_sequences)<batch_size:
				gap = batch_size - len(cur_input_sequences)
				for j in range(gap):
					cur_output_sequences = np.vstack( (cur_output_sequences, cur_output_sequences[0]) )
					cur_input_sequences = np.vstack( (cur_input_sequences, cur_input_sequences[0]) )
			cur_probs = self.runInference(config, cur_input_sequences, cur_output_sequences, reverse_vocab, sess=sess, print_all=False)
			probs.extend( cur_probs[:lim] )
		probs = np.array(probs)
		print "PREPLEXITY = ", utils.getPerplexityFromSumProbs(probs, output_sequences)
		if dump_seq_prob:
			prob_vals = utils.getSequenceProbs(probs, output_sequences)
			print "Dumping values to path= ",dump_seq_prob_path
			print "prob_vals.shape = ",prob_vals.shape
			fw = open(dump_seq_prob_path,"w")
			for gt_sequence,seq_prob in zip(output_sequences, prob_vals):
				#print "gt_sequence = ",gt_sequence
				#print "seq__prob = ",seq_prob
				gt_str_seq = ""
				for j in gt_sequence:
					if j==0:
						break
					gt_str_seq = gt_str_seq + " " + reverse_vocab[j]
				fw.write( gt_str_seq + "\t" + str(seq_prob) + "\n" )
			fw.close()

	###################################################################################

	def getLoss(self, config, input_sequences, output_sequences, inp_placeholder, out_placeholder, mask_placeholder,  loss_variable, sess): # Probabilities
		print " getLoss ...... ============================================================"
		batch_size = config['batch_size']
		num_batches = ( len(input_sequences) + batch_size - 1)/ batch_size 
		loss = []
		for i in range(num_batches):
			#print "i= ",i
			cur_input_sequences = input_sequences[i*batch_size:(i+1)*batch_size]
			cur_output_sequences = output_sequences[i*batch_size:(i+1)*batch_size]
			lim = len(cur_input_sequences)
			if len(cur_input_sequences)<batch_size:
				gap = batch_size - len(encoder_inputs_cur)
				for j in range(gap):
					cur_output_sequences = np.vstack( (cur_output_sequences, cur_output_sequences[0]) )
					cur_input_sequences = np.vstack( (cur_input_sequences, cur_input_sequences[0]) )
			feed_dct = {inp_placeholder:cur_input_sequences, out_placeholder:cur_output_sequences}
			mask = np.zeros(cur_output_sequences.shape, dtype=np.float)
			x,y = np.nonzero(cur_output_sequences)
			mask[x,y]=1
			feed_dct[mask_placeholder]=mask
			cur_loss = sess.run(loss_variable, feed_dct)
			loss.append( cur_loss )
		loss = np.array(loss)
		print "LOSS = ", np.mean(loss)


########################################################################################
