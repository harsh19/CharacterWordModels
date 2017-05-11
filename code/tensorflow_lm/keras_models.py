import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge, SimpleRNN, TimeDistributed
import keras
import utilities

model = None
validation_data = None

class PerplexityCalculator(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global model
        global validation_data
        res = model.predict(validation_data[0])
        val_y = validation_data[1]
        pred = utilities.getPerplexityFromProbs(res, val_y)
        print "\n\n"
        print "Perplexity: " + pred
        print "\n\n"


class RNNModel:
	def getModel(self, params, weight=None  ):
		global model
		
		lstm_cell_size = params['lstm_cell_size']
		initial_state= np.random.rand(lstm_cell_size)
		print "params['embeddings_dim'] = ", params['embeddings_dim']
		print "lstm_cell_size= ", lstm_cell_size
		inp = Input(shape=(params['inp_length'],), dtype='int32', name="inp")
		embedding = Embedding(input_dim = params['vocab_size']+1, output_dim = params['embeddings_dim'],
			input_length = params['inp_length'],
			dropout=0.2,
			mask_zero=True,
			trainable=True) (inp)
		lstm_out = LSTM(lstm_cell_size, return_sequences=True)(embedding)
		#lstm_out = HiddenStateLSTM(lstm_cell_size, return_sequences=True)(embedding,initial_state)
		out = TimeDistributed( Dense(params['vocab_size'], activation='softmax') )(lstm_out)
		model = Model(input=[inp], output=[out])
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
		print model.summary()

		return model

	