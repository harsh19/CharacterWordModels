
char_or_word= "char"  #"word"
max_word_seq_length = 40
max_char_sequence_length = 200
if char_or_word=="word":
	MAX_SEQUENCE_LENGTH = max_word_seq_length
else:
	MAX_SEQUENCE_LENGTH = max_char_sequence_length
print "max_input_seq_length= ",MAX_SEQUENCE_LENGTH
#MAX_VOCAB_SIZE=1500
embeddings_dim = 200
batch_size = 32
dropout_val = 0.2
lstm_cell_size=200
data_src = "./data/ptb"
#print "embeddings_dim = ", embeddings_dim
