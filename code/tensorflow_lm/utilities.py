from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import random
import heapq
import re
import numpy as np
import scipy.stats

def generateCandidates(w1, w2):
    n=len(w1)
    m=len(w2)
    all_candidates = []
    for i in range(1,n+1):
        for j in range(m):
            all_candidates.append(w1[0:i]+w2[j:])
    #print "len(all_candidates)= ",len(all_candidates)
    all_candidates=set(all_candidates)
    #print "len(all_candidates) after deduplication= ",len(all_candidates)
    return all_candidates

def getMaxSubsequence(w1,w2):
    ret=0
    for i in range(min(len(w1),len(w2))):
        if w1[i]!=w2[i]:
            return i
    return min( len(w1),len(w2) )

def getMaxSubsequenceRev(w1,w2):
    wtemp1=''.join(reversed(list(w1)))
    wtemp2=''.join(reversed(list(w2)))
    return getMaxSubsequence(wtemp1,wtemp2)

def scoresToRanks(scores, rev=False):
    sorted_scores = sorted(scores)
    ranks=[]
    prev=-1
    prev_rank=None
    m=len(scores)
    for i,score in enumerate(sorted_scores):
        if score==prev:
            ranks.append(prev_rank)
        else:
            ranks.append(i+1)
            prev_rank=i+1
            prev=score
    ranks = {score:rank for score,rank in zip(sorted_scores,ranks)}
    if rev:
        return [(m-ranks[s]+1) for s in scores]
    else:
        return [ranks[s] for s in scores]

def getEditDistance(w1,w2):
    ret=0
    n,m = len(w1),len(w2)
    dp=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            dp[i][j]=n*m
    for i,ch1 in enumerate(w1):
        for j,ch2 in enumerate(w2):
            if i==0 and j==0:
                if ch1==ch2:
                    dp[i][j]=0
                else:
                    dp[i][j]=2
            elif i==0:
                if ch1==ch2:
                    dp[i][j]=j
                else:
                    dp[i][j]=1+dp[i][j-1]
            elif j==0:
                if ch1==ch2:
                    dp[i][j]=i
                else:
                    dp[i][j]=1+dp[i-1][j]
            else:
                cost=1
                dp[i][j] = min( dp[i][j], cost + min(dp[i-1][j], dp[i][j-1])  )
                if ch1==ch2:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1])
    return dp[n-1][m-1]


def spitToBlendableOrNot(data):
    all_w1,all_w2,all_gold=data
    typ=[]
    for w1,w2,gold in zip(all_w1,all_w2,all_gold):
        a=getMaxSubsequence(w1,gold)
        b=getMaxSubsequenceRev(w2,gold)
        if (a+b)>=len(gold):
            typ.append(1)
        else:
            typ.append(0)
    return typ

def evaluate(gold, pred):
    edits = []
    em=0
    for g,p in zip(gold, pred):
        edits.append(getEditDistance(g,p))
        if g==p:
            em+=1
    avg_edit = np.mean(edits)
    print "avg_edit, em: ",avg_edit,em

def readWholeData(knight_only=False, f = "../Data/dataset.csv"):
    ret = []
    for line in open(f,"r"):
        vals=re.split("\W+",line)
        if knight_only:
            if vals[3]!='knight':
                continue
        vals=vals[:3]
        words=[word.lower().strip() for word in vals]
        if len(words)!=3:
            print "ERRRR "

        ret.append(words)
    return ret



################################################################
class TopN(object):
	"""Maintains the top n elements of an incrementally provided set."""

	def __init__(self, n):
		self._n = n
		self._data = []

	def size(self):
		assert self._data is not None
		return len(self._data)

	def push(self, x):
		"""Pushes a new element."""
		assert self._data is not None
		if len(self._data) < self._n:
			heapq.heappush(self._data, x)
		else:
			heapq.heappushpop(self._data, x)

	def extract(self, sort=False):
		"""Extracts all elements from the TopN. This is a destructive operation.
		The only method that can be called immediately after extract() is reset().
		Args:
			sort: Whether to return the elements in descending sorted order.
		Returns:
			A list of data; the top n elements provided to the set.
		"""
		assert self._data is not None
		data = self._data
		self._data = None
		if sort:
			data.sort(reverse=True)
		return data

	def reset(self):
		"""Returns the TopN to an empty state."""
		self._data = []

################################################################

class OutputSentence(object):
	"""Represents a complete or partial caption."""

	def __init__(self, sentence, state, logprob, score, metadata=None):
		"""Initializes the Caption.
		Args:
			sentence: List of word ids in the caption.
			state: Model state after generating the previous word.
			logprob: Log-probability of the caption.
			score: Score of the caption.
			metadata: Optional metadata associated with the partial sentence. If not
				None, a list of strings with the same length as 'sentence'.
		"""
		self.sentence = sentence
		self.state = state
		self.logprob = logprob
		self.score = score
		self.metadata = metadata

	def __cmp__(self, other):
		"""Compares Captions by score."""
		assert isinstance(other, OutputSentence)
		if self.score == other.score:
			return 0
		elif self.score < other.score:
			return -1
		else:
			return 1
	
	# For Python 3 compatibility (__cmp__ is deprecated).
	def __lt__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score < other.score
	
	# Also for Python 3 compatibility.
	def __eq__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score == other.score

################################################################

def sampleFromDistribution(vals):
		p = random.random()
		s=0.0
		for i,v in enumerate(vals):
				s+=v
				if s>=p:
						return i
		return len(vals)-1




def sampleFromDistribution(vals):
    p = random.random()
    s=0.0
    for i,v in enumerate(vals):
        s+=v
        if s>=p:
            return i
    return len(vals)-1

def generateSentence(model, word_to_index, start_token, end_token, unknown_token, maxlen = config.MAX_SEQUENCE_LENGTH-1 ):
    x = [ word_to_index[start_token] ]
    i=1
    while i<maxlen:
        x_temp = pad_sequences([x], maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        data = np.array( [ sequence[:-1] for sequence in x_temp] ) # only 1 sequence is actually there
        y = model.predict( data )
        #if i==1:
        idx = sampleFromDistribution(y[0][i])    
        #else:
        #    idx = np.argmax(y[0][i])
        if idx == word_to_index[end_token]:
            return x
        if idx == word_to_index[unknown_token]:
            i = 1
            x = [ word_to_index[start_token] ]
            print "Found unknown char. Retrying."
            continue
        x.append(idx)
        i += 1
    return x[1:] # removing sentence start
