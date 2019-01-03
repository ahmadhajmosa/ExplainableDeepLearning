from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from attention_decoder import AttentionDecoder
from collections import deque
import keras
import random
import numpy as np
from functions import random_data, random_rules
import pandas as pd


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Deep Q-learning Agent
class DataEnv:
    def __init__(self, data_len, n_variable):
        self.state_len = data_len
        self.n_variable = n_variable
        self.max_n_variable=6
        self.state=0
        self.data=[]


    def generate_env(self,rule):
        # Neural Net for Deep-Q learning Model

        self.state, self.data = random_data(self.n_variable, self.state_len)
        self.define_rule(rule)
        self.create_langs()
        self.input_text = self.state
        self.input_tokenizer, self.input_word_index, self.input_index2word, self.input_data = self.tokenize(self.state, self.max_n_variable,
                                                                                        self.input_lang.n_words)

        self.output_tokenizer, self.output_word_index, self.output_index2word, self.output_data = self.tokenize(self.output_text,
                                                                                            self.max_n_variable,
                                                                                       self.output_lang.n_words)

    def define_rule(self,rule):
        # Neural Net for Deep-Q learning Model
        self.response = np.empty(shape=(self.state_len, 1))
        # real_action = random_rules(1, size=np.random.randint(1,3))
        self.data = self.data.astype(bool)
        if rule:
            self.target = eval(rule)

        else:
            # if empty rule use xor rule
            xor_rule = "(self.data['X0'] | self.data['X1']) & (~self.data['X0'] | ~self.data['X1'])"
            xor_rule = "(self.data['X0'] | self.data['X1'])"

            self.target=eval(xor_rule)


        for j in range(self.data.shape[0]):
            if self.target.iloc[j]:
                self.state[j] = self.state[j] + 'X{}= 1'.format(self.n_variable)
                self.response[j] = int(1)
            else:
                self.state[j] = self.state[j] + 'X{}= 0'.format(self.n_variable)
                self.response[j] = int(0)
        new_col = pd.DataFrame(self.response, columns=['X{}'.format(self.n_variable)])
        self.data = self.data.join(new_col)
        self.data = self.data.astype(bool)

    def create_input_lang(self):
        self.input_text = []
        for col in self.data.columns:
            self.input_lang.addSentence(col + '= ')
            self.input_text.append(col + '= ')
        self.input_lang.addSentence('0')
        self.input_text.append('0')

        self.input_lang.addSentence('1')
        self.input_text.append('1')

    def create_output_lang(self):
        self.output_text = []

        for col in self.data.columns[:-1]:
            self.output_lang.addSentence("self.data['{}']".format(col))
            self.output_lang.addSentence("~self.data['{}']".format(col))
            self.output_text.append("self.data['{}']".format(col))
            self.output_text.append("~self.data['{}']".format(col))

        self.output_lang.addSentence('|')
        self.output_text.append('|')
        self.output_lang.addSentence('&')
        self.output_text.append('&')
        self.output_lang.addSentence('~')
        self.output_text.append('~')
        self.output_lang.addSentence('(')
        self.output_text.append('(')
        self.output_lang.addSentence(')')
        self.output_text.append(')')
        self.output_lang.addSentence(' <EOS> ')
        self.output_text.append(' <EOS> ')

    def create_langs(self):

        # Read the file and split into lines

        # Split every line into pairs and normalize

        # Reverse pairs, make Lang instances

        self.input_lang = Lang('input values')
        self.output_lang = Lang('rules')

        self.create_input_lang()

        self.create_output_lang()

    def tokenize(self,texts,MAX_SEQUENCE_LENGTH, MAX_NB_WORDS):
        tokenizer = Tokenizer(MAX_NB_WORDS,filters=":",lower=False)
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index  # the dict values start from 1 so this is fine with zeropadding
        index2word = {v: k for k, v in word_index.items()}
        print('Found %s unique tokens' % len(word_index))
        sequences = tokenizer.texts_to_sequences(texts)
        data_1 = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
        return tokenizer, word_index, index2word, data_1

    def step(self,action,action_index,prob):
        reward = []
        actionpost=[]
        action_indexpost=[]
        probpost=[]
        erro=0
        for i in range(1, len(action)):
            try:

                reward.append(np.sum(np.equal(eval(' '.join(action[0:i])), self.data[self.data.columns[-1]])))
                actionpost.append(action[i-1])
                action_indexpost.append(action_index[i-1])
                probpost.append(prob[i-1])
                #reward[i] = eval(action[0:i])
            except:
                reward.append(erro)
                actionpost.append(action[i-1])
                action_indexpost.append(action_index[i-1])
                probpost.append(prob[i-1])
                #reward[i] = eval(action[0:i
            if (action[i] == '<EOS>') | (action[i] == 'None'):
                erro=-1
        reward.append(erro)
        actionpost.append(action[-1])
        action_indexpost.append(action_index[-1])
        probpost.append(prob[-1])
        envindex=np.random.randint(0,len(self.state))
        return self.state[envindex],reward,actionpost,action_indexpost,probpost

    def reset(self):
        return self.state[0]