import numpy as np
import pandas as pd
import sympy
from sympy import *
from collections import deque
from functions import random_data, random_rules
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, \
    Activation, Dropout, Reshape, Concatenate, Flatten, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from scipy import spatial
from keras import regularizers
from AttentionDecoder import AttentionDecoder
from collections import deque
import keras
import random
from sympy.logic import simplify_logic

import numpy as np
import pandas as pd
from functions import random_data, random_rules
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import tensorflow as tf


# Deep Q-learning Agent
class PolicyAgent:
    def __init__(self,intermediate_dim,MAX_SEQUENCE_LENGTH,NumCol,NB_VARS,NB_WORDS_OUT):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.99  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.intermediate_dim=100
        self.MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH
        self.NB_VARS=NB_VARS
        self.NB_WORDS_OUT=NB_WORDS_OUT
        self.emb_dim=50
        self.NumCol=NumCol
        self.model = self._build_model()#
        self.states = []
        self.actions = []

        self.gradients = []
        self.rewards = []
        self.probs = []
        self.log_path = './logs/lr8'

    def write_log(self, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, batch_no)
            self.callback.writer.flush()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model_hyperparameters= dict()
        model_hyperparameters['inp_seq_len'] = self.MAX_SEQUENCE_LENGTH

        model_hyperparameters['Bidirectional_Layer1'] = self.intermediate_dim
        model_hyperparameters['Bidirectional_Layer2'] = self.intermediate_dim

        model_input = Input(shape=(self.NB_VARS,1), name='model_input')
        model_input2 = Input(shape=(self.NB_VARS,1), name='model_input2')

        with K.name_scope("Bidirectional_Layer1"):  # Bidirectional LSTM over the input sequences
            bi_directional1 = Bidirectional(
                LSTM(model_hyperparameters['Bidirectional_Layer1'], return_sequences=True, recurrent_dropout=0.1,
                     kernel_initializer='he_uniform', name="bi_directional1"),merge_mode='concat')(model_input)
        with K.name_scope("Bidirectional_Layer2"):  # Bidirectional LSTM over the input sequences
            bi_directional2 = Bidirectional(
                LSTM(model_hyperparameters['Bidirectional_Layer1'], return_sequences=True, recurrent_dropout=0.1,
                     kernel_initializer='he_uniform', name="bi_directional1"),merge_mode='concat')(model_input2)

        merged_vector = concatenate([bi_directional1, bi_directional2], axis=-1)

        with K.name_scope('Attention_layer'):
            attention_decoder = AttentionDecoder(model_hyperparameters['Bidirectional_Layer2'], 1,
                                                 name="output_attention_decoder",return_probabilities=False)(merged_vector)
            attention_prop = AttentionDecoder(model_hyperparameters['Bidirectional_Layer2'], 1,
                                              name="output_attention_decoder",
                                              return_probabilities=True)(merged_vector)
        with K.name_scope('Output_layer'):

            model = Model([model_input, model_input2], attention_decoder)
            model_prop = Model([model_input, model_input2], attention_prop)
        model.compile(loss='binary_crossentropy',optimizer=SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1. ))
        self.model= model
        self.log_path = './logs/lr8'

        self.callback = TensorBoard(self.log_path)
        self.callback.set_model(self.model)
        self.model.metrics_names.append("weights_gradient_norm")
        self.model.metrics_tensors.append(self.get_gradient_norm())

        print(model.summary())
        return model

    def remember(self, state, action, action_index,reward, prob):
        y = np.zeros(np.shape(prob))
        for i in range(len(prob)):
            y[int(action_index[i])] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.actions.append(action)

        self.rewards.append(reward)
        self.probs.append(prob)


    def conv_act(self, state,state_list,vec_to_symbol,symbol_to_vec):
        #if np.random.rand() <= self.epsilon:
        #    rule=random_rules(1, size=self.NumCol)
        #    sequences = output_tokenizer.texts_to_sequences(rule)
         #   rule_indexes = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

         #   return rule, rule_indexes

        #sequences = input_tokenizer.texts_to_sequences(state.split(' '))
        #state = pad_sequences(np.transpose(sequences), maxlen=MAX_SEQUENCE_LENGTH)

        aprob = self.model.predict(state)
        aprob=aprob.squeeze()

        #self.probs.append(aprob)
        #prob = aprob / np.sum(aprob,axis=1)
        action_index=np.zeros(self.NB_VARS)
        action = []
        for i in range(self.NB_VARS):
            #print(i)
            action_index[i] = np.random.choice(2, 1, p=[aprob[i],1-aprob[i]])[0]
            if action_index[i] == 1:
                    action.append(vec_to_symbol[i])

        if self.epsilon > 0.1:
            if np.random.rand() <= self.epsilon:
                random_or_simplifier = np.random.choice(2,1)
                #random_or_simplifier = 0
                if random_or_simplifier == 0:
                    action ,action_index = self.simplify_expr(state_list, vec_to_symbol,symbol_to_vec)
                    if(len(action_index) == 0):
                        action_index = self.get_random_action(state_list)
                        #action_index = np.zeros(self.NB_VARS)
                else:
                        action_index = self.get_random_action(state_list)
                action = []
                for i in range(len(action_index)):
                    if action_index[i] == 1:
                        action.append(vec_to_symbol[i])



            #rule=random_rules(1, size=self.NumCol)
            #sequences = output_tokenizer.texts_to_sequences(rule)
            #action_index = pad_sequences(sequences,padding='post', maxlen=MAX_SEQUENCE_LENGTH)[0]



        return action,action_index,aprob  # returns action

    def get_random_action(self, state_list):
        indexes = []
        for i in range(2):
            indexes.append(list(np.where(state_list[i] == 1)))
        current_indexes = np.unique(np.concatenate(indexes,axis=1))
        random_per = np.random.choice(2, len(current_indexes))
        action_index = np.zeros(self.NB_VARS)
        action_index[current_indexes] = random_per
        return action_index

    def simplify_expr(self,terms, vec_to_symbol,symbol_to_vec):
            term1 = []
            term2 = []

            sms= symbols('X0:{}'.format(self.NB_VARS/2))
            for i in range(self.NB_VARS):
                    #print(i)
                    if terms[0][i] == 1:
                        term1.append(vec_to_symbol[i])
                    if terms[1][i] == 1:
                        term2.append(vec_to_symbol[i])
            expr = '(' + ' & '.join(term1) + ')' + '|' + '(' + ' & '.join(term2) + ')'
            if (term2 == []) or (term1 == []):
                return [],[]

            simplified_expr = simplify_logic(expr, form = 'cnf',deep = False)
            action_index = np.zeros(self.NB_VARS)
            action = []
            if(str(simplified_expr)=='True'):
                print('check')
            if(str(simplified_expr)=='False'):
                print('check')
            if ("|" in str(simplified_expr)) or (str(simplified_expr)=='True') or (str(simplified_expr)=='False'):
                return [],[]
            else:
                for sy in str(simplified_expr).split('&') :
                    action_index[symbol_to_vec[sy.strip()]] = 1
                    action.append(sy.strip())

                return    action, action_index


    def act(self, state,output_index2word,output_tokenizer,MAX_SEQUENCE_LENGTH,input_tokenizer):
        #if np.random.rand() <= self.epsilon:
        #    rule=random_rules(1, size=self.NumCol)
        #    sequences = output_tokenizer.texts_to_sequences(rule)
         #   rule_indexes = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

         #   return rule, rule_indexes

        sequences = input_tokenizer.texts_to_sequences(state.split(' '))
        state = pad_sequences(np.transpose(sequences), maxlen=MAX_SEQUENCE_LENGTH)

        aprob = self.model.predict(state)
        aprob=aprob.squeeze()

        self.probs.append(aprob)
        #prob = aprob / np.sum(aprob,axis=1)
        action_index=np.zeros(MAX_SEQUENCE_LENGTH)
        for i in range(MAX_SEQUENCE_LENGTH):
            action_index[i] = np.random.choice(output_tokenizer.num_words, 1, p=aprob[i])[0]

        if np.random.rand() <= self.epsilon:
            rule=random_rules(1, size=self.NumCol)
            sequences = output_tokenizer.texts_to_sequences(rule)
            action_index = pad_sequences(sequences,padding='post', maxlen=MAX_SEQUENCE_LENGTH)[0]

        action = np.vectorize(output_index2word.get)(action_index)


        return action,action_index,aprob  # returns action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards
    def get_gradient_norm(self):
        with K.name_scope('gradient_norm'):
            grads = K.gradients(self.model.total_loss, self.model.trainable_weights)
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        return norm

    def train(self,batch_no):

        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)

        rewards = rewards / (np.std(rewards - np.mean(rewards))+1e-20)
        gradients *= np.reshape(rewards,(-1,1))
        X1 = np.reshape(self.states,(-1,self.NB_VARS,1,2))[:,:,:,0]
        X2 = np.reshape(self.states,(-1,self.NB_VARS,1,2))[:,:,:,1]

        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        loss, gradient_norm  = self.model.train_on_batch([X1,X2],np.reshape(Y,(-1,self.NB_VARS,1)))
        current_actions=self.actions.copy()
        train_names = ['train_loss', 'train_w_gradient_norm','train_policy_gradient_norm','rewards','values']
        policy_gradient_norm = np.sqrt(sum([np.sum(np.square(g)) for g in gradients]))
        self.write_log(train_names, [loss, gradient_norm, policy_gradient_norm, rewards[-1],np.sum(rewards)], batch_no)
        if self.epsilon > self.epsilon_min:
            print(self.epsilon)
            self.epsilon *= self.epsilon_decay
        #print(policy_gradient_norm, gradient_norm)
        self.states, self.probs, self.gradients, self.rewards, self.actions = [], [], [], [], []
        return loss,current_actions



    def replay(self, batch_size,input_tokenizer,MAX_SEQUENCE_LENGTH,output_index2word):
        minibatch = random.sample(self.memory, batch_size)
        for state, action,action_index, reward, next_state in minibatch:
            sequences = input_tokenizer.texts_to_sequences(next_state.split(' '))
            next_state = pad_sequences(np.transpose(sequences),padding='post', maxlen=MAX_SEQUENCE_LENGTH)
            sequences = input_tokenizer.texts_to_sequences(state.split(' '))
            state = pad_sequences(np.transpose(sequences),padding='post', maxlen=MAX_SEQUENCE_LENGTH)

            target = reward + self.gamma * np.amax(self.model.predict(next_state) )
            target_f = self.model.predict(state)
            praction = np.vectorize(output_index2word.get)(np.argmax(target_f, axis=2))

            #target_fdf=pd.DataFrame(target_f[0])
            #action_index_df=pd.DataFrame(action_index[0])
            #target_fdf.iloc[action_index_df] =target

            for i in range(len(action_index[0])):
                target_f[0][i][action_index[0][i]]=target

            self.model.fit(state, target_f, epochs=3, verbose=0)
        print(praction)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
