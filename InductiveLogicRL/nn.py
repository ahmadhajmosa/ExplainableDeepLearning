import numpy as np
import pandas as pd
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
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from scipy import spatial
from keras import regularizers
from AttentionDecoder import AttentionDecoder
import pickle
from keras.models import Sequential
from policyagent import PolicyAgent


# generate data stream (a.k.a. recall memory)
D = deque()
D_raw = deque()
#with open('D_raw_1.pkl', 'rb') as fp:
#    D_raw=pickle.load(fp)



mem_len = 1000000
state_len = 10
n_variable = 10
trails=100

np.random.seed(1)

import sympy
from sympy import *
sm= symbols('X0:3')
import itertools
import pandas as pd
n_variable = 3
truth_table = list(itertools.product([0, 1], repeat=n_variable))
truth_table_df= pd.DataFrame(truth_table, columns= np.asarray(sm).astype('str'))
# write a target logical expression, that the sympy should find
y=(truth_table_df['X0'] & truth_table_df['X1']) |  truth_table_df['X2']
# find the miniterms, where y is true

def target_vec_rule_dict(n_variable):
    vec_to_rule = dict()
    rule_to_vec = dict()
    vec_to_op = dict()
    op_to_vec = dict()


    for i in range(n_variable):
        vec_to_rule[i] = 'X' + str(i)
        rule_to_vec['X' + str(i)] = i

        vec_to_rule[i + n_variable ] = '~X' + str(i)
        rule_to_vec['~X' + str(i)] = i + n_variable

    vec_to_op[0]='&'
    vec_to_op[1]='|'
    vec_to_op[2]='='

    op_to_vec['&'] = 0
    op_to_vec['|'] = 1
    op_to_vec['='] = 2

    return  vec_to_rule, rule_to_vec, vec_to_op, op_to_vec


vec_to_symbol, symbol_to_vec, vec_to_op, op_to_vec = target_vec_rule_dict(n_variable)

minterms=truth_table_df[y==1].values.tolist()
minterms
agent = PolicyAgent(intermediate_dim=50,MAX_SEQUENCE_LENGTH=1,NumCol=2,NB_VARS=n_variable*2,NB_WORDS_OUT=n_variable*2)
print(agent.model.summary())

def term_to_expr(minterms,vec_to_symbol,symbol_to_vec):
        expr_list = []
        for i in range(len(minterms)):
            expr=[]
            for k in range(len(minterms[i])):
                if minterms[i][k] == 0 :
                    expr.append(vec_to_symbol[k+n_variable])
                else:
                    expr.append(vec_to_symbol[k])
            expr_list.append(expr)
        return expr_list

expr_list = term_to_expr(minterms,vec_to_symbol,symbol_to_vec)

def cov_over_seq(expr_list,vec_to_symbol,symbol_to_vec):
    input1_sample = []
    input2_sample = []
    expr_list_conv = []
    index=np.arange(0,len(expr_list))

    for i in range(len(expr_list)):
        for j in range(len(expr_list)):
            if( i != j ):
                inp = np.zeros(n_variable*2)
                inp2 = np.zeros(n_variable*2)

                for k in range(len(expr_list[i])):

                    inp[symbol_to_vec[expr_list[i][k]]] = 1

                for k in range(len(expr_list[j])):

                    inp2[symbol_to_vec[expr_list[j][k]]] = 1

                input1_sample.append(inp)
                input2_sample.append(inp2)
                expr = [ex for num,ex in enumerate(expr_list) if (num !=i) &  (num !=j)]
                expr_list_conv.append(expr)
    return input1_sample, input2_sample, expr_list_conv

def get_reward_conv(action,truth_table_df,expr_list, y):
    expr_list_temp = expr_list.copy()
    #print(expr_list_temp)
    term_pre = np.zeros(len(truth_table_df)).astype(bool)
    for i in expr_list_temp:
        #print(i)
        term_val = np.ones(len(truth_table_df)).astype(bool)
        for j in i:
            #print(j)
            if j[0] == '~':
                #print(j[1:])
                term_val = term_val & ~truth_table_df[j[1:]].astype(bool) # Not then And
            else:
                #print(j)

                term_val = term_val & truth_table_df[j].astype(bool) # And
            #print(term_val)
        term_pre =  term_pre |  term_val
        #print(term_pre)  # Or
    #reward = 1 if np.sum(np.equal(term_pre,y)) / len(y) == 1 else -1
    reward = np.sum(np.equal(term_pre,y)) / len(y)

    return reward
expr_list_target= expr_list.copy()

#expr_list= expr_list_target.copy()
for iter in range(10000):
        expr_list = expr_list_target.copy()
        for depth in range(10):
            try:

                input1_sample, input2_sample, expr_list_conv = cov_over_seq(expr_list,vec_to_symbol,symbol_to_vec)
                prob_list = []
                prob_list_all = []
                action_list = []
                action_index_list = []
                reward_list = []
                state_list = []

                for conv_iter in range(len(input1_sample)):
                    state = [np.reshape(input1_sample[conv_iter],(1,6,1)), np.reshape(input2_sample[conv_iter],(1,6,1))]
                    action, action_index, prob = agent.conv_act(state,vec_to_symbol,symbol_to_vec)
                    expr_list_conv[conv_iter].append(action)
                    reward = get_reward_conv(action, truth_table_df, expr_list_conv[conv_iter],y)
                    reward_list.append(reward)
                    prob_list.append(prob)
                    prob_list_all.append(np.prod(prob))
                    action_list.append(action)
                    action_index_list.append(action_index)
                    state_list.append(state)

                winner_comb = np.argmax(reward_list)
                expr_list = expr_list_conv[winner_comb].copy()
                print(expr_list)
                print('rew',reward_list[winner_comb])
                #print(np.shape(prob_list[winner_comb]))
                agent.remember(state_list[winner_comb], action_list[winner_comb], action_index_list[winner_comb],reward_list[winner_comb], prob_list[winner_comb])
            except:
                break
        loss,ac = agent.train(iter)
        print('iter: ',iter, 'loss: ', loss)
        #print('sol: ',ac)
