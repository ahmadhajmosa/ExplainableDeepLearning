import numpy as np
import pandas as pd
from collections import deque
from functions import random_data, random_rules
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, \
    Activation, Dropout, Reshape, Concatenate, Flatten
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
def create_model(state_len,n_variable,rule_size):
        model_hyperparameters= dict()
        model_hyperparameters['inp_seq_len'] = state_len

        model_hyperparameters['Bidirectional_Layer1'] = 100
        model_hyperparameters['Bidirectional_Layer2'] = 100

        model_input = Input(shape=(model_hyperparameters['inp_seq_len'], n_variable), name='model_input')

        with K.name_scope("Bidirectional_Layer1"):  # Bidirectional LSTM over the input sequences
            bi_directional1 = Bidirectional(
                LSTM(model_hyperparameters['Bidirectional_Layer1'], return_sequences=True, recurrent_dropout=0.1,
                     kernel_initializer='he_uniform', name="bi_directional1"),merge_mode='concat')(model_input)

        with K.name_scope("Bidirectional_Layer2"):  # Bidirectional LSTM over the input sequences
            bi_directional2 = Bidirectional(
                LSTM(model_hyperparameters['Bidirectional_Layer2'], return_sequences=True, recurrent_dropout=0.1,
                     kernel_initializer='he_uniform', name="bi_directional2"),merge_mode='concat')(bi_directional1)
        with K.name_scope('Attention_layer'):
            attention_decoder = AttentionDecoder(model_hyperparameters['Bidirectional_Layer2'], n_variable*2 + 3,
                                                 name="output_attention_decoder",return_probabilities=False)(bi_directional2)
            attention_prop = AttentionDecoder(model_hyperparameters['Bidirectional_Layer2'], n_variable*2 + 3,
                                              name="output_attention_decoder",
                                              return_probabilities=True)(bi_directional2)
        with K.name_scope('Output_layer'):

            model = Model(model_input, attention_decoder)
            model_prop = Model(model_input, attention_prop)


        return model
def target_vec_rule_dict(n_variable, state_len):
    vec_to_rule = dict()
    rule_to_vec = dict()

    for i in range(n_variable):
        vec_to_rule[i] = 'X{}' + str(i)
        rule_to_vec['X' + str(i)] = i

        vec_to_rule[i + n_variable ] = '~X' + str(i)
        rule_to_vec['~X' + str(i)] = i + n_variable

    vec_to_rule[n_variable*2+1]='&'
    vec_to_rule[n_variable*2+2]='|'
    vec_to_rule[n_variable*2+3]='->'

    rule_to_vec['&'] = n_variable*2+1
    rule_to_vec['|'] = n_variable*2+2
    rule_to_vec['->'] = n_variable*2+3

    return  vec_to_rule, rule_to_vec

vec_to_rule, rule_to_vec = target_vec_rule_dict(n_variable, state_len)

def evaluate_rule(real_action,data,n_variable):

    num_op = int((len(real_action) +1) /  2  - 1)
    op =  1
    for op in range(num_op):
        if op == 0:
            oper = real_action[2 * op + 1]
            var1 = real_action[2 * op]
            var2 = real_action[2 * op + 2]
            if '~' not in var1 :

                p1 = data.iloc[:,rule_to_vec[var1]]
            else:
                p1 = data.iloc[:,rule_to_vec[var1]- n_variable]

            if '~' not in var2 :

                p2 = data.iloc[:,rule_to_vec[var2]]
            else:
                p2 = data.iloc[:,rule_to_vec[var2]- n_variable]
            result = eval('p1' + oper + 'p2')
        else:
            oper = real_action[2 * op + 1]
            var2 = real_action[2 * op + 2]
            if '~' not in var2 :

                p2 = data.iloc[:,rule_to_vec[var2]]
            else:
                p2 = data.iloc[:,rule_to_vec[var2]- n_variable]

            result = eval('result' + oper + 'p2')

        print(oper, var2 ,result)
    return result

def make_recursive(real_action, data, rule_to_vec, n_variable):
    op =  0
    num_op_org = int((len(real_action) +1) /  2  - 1)
    # num_op_org=3
    rule_iter = []
    data_iter = [data]
    for op_org in range(num_op_org-2):
        print(op_org)
        num_op = int((len(real_action) +1) /  2  - 1)
        op = 0
        if True:
            oper = real_action[2 * op + 1]
            var1 = real_action[2 * op]
            var2 = real_action[2 * op + 2]

            rule_iter.append([var1,oper,var2])
            if '~' not in var1 :

                p1 = data.iloc[:,rule_to_vec[var1]]
            else:
                p1 = data.iloc[:,rule_to_vec[var1]- n_variable]

            if '~' not in var2 :

                p2 = data.iloc[:,rule_to_vec[var2]]
            else:
                p2 = data.iloc[:,rule_to_vec[var2]- n_variable]
            result = eval('p1' + oper + 'p2')
            data_rec = data.copy()
            if int(var1.split('X')[1]) < int(var2.split('X')[1]) :
                if '~' not in var1:
                    data_rec[var1] = result


                else:
                    data_rec.iloc[:,rule_to_vec[var1]- n_variable] =  result
                if '~' not in var2:

                    data_rec = data_rec.drop(data_rec.columns[rule_to_vec[var2]],axis=1)
                else:
                     data_rec = data_rec.drop(data_rec.columns[rule_to_vec[var2]- n_variable],axis=1)

                if (num_op>2):
                    var3  = real_action[2 * (op+1) + 2]
                    oper2 = real_action[2 * (op+1) + 1]
                    if int(var3.split('X')[1]) > int(var1.split('X')[1]):
                        var3 =  'X' + str(int(var3.split('X')[1]) - 1)
                    if int(real_action[-1].split('X')[1]) > int(var1.split('X')[1]):
                        real_action[-1] = 'X' + str(int(real_action[-1].split('X')[1]) - 1)
                    del real_action[0:2]
                    real_action[0] = 'X' + var1.split('X')[1]
                    real_action[2] = var3
                elif (num_op == 2):
                    var3  = real_action[2 * (op+1) + 2]
                    oper2 = real_action[2 * (op+1) + 1]
                    real_action[0] = 'X' + var1.split('X')[1]
                    real_action[2] = var3
                    rule_iter.append(real_action)

            else:
                if '~' not in var2:
                    data_rec[var2] = result
                else:
                    data.iloc[:,rule_to_vec[var2]- n_variable] =  result
                if '~' not in var1:

                    data_rec = data_rec.drop(data_rec.columns[rule_to_vec[var1]],axis=1)
                else:
                     data_rec = data_rec.drop(data_rec.columns[rule_to_vec[var1]- n_variable],axis=1)
                if (num_op>2):
                    var3  = real_action[2 * (op+1) + 2]
                    oper2 = real_action[2 * (op+1) + 1]
                    if int(var3.split('X')[1]) > int(var2.split('X')[1]):
                        var3 =  'X' + str(int(var3.split('X')[1]) - 1)
                    if int(real_action[-1].split('X')[1]) > int(var2.split('X')[1]):
                        real_action[-1] = 'X' + str(int(real_action[-1].split('X')[1]) - 1)
                    del real_action[0:2]
                    real_action[0] = 'X' + var2.split('X')[1]
                    real_action[2] = var3
                elif (num_op == 2):
                    var3  = real_action[2 * (op+1) + 2]
                    oper2 = real_action[2 * (op+1) + 1]
                    real_action[0] = 'X' + var2.split('X')[1]
                    real_action[2] = var3
                    rule_iter.append(real_action)


            new_columns = []
            for col in data_rec.columns:
                print(col)
                if int(var1.split('X')[1]) < int(var2.split('X')[1]):

                    if int(col.split('X')[1]) > int(var2.split('X')[1]):
                        col_new = 'X' + str(int(col.split('X')[1]) - 1)
                        new_columns.append(col_new)
                    else:
                        col_new = col
                        new_columns.append(col_new)
                else:
                    if int(col.split('X')[1]) > int(var1.split('X')[1]):
                        col_new = 'X' + str(int(col.split('X')[1]) - 1)
                        new_columns.append(col_new)
                    else:
                        col_new = col
                        new_columns.append(col_new)
            data_rec.columns = new_columns
            data_rec['X' + str(int(new_columns[-1].split('X')[1])+1)] = np.ones(len(data_rec)) *-1
            data_iter.append(data_rec)
            data = data_rec.copy()


    return data_iter, rule_iter




def create_training_batch(n_variable, state_len):
                state, data = random_data(n_variable, state_len)
                data=data.astype(bool)

                data_neg = ~data
                data_neg=data_neg.astype(int)
                response = np.empty(shape=(state_len, 1))

                real_action, target_column = random_rules(1, size=n_variable, brackets= False)
                real_action = real_action[0].split(' ')
                data=data.astype(bool)
                # define a true rule to generate the X10 variable
                # here: X1 & X2 & (~X5 | ~X8)
                target = evaluate_rule(real_action,data,n_variable)

                data[target_column[0]] = target

                data=data.astype(int)
                real_action.append('->')
                real_action.append(target_column[0])
                date_iter,rule_iter = make_recursive(real_action, data, rule_to_vec, n_variable)

                target_vec = np.zeros((1,state_len,n_variable*2+2))



                return data
n_variable=10
state_len= 100
model = create_model(state_len,n_variable,2)
model.summary()
model.compile(loss='losses', loss_weights=lossWeights,optimizer='adam',metrics=['accuracy'])
data=create_training_batch(n_variable, state_len)
for i in range(mem_len):
    print(i, 'from: ',mem_len)
    try:


            # generate the state of each iteration







            D_raw.append(data)
            if i % 10 ==0:
                print('writing_{}'.format(i))
                with open('D_raw_2.pkl', 'wb') as fp:
                     pickle.dump(D_raw, fp)


    except:
          continue
