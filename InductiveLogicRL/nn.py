import numpy as np
import pandas as pd
from collections import deque
import json

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
import sympy
from sympy import *
import itertools
import pandas as pd
from sympy.logic.inference import satisfiable
from sympy.logic import simplify_logic
from sympy.logic.boolalg import is_cnf
import  time
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


sm= symbols('X0:3')

n_variable = 3
truth_table = list(itertools.product([0, 1], repeat=n_variable))
truth_table_df= pd.DataFrame(truth_table, columns= np.asarray(sm).astype('str'))

# write a target logical expression, that the sympy should find
y=(truth_table_df['X0'] & truth_table_df['X1']) |  truth_table_df['X2']  # x0x1 + x2


def target_vec_rule_dict(n_variable):  # create vec to symbol and symbol to vec dictinaries
    vec_to_sym = dict()   # vector keys and boolian variable symbol values
    sym_to_vec = dict()  # boolian symbol keys and vector values
    vec_to_op = dict()   # vect key and operation symbol values
    op_to_vec = dict()  # operation symbol keys and vect values


    for i in range(n_variable):
        vec_to_sym[i] = 'X' + str(i)
        sym_to_vec['X' + str(i)] = i

        vec_to_sym[i + n_variable ] = '~X' + str(i)
        sym_to_vec['~X' + str(i)] = i + n_variable

    vec_to_op[0]='&'
    vec_to_op[1]='|'
    vec_to_op[2]='='

    op_to_vec['&'] = 0
    op_to_vec['|'] = 1
    op_to_vec['='] = 2

    return  vec_to_sym, sym_to_vec, vec_to_op, op_to_vec


vec_to_symbol, symbol_to_vec, vec_to_op, op_to_vec = target_vec_rule_dict(n_variable)  # get mapping dictinaries

minterms=truth_table_df[y==1].values.tolist()  # slice rows where target is true

agent = PolicyAgent(intermediate_dim=50,MAX_SEQUENCE_LENGTH=1,NumCol=2,NB_VARS=n_variable*2,NB_WORDS_OUT=n_variable*2)  # initialize policy agent
print(agent.model.summary())


def term_to_expr(minterms, vec_to_symbol, symbol_to_vec):  # get miniterms (rows of binary vectors) and generate boolean expressions
        expr_list = []
        for i in range(len(minterms)):
            expr=[] # single expression initialization
            for k in range(len(minterms[i])): # for every elemnt in the row
                if minterms[i][k] == 0 :
                    expr.append(vec_to_symbol[k+n_variable]) # if the binary values is 0 then get the negative symbol ~X_k
                else:
                    expr.append(vec_to_symbol[k]) # get X_k
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
    symbols=[]
    for i in expr_list_temp:
        #print(i)

        term_val = np.ones(len(truth_table_df)).astype(bool)
        for j in i:
            symbols.append(j)
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

    acc = np.sum(np.equal(term_pre,y)) / len(y)
    if (acc > 0.7):
        reward = acc - 0.01 * np.log(len(symbols)+1e-10)
    else:
        reward = -acc - 0.01 * np.log(len(symbols)+1e-10)
    return reward, acc
expr_list_target= expr_list.copy()

state_id = 0

memory_dict = dict()
memory_dict['{}'.format(str(expr_list_target))] = dict()
memory_dict['{}'.format(str(expr_list_target))]['N'] = 0
memory_dict['{}'.format(str(expr_list_target))]['id'] = state_id
tree_dict = []
root = dict()
root['name'] = 'depth_{}_{}'.format(0,str(expr_list_target))
root['parent'] = 'null'
root['N'] = 0
root['W'] = 0
root['u'] = 0
root['Q'] = 0
root['P'] = 0

root['cost'] = 0

root['state'] = '{}'.format(str(expr_list_target))
root['children'] = []
tree_dict.append(root)
best_found_expr = []
for iter in range(1000):
        expr_list = expr_list_target.copy()
        memory_dict['{}'.format(str(expr_list_target))]['N'] += 1
        tree_dict[0]['N'] += 1
        node_names = ['depth_{}_{}'.format(0,str(expr_list_target))]
        branch_all=tree_dict[0]
        parent_exp = "tree_dict[0]"
        for depth in range(10):
            #try:

                input1_sample, input2_sample, expr_list_conv = cov_over_seq(expr_list, vec_to_symbol, symbol_to_vec)

                prob_list = []
                prob_list_all = []
                action_list = []
                action_index_list = []
                reward_list = []
                state_list = []
                cost_list = []
                child_index_list = []
                child_index = 0
                mcs_list = []
                u_list_all = []
                for conv_iter in range(len(input1_sample)):

                    state = [np.reshape(input1_sample[conv_iter],(1,6,1)), np.reshape(input2_sample[conv_iter],(1,6,1))]
                    state_l = [input1_sample[conv_iter], input2_sample[conv_iter]]

                    action, action_index, prob = agent.conv_act(state,state_l,vec_to_symbol,symbol_to_vec)

                    expr_list_conv[conv_iter].append(action)
                    reward, cost = get_reward_conv(action, truth_table_df, expr_list_conv[conv_iter],y)
                    if(cost == 1) and (depth >1):
                        best_found_expr.append(expr_list_conv[conv_iter])
                    reward_list.append(reward)
                    if len(eval(parent_exp)['children']) == 0:
                        Q = reward / (1 + 1e-120)
                        u = np.prod(prob) / 2  # N is not added yet

                        mcs_list.append(Q + u)
                        child_index_list.append(0)

                    else:
                        found = False
                        for ind, child in enumerate(eval(parent_exp)['children']):
                            if child['name'] == 'depth_{}_{}'.format(depth+1, str(expr_list_conv[conv_iter])):
                                    Q = (child['W'] + reward)/(1+ child['N'] + 1e-120)
                                    u = np.prod(prob) / (2 + child['N'])  # N is not added yet

                                    mcs_list.append(Q+u)
                                    child_index_list.append(ind)
                                    found = True
                        if not(found):
                                    Q = reward / (1 + 1e-120)
                                    u = np.prod(prob) / 2  # N is not added yet

                                    mcs_list.append(Q+u)
                                    child_index_list.append(len(eval(parent_exp)['children']))


                    cost_list.append(cost)
                    prob_list.append(prob)
                    prob_list_all.append(np.prod(prob))
                    action_list.append(action)
                    action_index_list.append(action_index)
                    state_list.append(state)
                if (len(reward_list)== 0):
                    break
                else:

                    winner_comb = np.argmax(mcs_list)
                    parent_exp = parent_exp + "['children']" + "[{}]".format(child_index_list[winner_comb])
                    expr_list = expr_list_conv[winner_comb].copy()
                    if agent.epsilon < 1:
                        node_names.append('depth_{}_{}'.format(depth+1,str(expr_list)))
                        branch = ''
                        parents = ["tree_dict[0]"]

                        for node in range(len(node_names[:-1])):
                            branch = branch + "['children']"
                            children_size = len(eval("tree_dict[0]" + branch))

                            if (node == len(node_names[:-2])) & (children_size == 0):  # parent of the leaf node that has no child
                                tempdict = dict()
                                tempdict['name'] = node_names[node+1]
                                tempdict['parent'] = node_names[node]
                                tempdict['N'] = 0
                                tempdict['cost'] = cost_list[winner_comb]
                                tempdict['W'] = reward_list[winner_comb]
                                tempdict['u'] = prob_list_all[winner_comb] / (1+tempdict['N'])
                                tempdict['Q'] = 0
                                tempdict['P'] = prob_list_all[winner_comb].astype(np.float64)


                                tempdict['state'] = node_names[node+1]
                                tempdict['children'] = []
                                #eval(parent)['W'] += reward_list[winner_comb]
                                for nod in parents: #rollout
                                    if(eval(nod)['W'] < reward_list[winner_comb]):

                                         eval(nod)['W'] += reward_list[winner_comb]

                                eval("tree_dict[0]" + branch).append(tempdict)
                            elif (node == len(node_names[:-2])) & (children_size != 0): # parent of the leaf node that has childs
                                children_list = eval("tree_dict[0]" + branch)
                                found = False
                                for ind, child in enumerate(children_list):
                                    if node_names[node+1] == child['name']:
                                        found = True

                                        eval("tree_dict[0]" + branch)[ind]["N"]+=1
                                        eval("tree_dict[0]" + branch)[ind]['cost'] = cost_list[winner_comb]
                                        eval("tree_dict[0]" + branch)[ind]['W'] = reward_list[winner_comb]
                                        eval("tree_dict[0]" + branch)[ind]['u'] = prob_list_all[winner_comb]/(1+eval("tree_dict[0]" + branch)[ind]["N"])
                                        eval("tree_dict[0]" + branch)[ind]['Q'] = eval("tree_dict[0]" + branch)[ind]['W'] / eval("tree_dict[0]" + branch)[ind]['N']
                                        eval("tree_dict[0]" + branch)[ind]['P'] =prob_list_all[winner_comb].astype(np.float64)
                                if not(found):
                                    tempdict = dict()
                                    tempdict['name'] = node_names[node + 1]
                                    tempdict['parent'] = node_names[node]
                                    tempdict['N'] = 0
                                    tempdict['cost'] = cost_list[winner_comb]
                                    tempdict['W'] = reward_list[winner_comb]
                                    tempdict['u'] = prob_list_all[winner_comb] / (1 + tempdict['N'])
                                    tempdict['Q'] = 0
                                    tempdict['P'] = prob_list_all[winner_comb].astype(np.float64)

                                    tempdict['state'] = node_names[node + 1]
                                    tempdict['children'] = []
                                    eval("tree_dict[0]" + branch).append(tempdict)
                                # roll out
                                for nod in parents:
                                    if(eval(nod)['W'] < reward_list[winner_comb]):
                                        eval(nod)['W'] += reward_list[winner_comb]

                            else:   # not a leaf node
                                children_list = eval("tree_dict[0]" + branch)
                                child_ind = "[{}]".format(0)
                                for ind, child in enumerate(children_list):
                                    if node_names[node+1] == child['name']:
                                        child_ind = "[{}]".format(ind)
                                branch = branch + child_ind
                                parents.append("tree_dict[0]" + branch)

                            with open('D3treelayout/treedata.json', 'w') as fp:
                                json.dump(tree_dict, fp)






                    if '{}'.format(str(expr_list)) in memory_dict.keys():
                        memory_dict['{}'.format(str(expr_list))]['N'] += 1
                        memory_dict['{}'.format(str(expr_list))]['W'] += reward_list[winner_comb]
                        memory_dict['{}'.format(str(expr_list))]['cost'] = cost_list[winner_comb]

                    else:
                        state_id += 1
                        memory_dict['{}'.format(str(expr_list))] = dict()
                        memory_dict['{}'.format(str(expr_list))]['N'] = 0
                        memory_dict['{}'.format(str(expr_list))]['W'] = reward_list[winner_comb]
                        memory_dict['{}'.format(str(expr_list))]['cost'] = cost_list[winner_comb]
                        branch = ''
                        for k in range(depth + 1):
                            branch = branch + "['children']"
                        parent = ''
                        for k in range(depth):
                            parent = parent + "['children']"

                        #tempdict = dict()
                        #tempdict['name'] = state_id
                        #tempdict['parent'] = eval("tree_dict[0]" + parent + "['name']")
                        #tempdict['N'] = 0
                        #tempdict['state'] = '{}'.format(str(expr_list))
                        #tempdict['children'] = []

   #                     eval("tree_dict[0]"+branch + "= []")
                        #eval("tree_dict[0]" + branch).append(tempdict)
                        memory_dict['{}'.format(str(expr_list))]['id'] = state_id
                        print('new state: ', state_id )

                    print(mcs_list)
                    print(expr_list)
                    print(memory_dict['{}'.format(str(expr_list))])
                    print('depth: ', depth,winner_comb, input1_sample[winner_comb], input2_sample[winner_comb])

                    print('reward', reward_list[winner_comb])
                    print('cost', cost_list[winner_comb])

                    agent.remember(state_list[winner_comb], action_list[winner_comb], action_index_list[winner_comb],reward_list[winner_comb], prob_list[winner_comb])
            #except:
             #       break
        loss,ac = agent.train(iter)
        print('iter: ',iter, 'loss: ', loss)
print(memory_dict)



