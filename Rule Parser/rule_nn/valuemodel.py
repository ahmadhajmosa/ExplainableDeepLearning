from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import keras


class ValueModel():
    def __init__(self, max_len_rule, max_len_seq, NB_WORDS_rule, NB_WORDS_seq,emb_dim,intermediate_dim):
        super(ValueModel, self).__init__()
        self.max_len_rule=max_len_rule
        self.max_len_seq=max_len_seq
        self.NB_WORDS_rule=NB_WORDS_rule
        self.NB_WORDS_seq=NB_WORDS_seq

        self.emb_dim=emb_dim
        self.intermediate_dim=intermediate_dim



    def buildModel(self):

        act = ELU()

        #rule network
        x_rule = Input(batch_shape=(None, self.max_len_rule),name='rule_input')
        x_rule_embed = Embedding(self.NB_WORDS_rule, self.emb_dim,
                                    input_length=self.max_len_rule, trainable=True)(x_rule)

        h_rule = Bidirectional(LSTM(self.intermediate_dim, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(x_rule_embed)

        h_rule = Dropout(0.2)(h_rule)
        h_rule = Dense(self.intermediate_dim, activation='linear')(h_rule)
        h_rule =act(h_rule)

        #seq network

        x_seq = Input(batch_shape=(None, self.max_len_seq),name='seq_input')
        x_seq_embed = Embedding(self.NB_WORDS_seq, self.emb_dim,
                                    input_length=self.max_len_seq, trainable=True)(x_seq)

        h_seq = Bidirectional(LSTM(self.intermediate_dim, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(x_seq_embed)

        h_seq = Dropout(0.2)(h_seq)
        h_seq = Dense(self.intermediate_dim, activation='linear')(h_seq)
        h_seq =act(h_seq)

        concatenated = keras.layers.concatenate([h_rule, h_seq])
        concatenated = Dense(self.intermediate_dim, activation='relu')(concatenated)


        #repeated_context = RepeatVector(self.max_len_rule)(concatenated)
        decoder_h = LSTM(self.max_len_rule, return_sequences=True, recurrent_dropout=0.2)(concatenated)
        out = TimeDistributed(Dense(self.NB_WORDS_rule, activation='linear'))(decoder_h)



        out = Dense(1, activation='sigmoid')(out)

        self.value_model = Model([x_rule, x_seq], out)

        print(self.value_model.summary())

        from keras.utils import plot_model
        plot_model(self.value_model, to_file='valueModel.png')
