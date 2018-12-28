from valuemodel import ValueModel

valuemodel=ValueModel(max_len_rule=1000, max_len_seq=1000, NB_WORDS_rule=100,NB_WORDS_seq=100, emb_dim=510,intermediate_dim=50)
valuemodel.buildModel()