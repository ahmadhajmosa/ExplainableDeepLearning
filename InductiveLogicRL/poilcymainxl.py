from policyagent import PolicyAgent
from dataenv import DataEnv



if __name__ == "__main__":
    # initialize data environment and the agent

    env = DataEnv(data_len=100, n_variable=2)
    env.generate_env(rule=None)
    print('h')
    max_n_variable=env.max_n_variable
    input_lang_n_words=env.input_lang.n_words
    output_lang_n_words=env.output_lang.n_words
    agent = PolicyAgent(intermediate_dim=50,MAX_SEQUENCE_LENGTH=max_n_variable,NumCol=2,NB_WORDS=input_lang_n_words,NB_WORDS_OUT=output_lang_n_words)
    # Iterate the learning
    episodes=10000
    for e in range(episodes):
        state=env.reset()

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(10):
            # Decide action
            action, action_index, prob = agent.act(state,env.output_index2word,env.output_tokenizer,max_n_variable,env.input_tokenizer)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward,action,action_index,prob = env.step(action,action_index,prob)
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, action_index,reward, prob)

            agent.train(env.input_tokenizer,max_n_variable,env.output_index2word)

            # make next_state the new current state for the next frame.
            state = next_state

            #print("episode: {}/{}, score: {}".format(e, episodes, time_t))
        print(action)
        print(reward)
        #agent.replay(32,env.input_tokenizer,max_n_variable,env.output_index2word)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay