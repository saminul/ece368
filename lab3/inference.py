#ECE368 LAB3 - SUBMISSION BY SAMINUL ISLAM (1004511833)

import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    for i in range(num_time_steps):
        # TODO: Compute the forward messages
        forward_messages[i] = rover.Distribution({})  
        if i == 0:
            for z_n in prior_distribution:
                z0 = observations[i]
                forward_messages[i][z_n] = prior_distribution[z_n] * observation_model(z_n)[z0]
            forward_messages[i].renormalize()
        else:
            xy_n = observations[i]
            for z_n in all_possible_hidden_states:
                if xy_n == None:
                    cond = 1
                else:
                    cond = observation_model(z_n)[xy_n]
                if cond != 0:
                    prob = 0
                    for prev in (forward_messages[i-1]):
                        prob += forward_messages[i-1][prev] * transition_model(prev)[z_n]
                    if prob != 0:
                        forward_messages[i][z_n] = cond * prob
            forward_messages[i].renormalize()
        #print(forward_messages[i], "for i = ",i)

        # TODO: Compute the backward messages
        last = num_time_steps - 1 - i
        backward_messages[last] = rover.Distribution({})
        if i == 0:
            for z_n in all_possible_hidden_states:
                backward_messages[last][z_n] = 1
        else:
            xy_n1 = observations[last + 1]
            for z_n in all_possible_hidden_states:
                prob = 0
                for z_n1 in backward_messages[last+1]:
                    if xy_n1 == None:
                        cond = 1
                    else:
                        cond = observation_model(z_n1)[xy_n1]
                    prob += backward_messages[last + 1][z_n1] * transition_model(z_n)[z_n1] * cond
                if prob != 0:
                    backward_messages[last][z_n] = prob
            backward_messages[last].renormalize()

    # TODO: Compute the marginals

    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for z_n in all_possible_hidden_states:
            alpha = forward_messages[i][z_n]
            beta = backward_messages[i][z_n]
            if alpha * beta != 0:
                marginals[i][z_n] = (alpha * beta)
        marginals[i].renormalize()
    #print("Marginal for is",marginals[1], "for i = 1")
    return marginals


def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    w = [None] * num_time_steps
    max_path = [None] * num_time_steps

    #Initialization
    w[0] = rover.Distribution()
    for z_n in prior_distribution:
        z0 = observations[0]
        p_z0 = prior_distribution[z_n]
        p_given_z0 = observation_model(z_n)[z0]
        if p_z0 * p_given_z0 != 0:
            w[0][z_n] = np.log(p_z0*p_given_z0)
    
    for i in range(1,num_time_steps):
        w[i] = rover.Distribution()
        xy_n = observations[i]
        max_path[i] = dict()
        for z_n in all_possible_hidden_states:
            if xy_n == None:
                p_x_given_z = 1
            else:
                p_x_given_z = observation_model(z_n)[xy_n]
            if p_x_given_z != 0:
                max_val = -100000000.00
                for prev in (w[i-1]):
                    p_z_given_prev = transition_model(prev)[z_n]
                    if p_z_given_prev != 0:
                        temp_val = np.log(p_z_given_prev) + w[i-1][prev]
                        if temp_val > max_val:
                            max_path[i][z_n] = prev
                            max_val = temp_val
                w[i][z_n] = np.log(p_x_given_z) + max_val

    #back track to find estimaded states
    for i in range(num_time_steps):
        last = num_time_steps - 1 - i
        if i == 0:
            max_z = None
            max_val = -100000000.00
            for z_n in w[last]:
                temp = w[last][z_n]
                if temp > max_val:
                    max_val = temp
                    max_z = z_n
            estimated_hidden_states[last] = max_z
        else:
            estimated_hidden_states[last] = max_path[last+1][estimated_hidden_states[last+1]]
    
    return estimated_hidden_states

def Pe_fb(marginals,true_states):
    corr = 0.0
    for i in range(len(marginals)):
        if marginals[i].get_mode() == true_states[i]:
            corr += 1.0
    err = 1.0 - (corr/100.0)
    return err

def Pe_v(estimated_states,true_states):
    corr = 0.0
    for i in range(len(estimated_states)):
        if estimated_states[i] == true_states[i]:
            corr += 1.0
    err = 1.0 - (corr/100.0)
    return err
   

if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step], "for z",time_step)

    fb_error = Pe_fb(marginals,hidden_states)
    v_error = Pe_v(estimated_states,hidden_states)
    print(" Error due to forward-backward is",fb_error)
    print(" Error due to viterbi is",v_error)

    #check for invalid sequence
    # for time_step in range(1, num_time_steps):
    #     print(marginals[time_step].get_mode(), "for z",time_step)
        

    
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
