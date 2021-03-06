import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.metrics import classification_report

def normalize(u):
    Z = u.sum()
    return u/Z 

def normalize_by_rows(matrix):
    out_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        out_matrix[i,:] = matrix[i,:]/sum(matrix[i,:])

    return out_matrix
        
def forwards(symb_seq, pp, A, B):

    K = A.shape[0]
    T = len(symb_seq)
    alpha = np.zeros((K,T))
    psit_alpha = B[:, symb_seq[0]]
    alpha[:,0] = normalize(psit_alpha*pp)
    for t in range(1,T):
        psit_alpha = B[:, symb_seq[t]]
        alpha[:,t] = normalize( (A.T@alpha[:,t-1])*psit_alpha )
     
    return np.argmax(alpha, axis=0), alpha

def forwards_backwards(symb_seq, pp, A, B):
 
    K = A.shape[0]
    T = len(symb_seq)
    alpha = np.zeros((K,T))
    beta = np.ones((K,T))
    gamma = np.zeros((K,T))
    
    psit_alpha = B[:, symb_seq[0]]
    psit_beta = B[:, symb_seq[T-1]]

    #pdb.set_trace()
    alpha[:,0] = normalize( psit_alpha*pp )
    beta[:,T-1] = normalize( psit_beta*pp )

    for t in range(1,T):
        #tb = T-t-1
        psit_alpha = B[:, symb_seq[t]]
        #psit_beta = B[:, symb_seq[tb]]
        alpha[:,t] = normalize( (A.T@alpha[:,t-1])*psit_alpha )
        #beta[:,tb] = normalize( A@(psit_beta*beta[:,tb+1]) )

    for tb in reversed(range(T-1)):
        psit_beta = B[:, symb_seq[tb+1]]
        beta[:,tb] = normalize( A@(psit_beta*beta[:,tb+1]) )
        
    for t in range(T):
        gamma[:,t] = normalize(alpha[:,t]*beta[:,t])

    epsilon = get_epsilon_two_slice_marginal(symb_seq, A, B, alpha, beta)
    return np.argmax(gamma, axis=0), gamma, epsilon

def get_epsilon_two_slice_marginal(symb_seq, A, B, alpha, beta):
    '''
    ξ_(t,t+1) ∝ Ψ .* (α_t (φ_(t+1) .* β_(t+1))^T ) 
    
    Ψ -- transition matrix -- A
    φ_(t+1) = p(X_(t+1)|z_(t+1)) =  emission prob vector -- b_j
    ''' 
    K = A.shape[0]
    T = len(symb_seq)

    epsilon = np.zeros((T-1,K,K))

    for t in range(T-1):
        epsilon[t, :, :] = normalize(A * np.outer(alpha[:,t], B[:,symb_seq[t+1]]*beta[:,t+1] ) )

    return epsilon

def show_prior_vector(matrix):
    matrix = pd.DataFrame(matrix).round(3)
    print(f'The prior vector:\n{matrix}\n')
    
def show_emission_matrix(matrix):
    matrix = pd.DataFrame(matrix).round(3)
    print(f'The emission matrix:\n{matrix}\n')
    
def show_transition_matrix(matrix):
    matrix = pd.DataFrame(matrix).round(3)
    print(f'The transition matrix:\n{matrix}\n')
    

def get_transition_matrix_with_training(state_seq):

    state_num = len(set(state_seq))
    N = np.zeros((state_num,state_num))
    states = list(set(state_seq))
    for j, state_j in enumerate(states):
        for k, state_k in enumerate(states):
            #count number of each possible transition
            for t in range(len(state_seq)-1):
                if state_seq[t]==state_j and state_seq[t+1]==state_k:
                    N[j,k] += 1

    #normalize counts to probabilites
    trans_matrix = normalize_by_rows(N)
    #show_transition_matrix(trans_matrix)
    return trans_matrix       

def get_emission_matrix_with_training(state_seq, symb_seq):

    state_num = len(set(state_seq))
    symb_numb = len(set(symb_seq))
    N = np.zeros((state_num,symb_numb))
    states = list(set(state_seq))
    symbols = list(set(symb_seq))
    for j, state_j in enumerate(states):
        for l, symbol_l in enumerate(symbols):
            #count number of each possible transition
            for t in range(len(state_seq)-1):
                if state_seq[t]==state_j and symb_seq[t]==symbol_l:
                    N[j,l] += 1
                pass
    #normalize counts to probabilites
    N = N.T / np.sum(N,axis=1)
    #show_emission_matrix(N.T)
    return N.T   

def get_priors_stupid_heuristic(state_seq):

    states = list(set(state_seq))
    priors = np.zeros(len(states))
    for i, state in enumerate(states):
        #put prob=1 to first symbol in the sequence
        if state==state_seq[0]:
            priors[i] = 1

    #print(f'The estimate of priors:\n{priors}\n')
    return priors


def init_BW_by_training(state_seq, symb_seq): 
    trans_matrix = get_transition_matrix_with_training(state_seq)
    emiss_matrix = get_emission_matrix_with_training(state_seq, symb_seq)
    priors = get_priors_stupid_heuristic(state_seq)
    
    show_prior_vector(priors)
    show_transition_matrix(trans_matrix)
    show_emission_matrix(emiss_matrix)
    return priors, emiss_matrix, trans_matrix


def init_BW_randomly(symb_seq, states_seq):
    
    states_numb = len(set(states_seq))
    symb_numb = len(set(symb_seq))
    
    trans_matrix = normalize_by_rows(np.random.rand(states_numb, states_numb))
    emiss_matrix = normalize_by_rows(np.random.rand(states_numb, symb_numb))
    priors = normalize(np.random.rand(states_numb))
    
    show_prior_vector(priors)
    show_transition_matrix(trans_matrix)
    show_emission_matrix(emiss_matrix)
    return priors, emiss_matrix, trans_matrix

def get_emission_matrix_update(symb_seq, gamma_t, matrix_shape):
    
    #normalization vector
    E_Nj = gamma_t.sum(axis=1)
    
    E_Mjl = np.zeros(matrix_shape)
    emiss_matrix = np.zeros(matrix_shape)
    
    for t, symbol in enumerate(symb_seq):
        E_Mjl[:, symbol] += gamma_t[:, t]
        
    for column in range(matrix_shape[1]):
        emiss_matrix[:, column] =  E_Mjl[:, column]/E_Nj

    return emiss_matrix

def get_priors_update(gamma_t):

    return gamma_t[:,0]

def get_transition_matrix_update(gamma_t, epsilon_t):

    epsilon = epsilon_t.sum(axis=0)
    gamma = gamma_t.sum(axis=1)
    states_number = len(gamma)
    trans_matrix = np.zeros((states_number, states_number)) 
    
    for i in range(states_number):
        trans_matrix[i,:] = epsilon[i,:]/gamma[i]
    
    return trans_matrix

def estimate_model_with_Baum_Welch(symb_seq, states_to_init, training_init = 1,
                                   init_sample_number = 100, iter_lim = 100, threshold = 0.001):

    if training_init:
        print(f'Initiating BW algorithm with #{init_sample_number} samples\n---------------')
        priors, emiss_matrix, trans_matrix = init_BW_by_training(states_to_init[:init_sample_number],
                                                            symb_seq[:init_sample_number])
    else:
        #print(f'Initiating BW algorithm with uniformly filled probabilities\n---------------')
        #priors, emiss_matrix, trans_matrix = init_BW_uniformly(symb_seq, states_to_init)
        print(f'Initiating BW algorithm with randomly filled probabilities\n---------------')
        priors, emiss_matrix, trans_matrix = init_BW_randomly(symb_seq, states_to_init)
    
    iter = 0
    max_diff = 1
    while (iter < iter_lim and max_diff > threshold):
        
        old = {'priors': priors, 
                      'trans' : trans_matrix,
                      'emiss' : emiss_matrix}
        
        iter += 1
        
        states, gamma_t, epsilon_t = forwards_backwards(symb_seq, priors, trans_matrix,emiss_matrix)

        priors = get_priors_update(gamma_t)
        trans_matrix = get_transition_matrix_update(gamma_t, epsilon_t)

        emiss_matrix = get_emission_matrix_update(symb_seq, gamma_t, emiss_matrix.shape)
        
        max_diff = max([np.max(abs(old['priors']- priors)),
                           np.max(abs(old['trans']- trans_matrix)), 
                           np.max(abs(old['emiss']- emiss_matrix))])
    

    
    
    print(f'BW after {iter} iterations yielded to ...')
    show_prior_vector(priors)
    show_transition_matrix(trans_matrix)
    show_emission_matrix(emiss_matrix)
   
    return priors, trans_matrix, emiss_matrix

def print_hmm_model(model):
    show_prior_vector(model.startprob_)
    show_transition_matrix(model.transmat_)
    show_emission_matrix(model.emissionprob_)

def set_hmm_model_example():
        
    model = hmm.MultinomialHMM(n_components=2)
    #state_names = ['usual','loaded']
    model.startprob_ = np.array([0.2, 0.8])
    
    model.transmat_ = np.array([[0.9,0.1],
                                [0.4,0.6]])
    
    model.emissionprob_ = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                                    [5/10, 1/10, 1/10, 1/10, 1/10, 1/10]])
    
    print('The model has been set with the following parameters:')
    print_hmm_model(model)
    
    return model
    
def main():
    
    pd.set_option('display.max_columns', 6)

    model = set_hmm_model_example()

    #generate samples from the model
    sample_number = 2000
    gener_seq, gener_states = model.sample(sample_number)
    gener_seq = gener_seq.ravel()
    
    mode_list = '''Choose the mode:
    1. Estimate the states given model and symbol sequence (Forwards and Forward-Backwards)
    2. Estimate the model given states and symbol sequence
    3. Estimate the model given the symbols (Baum-Welch)
    '''
    print(mode_list)
    
    mode = ''
    while (mode not in ['1','2','3'] and mode!='q'):
        mode = input('Enter the mode number (1,2,3) or "q":')

    if mode=='q':
        exit('Exiting...')          
        
    elif mode=='1':
    
        forw_states, forw_prob = forwards(gener_seq, model.startprob_, model.transmat_,
                                            model.emissionprob_)
        
        fb_states, fb_prob, epsilon = forwards_backwards(gener_seq, model.startprob_, model.transmat_,
                                            model.emissionprob_)
        
        
        '''
        The precision is the ratio tp / (tp + fp) where tp is the number of true 
        positives and fp the number of false positives. The precision is intuitively 
        the ability of the classifier not to label as positive a sample that is negative.
        
        The recall is the ratio tp / (tp + fn) where tp is the number of true 
        positives and fn the number of false negatives. The recall is intuitively 
        the ability of the classifier to find all the positive samples.
        
        The F-beta score can be interpreted as a weighted harmonic mean of the 
        precision and recall, where an F-beta score reaches its best value at 1
        and worst score at 0.
        '''
    
        accuracy_forward = sum(gener_states==forw_states)/sample_number*100
        accuracy_fb = sum(gener_states==fb_states)/sample_number*100
    
        print('Forwards:\nAccuracy {}\n'.format(accuracy_forward),classification_report(gener_states, forw_states))
    
        print('Forward-Backwards:\nAccuracy {}\n'.format(accuracy_fb),classification_report(gener_states, fb_states))
  
    elif mode=='2':
        
        get_transition_matrix_with_training(gener_states)
        
        get_emission_matrix_with_training(gener_states, gener_seq)
        
    elif mode=='3':
        
        estimate_model_with_Baum_Welch(gener_seq, gener_states)

    
if __name__ == "__main__":
        
    main()