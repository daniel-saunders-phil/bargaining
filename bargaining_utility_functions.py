import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
import time
from scipy import stats

# functions to reformat data

def import_bayaka():
    '''import the bayaka data set, drop necessary columns, relabel columns,
    convert camps names into indices, z-score all the social capital information
    by camp'''
    
    file = "data_couples_Rfile.csv"

    data = pd.read_csv(file)
    bad_columns = ['sex','capitalratio','HH','Unnamed: 21','ageratio','restm','restfem','socializem',
                   'socializefem','leisurem','leisurefem','capitalfem','capitalm','sticksratio','Couple']

    data = data.drop(columns=bad_columns)
    
    data = data.rename(columns={'sticksm':'male_social',
                               "sticksfem":"female_social"})
    
    # convert camp names to numbers

    camp = []

    for i in data.camp.values:
        if i == "Longa":
            camp.append(0)
        elif i == "Masia":
            camp.append(1)
        else:
            camp.append(2)

    data['camp'] = camp

    # find z-score of differences

    male_z_social = np.array([])
    female_z_social = np.array([])

    for i in range(3):
        camp_1 = data[data.camp == i]
        all_sticks = np.concatenate((camp_1.male_social.values,camp_1.female_social.values))
        z_sticks = (all_sticks - np.mean(all_sticks)) / np.std(all_sticks)
        camp_male_z_social = z_sticks[:len(camp_1)]
        camp_female_z_social = z_sticks[len(camp_1):]
        male_z_social = np.concatenate((male_z_social,camp_male_z_social))
        female_z_social = np.concatenate((female_z_social,camp_female_z_social))

    data['male_z_social'] = male_z_social
    data['female_z_social'] = female_z_social

    return data

def import_agta():
    '''import the agta data set, drop necessary columns, relabel columns'''
    
    file = "Agta_Analysis2_couples_Rfile.csv"

    data = pd.read_csv(file)

    bad_columns = ['sex','capitalratio','capitaldiff','hh','agediff','leisurem','leisurefem','leisurediff_count',
                   'leisurediff_count.1','Unnamed: 22','leisurediff','Apropm','Apropfem','Gift.Z.m','Gift.Z.fem',
                   'num.shown_m','num.shown_fem','A_countm','A_countfem','prop.shared.diff',
                   'capitalm','capitalfem','num.live.with.m','num.live.with.fem','camp.N.m','camp.N.fem',
                   'prop.live.with.m','prop.live.with.fem','prop.live.with.diff','prop.live.with.ratio',
                   'Live.With.Z.m','Live.With.Z.fem',
                   'ageratio','leisureratio','prop.shared.ratio']
    
    data = data.rename(columns={
                               "prop.sharedm":"male_social",
                               "prop.sharedfem":"female_social",
                               "leisurem_count":'leisurem_counts',
                               "leisurefem_count":"leisuref_counts"
    })

    data = data.drop(columns=bad_columns)
    
    
    data['budget'] = data.leisurem_counts.values + data.leisuref_counts.values
    
    # convert camp names to numbers
    
    data['camp'] = np.array(pd.Categorical(data['camp']).codes,dtype=np.int64)
    data = data.sort_values('camp')

    # find z-score of differences

    male_z_social = np.array([])
    female_z_social = np.array([])

    for i in range(10):
        
        if i == 8:
            
            male_z_social = np.concatenate((male_z_social,np.zeros(4)))
            female_z_social = np.concatenate((female_z_social,np.zeros(4)))
            
        else:
            camp_i = data[data.camp == i]
            all_tokens = np.concatenate((camp_i.male_social.values,camp_i.female_social.values))
            z_tokens = (all_tokens - np.mean(all_tokens)) / np.std(all_tokens)
            camp_male_z_social = z_tokens[:len(camp_i)]
            camp_female_z_social = z_tokens[len(camp_i):]
            male_z_social = np.concatenate((male_z_social,camp_male_z_social))
            female_z_social = np.concatenate((female_z_social,camp_female_z_social))

    data['male_z_social'] = male_z_social
    data['female_z_social'] = female_z_social

    return data

def payoff_fun(p1,q1,dp,budget):
    '''Payoff function for the Nash bargaining game. Given
    a pair of actions (p1, g1) and disagreement point for player 1
    and a budget, it returns the payoff to player 1'''

    if p1 + q1 <= budget:
        return p1

    if p1 + q1 > budget:
        return dp * budget

def generate_options(budget):
    '''generates a list of possible actions for each player,
    given a budget'''
    
    # list of strategies
    
    options_p = list(range(1,budget))
    options_q = list(range(1,budget))
    
    return options_p,options_q

def generate_tables(options_p,options_q,budget,p_dp,q_dp):
    '''generates the payoff talbe for each player,
    given the set of options, the budget and the disagreement
    point for each player'''
    
    player_1_table = []
    
    for j in options_p:
        row = [payoff_fun(j,i,p_dp,budget) for i in options_q]
        player_1_table.append(row)
        
    player_2_table = []
    
    for j in options_q:
        row = [payoff_fun(j,i,q_dp,budget) for i in options_p]
        player_2_table.append(row)
        
    player_1_table,player_2_table = np.array(player_1_table),np.array(player_2_table)
    
    #print(player_1_table)
    #print(player_2_table)
        
    return player_1_table,player_2_table

def normalize_table(table):
    '''rescales the table so all numbers of non-negative'''
    
    # the darwin dynamics can only process normalized tables
    # where all payoffs are non-negative

    # find constant to shift the whole scale up

    constant = np.amin(table)
    constant = abs(constant)
    
    # build a matrix to shift the whole matrix up

    shifted_table = table + constant
    
    #normalize = np.sum(shifted_table)
    #new_table = shifted_table * (1 / normalize)
    
    new_table = shifted_table
    
    return new_table

def darwin_dynamics(p,q,p_table,q_table):
    '''solves the game by applying the darwin dynamics 120 times
    returns a probability distribution for each player over their set of 
    options.'''
    
    for k in range(120):
        
        # find expect payoffs - player 1
        
        weights = np.outer(p,q.T)
        eup = weights * p_table
        new_p = eup @ np.ones(len(p)).T
        normalize = np.sum(new_p)
        new_p = new_p * (1 / normalize)

        weights = np.outer(q,p.T)
        euq = weights * q_table
        new_q = euq @ np.ones(len(q)).T
        normalize = np.sum(new_q)
        new_q = new_q * (1 / normalize)

        p = new_p       
        q = new_q
    
        
    return p,q


def dynamic_solve(p_dp,q_dp,budget):
    '''main function to coordinate all the above. Returns
    a predicted division of the resource, given
    disagreement points for each player and the budget'''

    # list of strategies
    
    options_p,options_q = generate_options(budget)
    
    # probability distribution

    p = np.array([1/len(options_p)] * len(options_p))
    q = np.array([1/len(options_q)] * len(options_q))
    
    # generate payoff tables
    
    p_table,q_table = generate_tables(options_p,options_q,budget,p_dp,q_dp)
    p_table,q_table = normalize_table(p_table),normalize_table(q_table) 
    
    # run the darwin dynamics to identify the equilibrium
    
    p,q = darwin_dynamics(p,q,p_table,q_table)
    
    # extract best performing strategy
        
    p = list(p)
    q = list(q)
    
    main_p = max(p)
    main_q = max(q)
    
    best_p_strategy = p.index(main_p)
    best_q_strategy = q.index(main_q)
    
    p_freetime = int(options_p[best_p_strategy])
    q_freetime = int(options_q[best_q_strategy])
    
    # if players choose to utilize their disagreement points
    # then return nan (indicating that they should not be in
    # the dataset)
    # otherwise, return the predicted number of leisure hours
    
    if p_freetime + q_freetime > budget:
        return [np.nan,np.nan]
    
    else:
        return [p_freetime, q_freetime]
    
def make_data_row(m_stick_dist,f_stick_dist,a=0,b=0,noise=0):
    '''generate the synthetic data for one household.
    Returns a list containing the stick count for the male
    the stick count for the female, the number of leisure hours 
    for each and the overall budget of leisure hours for the household.
    parameters:
    
    m_stick_dist - a discrete probability distribution
    f_stick_dist - a discrete probability distribution
    a - the baseline disagreement point, default 0
    b - the effect of sticks on disagreement points, default 0
    noise - the magnitude of the noise term, default 0'''
    
    
    male_stick = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=m_stick_dist)
    female_stick = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=f_stick_dist)
      
    male_dp = b * male_stick + a
    female_dp = b * female_stick + a
    
    budget = np.random.choice([30,40,50,60,70,80,90])
    
    male_leisure, female_leisure = dynamic_solve(male_dp,female_dp,budget)
    
    # a simple mechanism to make noisy data.
    
    magnitude = stats.binom(budget,noise).rvs()
    direction = np.random.choice([-1,1])
    shift = magnitude * direction
    male_leisure = male_leisure - shift
    female_leisure = female_leisure + shift
    
    return [male_stick,female_stick,male_leisure,female_leisure,budget]

def synthetic_data(population_size=50,mean_sticks=3,a=0,b=0,noise=0):
    '''Returns a pandas DataFrame representing a full synthetic dataset'''

    m_stick_dist = stats.binom(9,p=mean_sticks/9).pmf([0,1,2,3,4,5,6,7,8,9])
    f_stick_dist = stats.binom(9,p=mean_sticks/9).pmf([0,1,2,3,4,5,6,7,8,9])
    half_pop_size = int(population_size / 2)

    # generate data
    
    data = [make_data_row(m_stick_dist,f_stick_dist,a,b,noise) for i in range(half_pop_size)]

    data = pd.DataFrame(data,columns=['male_social',
                                      'female_social',
                                      'leisurem_counts',
                                      'leisuref_counts',
                                       'budget'],index=range(half_pop_size))


    
    data = data.dropna(axis='index',how='any')
    
    all_sticks = np.concatenate((data.male_social.values,data.female_social.values))
    z_sticks = (all_sticks - np.mean(all_sticks)) / np.std(all_sticks)
    male_z_social = z_sticks[:len(data)]
    female_z_social = z_sticks[len(data):]
    
    data['male_z_social'] = male_z_social
    data['female_z_social'] = female_z_social
    data['difference_in_z_score'] = male_z_social - female_z_social
    
    return data

def objective(a,b,data):
    '''Return the log likelihood of the data, given a dataset
    the intercept parameter and the slope parameter'''
    
    likelihoods = []

    for i in range(len(data)):
        
        # extract variables from dataset for individual i
        
        ms = data.male_z_social.values[i]
        fs = data.female_z_social.values[i]
        budg = data.budget.values[i]
        observed_male = data.leisurem_counts.values[i]

        # guess the disagreement points

        pred_dp_m = b * ms + a
        pred_dp_f = b * fs + a

        # plug our two dps and the budget
        # into the darwin dynamics
        # then divide the number of male hours by the total
        # predicte number of hours.
        
        pred_male_hours,pred_female_hours = dynamic_solve(pred_dp_m,pred_dp_f,budg)

        # if the predicted split is impossible
        # return negative infinity
        
        if np.isnan(pred_male_hours) or np.isnan(pred_female_hours):
            likelihoods = float('-inf')
            break
            
        # the expected split is the number of male leisure hours
        # divided by the budget

        mp = pred_male_hours / budg

        # we compare the observed value to the predicted value
        # and compute a log probability score using the binomial distribution

        likelihood = stats.binom(budg,mp).logpmf(observed_male)

        # these probabilities are stored in the table

        likelihoods.append(likelihood)

    return np.sum(likelihoods)

def objective_overdispersion(a,b,d,data):
    '''Same as objective() but with a beta-binomial distribution'''
    
    likelihoods = []

    for i in range(len(data)):

        ms = data.male_z_social.values[i]
        fs = data.female_z_social.values[i]
        budg = data.budget.values[i]
        observed_male = data.leisurem_counts.values[i]

        pred_dp_m = b * ms + a
        pred_dp_f = b * fs + a
     
        pred_male_hours,pred_female_hours = dynamic_solve(pred_dp_m,pred_dp_f,budg)
        
        if np.isnan(pred_male_hours) or np.isnan(pred_female_hours):
            likelihoods = float('-inf')
            break

        mp = pred_male_hours / budg

        # same as objective() but we use betabinomial outcome distribution

        likelihood = stats.betabinom(n=budg,a=mp*d,b=(1-mp)*d).logpmf(observed_male)

        # these probabilities are stored in the table

        likelihoods.append(likelihood)
    
    return np.sum(likelihoods)

def grid_search(a_grid,b_grid,data,progress):
    
    '''Returns a two-dimensional grid of likelihoods as
    a pandas dataframe
    
    a_grid - a list or array of intercept values to search
    b_grid - a list or array of slope values to search
    data - the dataset
    progress - a boolean for whether to print out progress.
    The search often takes between 10 minutes to an hour so
    advisable to have this on the first time you run it'''
    
    start = time.time()
    
    likelihood_table = []
    
    acc = 1
    
    for a in a_grid:
        
        likelihood_row = [objective(a,b,data) for b in b_grid]
            
        likelihood_table.append(likelihood_row)
        
        if progress == True:
            print('progress:',np.round(100 * acc / len(a_grid),2),'%')
            acc = acc + 1
        
    like_dist = pd.DataFrame(likelihood_table,columns=np.round(b_grid,4),index=np.round(a_grid,4))
        
    finish = time.time()
    print('runtime:', finish-start)
    
    return like_dist

def grid_search_overdispersion(a_grid,b_grid,d,data,progress):
    
    import time
    
    start = time.time()
    
    likelihood_table = []
    
    acc = 1
    
    for a in a_grid:
        
        likelihood_row = [objective_overdispersion(a,b,d,data) for b in b_grid]
            
        likelihood_table.append(likelihood_row)
        
        if progress == True:
            print('progress:',np.round(100 * acc / len(a_grid),2),'%')
            acc = acc + 1
        
    like_dist = pd.DataFrame(likelihood_table,columns=np.round(b_grid,4),index=np.round(a_grid,4))
        
    finish = time.time()
    print('runtime:', finish-start)
    
    return like_dist