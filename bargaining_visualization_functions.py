import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
import seaborn as sns
import time
from scipy import stats

def analyze_grid(mle):
    
    a_grid = mle.index.values
    b_grid = [float(i) for i in mle.columns.values]
    
    ticks_a = np.linspace(0,len(a_grid),10)
    ticks_b = np.linspace(0,len(b_grid),10)
    
    labels_a = np.round(np.linspace(min(a_grid),max(a_grid),10),2)
    labels_b = np.round(np.linspace(min(b_grid),max(b_grid),10),2)
    
    mle = mle.replace(to_replace=float('-inf'),value=np.nan)
    
    # raw likelihood distribution
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,constrained_layout=True)
    
    sns.heatmap(mle,ax=ax1) #,xticklabels=labels_b,yticklabels=labels_a)
    ax1.set_title('log likelihood distribution')
    #labels_b = np.round(np.linspace(min(b_grid),max(b_grid),10),2)
    ax1.xaxis.set_ticks(ticks_b)
    ax1.xaxis.set_ticklabels(labels_b)
    ax1.yaxis.set_ticks(ticks_a)
    ax1.yaxis.set_ticklabels(labels_a)
    
    # normalized likelihood distribution

    post = np.exp(mle)
    post = post.replace(to_replace=np.nan,value=0)
    post = post / post.values.sum()

    sns.heatmap(post,ax=ax2)
    ax2.set_title('natural likelihood distribution')
    ax2.xaxis.set_ticks(ticks_b)
    ax2.xaxis.set_ticklabels(labels_b)
    ax2.yaxis.set_ticks(ticks_a)
    ax2.yaxis.set_ticklabels(labels_a)
    
    #
    
    array_post = np.array(post)
    flat = array_post.flatten()
    
    peak = np.amax(flat)
    peak_index = np.where(flat == peak)
    
    i,j = np.unravel_index(peak_index,post.shape)
    print(i,j)

    a_s = []
    b_s = []
    for i in range(10000):

        sample_index = np.random.choice(a=flat.size,p=flat)

        i,j = np.unravel_index(sample_index,post.shape)

        a_s.append(post.index.values[i]),
        b_s.append(float(post.columns.values[j]))
        
    a_s_upper = np.percentile(a_s,95)
    a_s_lower = np.percentile(a_s,5)
    b_s_upper = np.percentile(b_s,95)
    b_s_lower = np.percentile(b_s,5)
    
    acc = 0
    for i in b_s:
        if i <= 0:
            acc = acc + 1
    slope_pvalue = acc / len(b_s)
    
    ax3.hist(a_s,bins=20,density=True,histtype='step')
    print('a - 95% credible interval:','[',a_s_lower, a_s_upper,']')
    ax3.set_title('credibility distribution over a')
        
    ax4.hist(b_s,bins=20,density=True,histtype='step')
    print('b - 95% credible interval:','[',b_s_lower, b_s_upper,']')
    print('b - p value:',slope_pvalue)
    ax4.set_title('credibility distribution over b')
    
    return f

def estimated_leisure_distribution(data,a,b):
    
    male_leisure_estimate = []
    female_leisure_estimate = []
    dp_male = []
    dp_female = []

    for i in range(len(data)):
        # ms = data.male_relational.values[i]
        # fs = data.female_relational.values[i]
        ms = data.male_z_relational.values[i]
        fs = data.female_z_relational.values[i]
        budg = data.budget.values[i]

        # guess the disagreement points

        pred_dp_m = b * ms + a
        pred_dp_f = b * fs + a
        dp_male.append(pred_dp_m)
        dp_female.append(pred_dp_f)

        # plug our two dps and the budget
        # into the darwin dynamics
        # then divide the number of male hours by the total
        # predicte number of hours.

        pred_male_hours,pred_female_hours = dynamic_solve(pred_dp_m,
                                                          pred_dp_f,
                                                          budg)

        male_leisure_estimate.append(pred_male_hours)
        female_leisure_estimate.append(pred_female_hours)

    data['male_leisure_estimate'] = male_leisure_estimate
    data['female_leisure_estimate'] = female_leisure_estimate
    data['male_dp_estimate'] = dp_male
    data['female_dp_estimate'] = dp_female

    return data

def visualize_predictive_dist(data):
    
    ob_male_leisure = data['leisurem_counts'].values
    ob_female_leisure = data['leisuref_counts'].values
    
    pred_male = data['male_leisure_estimate'].values
    fairness = data['budget'].values / 2
    
    household_id = data.index + 1

    plt.bar(household_id,ob_male_leisure,label='male leisure (obs)')
    plt.bar(household_id,ob_female_leisure,bottom=ob_male_leisure,label='female leisure (obs)')
    plt.plot(household_id,pred_male,'o',color='black',label='predicted split')
    plt.plot(household_id,fairness,'x',color='black',label='equal split')
    plt.ylabel('Leisure hours')
    plt.xlabel('Couple')
    
    ticks_b = list(range(int(len(data)/5),len(data)+1,int(len(data)/5)))
    plt.xticks(ticks_b)
    plt.title('Predicted vs observed division of leisure budget')
    plt.legend()
    
def visualize_predictive_dist_residuals(data):
    
    budget = data['budget'].values
    
    ob_male_leisure = data['leisurem_counts'].values / budget
    ob_female_leisure = data['leisuref_counts'].values / budget
    
    pred_male = data['male_leisure_estimate'].values / budget
    
    fairness = [0.5] * len(data)
    
    household_id = data.index + 1

    plt.bar(household_id,ob_male_leisure,label='male leisure (obs)')
    plt.bar(household_id,ob_female_leisure,bottom=ob_male_leisure,label='female leisure (obs)')
    plt.plot(household_id,pred_male,'o',color='black',label='predicted split')
    plt.plot(household_id,fairness,'--',color='black',label='equal split',alpha=0.5)
    plt.ylabel('Leisure hours')
    plt.xlabel('Couple')
    ticks_b = list(range(int(len(data)/5),len(data)+1,int(len(data)/5)))
    plt.xticks(ticks_b)
    plt.title('Predicted vs observed divisions of leisure budget')
    plt.legend()
    
    
def analyze_grid_simple(mle):
    
    a_grid = mle.index.values
    b_grid = [float(i) for i in mle.columns.values]
    
    ticks_a = np.linspace(0,len(a_grid),10)
    ticks_b = np.linspace(0,len(b_grid),10)
    
    labels_a = np.round(np.linspace(min(a_grid),max(a_grid),10),2)
    labels_b = np.round(np.linspace(min(b_grid),max(b_grid),10),2)
    
    mle = mle.replace(to_replace=float('-inf'),value=np.nan)
    
    # raw likelihood distribution
    
    # normalized likelihood distribution

    post = np.exp(mle)
    post = post.replace(to_replace=np.nan,value=0)
    post = post / post.values.sum()

    plot = sns.heatmap(post,cbar_kws = {'label':"low likelihood <---> high likelihood"})
    plt.title('Joint likelihood distribution for parameters of the bargaining model')
    plt.xticks(ticks_b,labels=labels_b)
    plt.xlabel("Slope parameter")
    plt.ylabel("Intercept parameter")
    plt.yticks(ticks_a,labels=labels_a)
    
    return plot