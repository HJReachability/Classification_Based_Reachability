# -*-: coding: utf-8 -*-
#  Copyright (c) 2018, The Regents of the University of California (Regents).
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo   ( vrubies@eecs.berkeley.edu )

import numpy as np
import tensorflow as tf
import itertools
import Utils
import pickle
import time

# System Parameters    
num_ac = 3                                                      #number of actions our system has
dist_ac = 3                                                     #number of actions available to the disturbance
layers = [6,40,40,2**num_ac]                                    #neural network architecture
t_hor = -0.5                                                    #time horizon for the problem
dt = 0.1                                                        #time discretization parameter

max_list = [0.1,0.1,11.81]                                      #upper bounds of the control
min_list = [-0.1,-0.1,7.81]                                     #lower bounds of the control

max_list_ = [0.5,0.5,0.5]                                       #upper bounds of the disturbance
min_list_ = [-0.5,-0.5,-0.5]                                    #lower bounds of the disturbance

# Learning Parameters
nrolls = 200000                                                 #number of points to sample in R^n
bts = 5000                                                      #batch size for gradient descent
nunu = 0.001                                                    #learning rate
mom = 0.95                                                      #momentum constant                                
renew = 1000                                                    #number of gradient steps before resampling

# Count number of parameters in Neural Net
print 'Starting worker-'
nofparams = 0
for i in xrange(len(layers)-1):
    nofparams += layers[i]*layers[i+1] + layers[i+1]
print 'Number of Params is: ' + str(nofparams)

####### Tensorflow specific

# Create a Neural Network classifier for the control and the disturbance
iters = int(np.abs(t_hor)/dt)*renew + 1 
states,y,output,L,_,_,_,_ = Utils.MakeNeuralNet("Control",False,layers)
states_,y_,output_,L_,_,_,_,_ = Utils.MakeNeuralNet("Disturbance",False,layers)

# Useful function to convert to one-hot encoding
hot_input = tf.placeholder(tf.int64,shape=(None)) 
make_hot = tf.one_hot(hot_input, 2**num_ac, on_value=1, off_value=0)

# Learning rate placeholder
nu = tf.placeholder(tf.float32, shape=[])

# Measure accuracy of the output
temp1 = tf.argmax(output,dimension=1)
temp2 = tf.argmax(y,dimension=1)
temp3 = tf.equal(temp1,temp2)
accuracy = tf.reduce_mean(tf.cast(temp3, tf.float32))
temp1_ = tf.argmax(output_,dimension=1)
temp2_ = tf.argmax(y_,dimension=1)
temp3_ = tf.equal(temp1_,temp2_)
accuracy_ = tf.reduce_mean(tf.cast(temp3_, tf.float32))

# Get references to neural net variables
C_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Control')
D_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Disturbance')

# Optimizer definition
train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L)
train_step_ = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L_)

# Graph Initialization
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

############################

# Generate list of all possible control action combinations
# TODO: Replace with newer version that uses binary classifiers
comb = list(itertools.product([-1,1], repeat=num_ac))
true_ac_list = []
for i in range(len(comb)):
    ac_tuple = comb[i]
    ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]
    true_ac_list.append(ac_list)

# Generate list of all possible disturbance action combinations       
comb_ = list(itertools.product([-1,1], repeat=dist_ac))
true_ac_list_ = []
for i in range(len(comb_)):
    ac_tuple_ = comb_[i]
    ac_list_ = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple_,min_list_,max_list_)]
    true_ac_list_.append(ac_list_)       

# Function that takes in two lists of weights and computes the input and targets for the next
# time step.
def FindNewOptimalActions(ALL_x,F_PI=[], F_PI_=[], subSamples=1):

    #Save current parameters for control and disturbance
    current_params = sess.run(C_func_vars)
    current_params_ = sess.run(D_func_vars)

    #Get the set of states I can evolve to for each control+disturbance combination
    next_states_ = []
    for k in range((len(comb))):
        next_states = []
        opt_a = np.asarray(true_ac_list[k])*np.ones([ALL_x.shape[0],1])
        for i in range(len(comb_)):
            opt_b = np.asarray(true_ac_list_[i])*np.ones([ALL_x.shape[0],1])
            for _ in range(subSamples): 
                Snx = Utils.RK4(ALL_x,dt/float(subSamples),opt_a,opt_b)
            next_states.append(Snx)
        next_states_.append(np.concatenate(next_states,axis=0))
    next_states_ = np.concatenate(next_states_,axis=0)
    #Get the value for every possible successor state
    values = Utils.V_0(next_states_[:,[0,1,2]])
    
    #Use previously computed policies to evolve system until t=0 
    for params,params_ in zip(F_PI,F_PI_):
        for ind in range(len(params)):
            sess.run(C_func_vars[ind].assign(params[ind]))
        for ind in range(len(params_)):
            sess.run(D_func_vars[ind].assign(params_[ind]))            

        tmp = Utils.PreProcess(next_states_)
        hots = sess.run(output,{states:tmp})
        opt_a = Utils.Hot_to_Cold(hots,true_ac_list)   
        hots = sess.run(output_,{states_:tmp})
        opt_b = Utils.Hot_to_Cold(hots,true_ac_list_)            
        for _ in range(subSamples):
            next_states_ = Utils.RK4(next_states_,dt/float(subSamples),opt_a,opt_b)
            values = np.max((values,Utils.V_0(next_states_[:,[0,1,2]])),axis=0)
    
    #After running all trajectories, now compare the values obtained for each
    #action that was taken at the beginning
    values_ = values
    pre_compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T
    final_v = []
    final_v_ = []
    per = len(comb)
    for k in range(len(comb_)):
        final_v.append(np.argmax(pre_compare_vals_[:,k*per:(k+1)*per,None],axis=1))
        final_v_.append(np.max(pre_compare_vals_[:,k*per:(k+1)*per,None],axis=1))
    finalF = np.concatenate(final_v_,axis=1)
    index_best_a = np.argmin(finalF,axis=1)
    finalF_ = np.concatenate(final_v,axis=1)
    index_best_b_ = np.array([finalF_[k,index_best_a[k]] for k in range(len(index_best_a))])
    
    #Reset weights to be the same as before (before we entered the function)
    for ind in range(len(current_params)):
        sess.run(C_func_vars[ind].assign(current_params[ind]))
    for ind in range(len(current_params_)):
        sess.run(D_func_vars[ind].assign(current_params_[ind]))
        
    return sess.run(make_hot,{hot_input:index_best_a}),sess.run(make_hot,{hot_input:index_best_b_})


# *****************************************************************************
# ============================= MAIN LOOP ====================================
# *****************************************************************************
ALL_PI = []
ALL_PI_= []
current_time = -dt

train_ac = []
test_ac = []
for i in xrange(iters):
    
    if(np.mod(i,renew) == 0 and i is not 0):       
        
        ALL_PI.insert(0,sess.run(C_func_vars))
        ALL_PI_.insert(0,sess.run(D_func_vars)) 

        k = 0
        t = time.time()
        ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]))
        ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0
        PI_c,PI_d = FindNewOptimalActions(ALL_x,ALL_PI,ALL_PI_,subSamples=1)
        pre_ALL_x = Utils.PreProcess(ALL_x)
        elapsed = time.time() - t
        print("Compute Data Time = "+str(elapsed))
        
        ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]))
        ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
        PI_c_,PI_d_ = FindNewOptimalActions(ALL_x_,ALL_PI,ALL_PI_,subSamples=1)
        pre_ALL_x_ = Utils.PreProcess(ALL_x_)
        
        current_time = current_time - dt
        print("Learning policies for time step " + str(current_time))
        
    elif(np.mod(i,renew) == 0 and i is 0):

        t = time.time()
        ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]))
        ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0                  
        PI_c,PI_d = FindNewOptimalActions(ALL_x,F_PI=[],F_PI_=[],subSamples=1)
        pre_ALL_x = Utils.PreProcess(ALL_x)
        elapsed = time.time() - t
        print("Compute Data Time = "+str(elapsed))
        
        ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]))
        ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
        PI_c_,PI_d_ = FindNewOptimalActions(ALL_x_,F_PI=[],F_PI_=[],subSamples=1)
        pre_ALL_x_ = Utils.PreProcess(ALL_x_)
        print("Learning policies for time step " + str(current_time))
        

    # |||||||||||| ----  PRINT ----- |||||||||||| 

    if(np.mod(i,50) == 0):
        
        train_acc = sess.run(accuracy,{states:pre_ALL_x,y:PI_c})
        test_acc = sess.run(accuracy,{states:pre_ALL_x_,y:PI_c_})
        train_acc_ = sess.run(accuracy_,{states_:pre_ALL_x,y_:PI_d})
        test_acc_ = sess.run(accuracy_,{states_:pre_ALL_x_,y_:PI_d_})
        train_ac.append(train_acc) 
        train_ac.append(train_acc_)
        test_ac.append(test_acc) 
        test_ac.append(test_acc_)
        
        print str(i) + ") control | TR_ACC = " + str(train_acc) + " | TE_ACC = " + str(test_acc) + " | Learning Rate = " + str(nunu)
        print str(i) + ") disturb | TR_ACC = " + str(train_acc_) + " | TE_ACC = " + str(test_acc_) + " | Learning Rate = " + str(nunu)

    tmp = np.random.randint(len(ALL_x), size=bts)
    sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI_c[tmp],nu:nunu})
    sess.run(train_step_, feed_dict={states_:pre_ALL_x[tmp],y_:PI_d[tmp],nu:nunu})

pickle.dump([ALL_PI,ALL_PI_],open( "policies.pkl", "wb" ))
pickle.dump([train_ac,test_ac],open( "train_logs.pkl", "wb" ))
