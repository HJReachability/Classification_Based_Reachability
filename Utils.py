# -*- coding: utf-8 -*-
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

import tensorflow as tf
import numpy as np

#Args
g = 9.81;


def lrelu(x):
  return tf.nn.relu(x) - 0.01*tf.nn.relu(-x)

# TODO: Update to newer version that uses binary classifiers
def MakeNeuralNet(scope=None, reuse=None, lsizes = None):
    with tf.variable_scope(scope, reuse=reuse):
        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
        y = tf.placeholder(tf.float32,shape=(None,lsizes[-1]),name="y");   
    
        lw = [];
        lb = [];
        l = [];
        reg = 0.0;
        for i in xrange(len(lsizes) - 1):
            lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
            lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
            reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
            
        l.append(lrelu(tf.add(tf.matmul(states,lw[0]), lb[0])))
        for i in xrange(len(lw)-2):
            l.append(lrelu(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1])));
        
        last_ba = tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end");
        
        l.append(tf.nn.softmax(last_ba));
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_ba,labels=y)
        L = tf.reduce_mean(cross_entropy)
        
        PI = l[-1];
        
    return states,y,PI,L,l,lb,reg,cross_entropy

def V_0(x):
    return np.linalg.norm(x,axis=1,keepdims=True) - 1.0

#def p_corr(ALL_x):
#    ALL_x = np.mod(ALL_x,2.0*np.pi);
#    return ALL_x;

# Replace this function with your own dynamics
def F(ALL_x,opt_a,opt_b):
   col1 = ALL_x[:,3,None] - opt_b[:,0,None]
   col2 = ALL_x[:,4,None] - opt_b[:,1,None]
   col3 = ALL_x[:,5,None] - opt_b[:,2,None]
   col4 = g*opt_a[:,0,None]
   col5 = -g*opt_a[:,1,None]
   col6 = opt_a[:,2,None] - g
   
   return np.concatenate((col1,col2,col3,col4,col5,col6),axis=1);

def RK4(ALL_x,dtt,opt_a,opt_b): #Runge Kutta 4

    k1 = F(ALL_x,opt_a,opt_b);
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
    #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

    k2 = F(ALL_tmp,opt_a,opt_b);
    ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
    #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

    k3 = F(ALL_tmp,opt_a,opt_b); 
    ALL_tmp = ALL_x + np.multiply(dtt,k3);
    #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

    k4 = F(ALL_tmp,opt_a,opt_b);

    Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4));
    #Snx[:,4] = p_corr(Snx[:,4]);
    return Snx;

def Hot_to_Cold(hots,ac_list):
    a = hots.argmax(axis=1);
    a = np.asarray([ac_list[i] for i in a]);
    return a;

def PreProcess(ALL_x):
    pos = ALL_x[:,[0,1,2]]/5.0;
    vel = ALL_x[:,[3,4,5]]/10.0;
    ret_val = np.concatenate((pos,vel),axis=1)
    return ret_val
