# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:54:03 2016

@author: nhern121@cs.fiu.edu

"""

import pandas as pd
import numpy as np
import networkx as nx
import decimal
import os
import sys
from decimal import *
import math
import aux_lbp as auxi

###############################################################################
#Fraudhunt
#Adapted Loopy Belief Propagation Algorithm.

#Inputs:
#g: a graph of type networkx.classes.graph.Graph. One can get to this type from 
#a csv file containing all the edges of a graph with the weight by using 
#nx.from_pandas_dataframe(pd.read_csv("csvfile.csv"), 'node1', 'node2', ['weight']).
# See the Networkx module in python for more details.

#Example of format of the csv file is:

# node1 node2 weight
#  a     b      5
#  a     c      10
#  .     .       .
#  .     .       .

#Then one can do :

#datafile=pd.read_csv("csvfile.csv")
#g=nx.from_pandas_dataframe(datafile, 'id1', 'id2', ['weight'])

#delta: value related to the compatibility potentials. It is in [0,1]. 
#Delta can be set based on domain knowledge or using ground truth data.
#For instance, the optimal value for our problem turned out to be 0.7 after
#an extensive numerical experimentation.


#It makes use of the functions: compatibility, prods, prodd, prodnode. Details
#about those function is found below. 

#Outputs:
# A data frame whose dimension is the number of nodes in the network. The columns
# are: id of the node, belief of being sybil or fraudster, degree, and sum of the 
#weights to its neighbors.

#Number of iterations taken to converge.


def lbp(g,delta):

#initial guess for messages (set them all to 1)    
    for (u,v) in g.edges():
        g[u][v]['mssd_h']=Decimal(1)
        g[u][v]['mssd_s']=Decimal(1)
        g[u][v]['msds_h']=Decimal(1)
        g[u][v]['msds_s']=Decimal(1)
        g[u][v]['mssdO_h']=Decimal(1)
        g[u][v]['mssdO_s']=Decimal(1)
        g[u][v]['msdsO_h']=Decimal(1)
        g[u][v]['msdsO_s']=Decimal(1)
    
#compute degree of each node
    for u in g.nodes():
        g.node[u]['neighbours']=list(set(nx.edges_iter(g,nbunch=u)))
        g.node[u]['degree']=Decimal(len(g.node[u]['neighbours']))
        
#compute sum of the weight.
    for u in g.nodes():
        n=g.node[u]['neighbours']
        suma=0
        for (h,w) in n:
            suma=suma+g[h][w]['weight']
        g.node[u]['sumweights']=Decimal(suma)
    
# the prior for all the users (based on weight and degree)
    for u in g.nodes():
        g.node[u]['prior_h']=Decimal(1)/(g.node[u]['sumweights']/g.node[u]['degree'])
        g.node[u]['prior_s']=Decimal(1)-g.node[u]['prior_h']

#compute compatibility potentials        
    for (u,v) in g.edges():
        comp=auxi.compatibility(g[u][v]['weight'],delta)
        g[u][v]['c_hh']=comp[0]
        g[u][v]['c_sh']=comp[1]
        g[u][v]['c_hs']=comp[2]
        g[u][v]['c_ss']=comp[3]

#set who is the source and who is the sink.
    for (u,v) in g.edges():
        g[u][v]['source']=u
        g[u][v]['dest']=v

# save the neighbours.
    for (u,v) in g.edges():
        g[v][u]['ns']=list(set(nx.edges_iter(g,nbunch=g[u][v]['source']))-set([(u,v)]))
        g[u][v]['nd']=list(set(nx.edges_iter(g,nbunch=g[u][v]['dest']))-set([(v,u)]))
        
#main loop: we iterate until the stopping criterion is reached on the  L-2 norm of the messages.
    
    delta=0.0001
    tol=1
    numedges=len(g.edges())
    vector0=[Decimal(10)]*numedges*4
    j=1

    while tol>delta:    
        
        vector=[]
        
        #message update from source to dest
        for (u,v) in g.edges():
            a=auxi.prods((u,v),'h',g)
            b=auxi.prods((u,v),'s',g)
            g[u][v]['mssd_h']=g[u][v]['c_hh']*g.node[g[u][v]['source']]['prior_h']*a+g[u][v]['c_sh']*g.node[g[u][v]['source']]['prior_s']*b
            g[u][v]['mssd_s']=g[u][v]['c_hs']*g.node[g[u][v]['source']]['prior_h']*a+g[u][v]['c_ss']*g.node[g[u][v]['source']]['prior_s']*b
            alpha=g[u][v]['mssd_h']+g[u][v]['mssd_s']
            g[u][v]['mssd_h']=g[u][v]['mssd_h']/alpha
            g[u][v]['mssd_s']=g[u][v]['mssd_s']/alpha
            vector.append(g[u][v]['mssd_h'])
            vector.append(g[u][v]['mssd_s'])
            
        #message update from dest to source
        for (u,v) in g.edges():
            a=auxi.prodd((u,v),'h',g)
            b=auxi.prodd((u,v),'s',g)
            g[u][v]['msds_h']=g[u][v]['c_hh']*g.node[g[u][v]['dest']]['prior_h']*a+g[u][v]['c_hs']*g.node[g[u][v]['dest']]['prior_s']*b
            g[u][v]['msds_s']=g[u][v]['c_sh']*g.node[g[u][v]['dest']]['prior_h']*a+g[u][v]['c_ss']*g.node[g[u][v]['dest']]['prior_s']*b
            alpha=g[u][v]['msds_h']+g[u][v]['msds_s']
            g[u][v]['msds_h']=g[u][v]['msds_h']/alpha
            g[u][v]['msds_s']=g[u][v]['msds_s']/alpha
            vector.append(g[u][v]['msds_h'])
            vector.append(g[u][v]['msds_s'])
        
        #update old messages            
        for (u,v) in g.edges():
            g[u][v]['mssdO_h']=g[u][v]['mssd_h']
            g[u][v]['mssdO_s']=g[u][v]['mssd_s']
            g[u][v]['msdsO_h']=g[u][v]['msds_h']
            g[u][v]['msdsO_s']=g[u][v]['msds_s']
        

        tol=np.linalg.norm(np.array(vector)-np.array(vector0),ord=2)
        vector0=vector    
        
        print j
        print tol
        j=j+1    


#compute final beliefs:
    
    for u in g.nodes():
        g.node[u]['belief_h']=g.node[u]['prior_h']*auxi.prodnode(u,'h',g)
        g.node[u]['belief_s']=g.node[u]['prior_s']*auxi.prodnode(u,'s',g)
        alpha=g.node[u]['belief_h']+g.node[u]['belief_s']
        g.node[u]['belief_h']=g.node[u]['belief_h']/alpha
        g.node[u]['belief_s']=g.node[u]['belief_s']/alpha
 
#convert back to float:

    for u in g.nodes():
        g.node[u]['belief_h']=float(g.node[u]['belief_h'])
        g.node[u]['belief_s']=float(g.node[u]['belief_s'])
        g.node[u]['prior_h']=float(g.node[u]['prior_h'])
        g.node[u]['prior_s']=float(g.node[u]['prior_s'])
        
    for (u,v) in g.edges():
         g[u][v]['mssd_h']=float(g[u][v]['mssd_h'])
         g[u][v]['mssd_s']=float(g[u][v]['mssd_s'])
         g[u][v]['msds_h']=float(g[u][v]['msds_h'])
         g[u][v]['msds_s']=float(g[u][v]['msds_s'])
         g[u][v]['c_hh']=float(g[u][v]['c_hh'])
         g[u][v]['c_hs']=float(g[u][v]['c_hs'])
         g[u][v]['c_sh']=float(g[u][v]['c_sh'])
         g[u][v]['c_ss']=float(g[u][v]['c_ss'])


    sybil_belief={}
    
    for u in g.nodes():
        sybil_belief[u]={'belief_s':g.node[u]['belief_s'], 'degree':len(g.node[u]['neighbours']),'sumweight':g.node[u]['sumweights']}

    users=pd.DataFrame.from_dict(sybil_belief,orient='index')
    users=users.reset_index()
    users.columns=['userid','belief_s','sumweight','degree']
    users = users.sort(['userid'], ascending=False)
    users[['userid']] = users[['userid']].astype(str)

    return users, j
    
