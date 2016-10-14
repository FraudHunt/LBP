# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:53:13 2016

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


#Complement functions to LBP.
###############################################################################
#This function computes the compatibility potentials as defined in the Paper.
# Note that this is based on the weight of the link between its end points.
# Delta can be set based on domain knowledge or using ground truth data.

def compatibility(w,delta):
    delta=Decimal(delta)
    return (delta,delta**Decimal(math.log(w)),Decimal(1)-delta,Decimal(1)-(delta**Decimal(math.log(w))))


###############################################################################
#Product of messages to the source.
#inputs:
#(u,v): an edge of a graph.
#c: the message sent (h for honest, s for sybil). In this case, c take values in
#{h,s} because we have a two-class network classification problem.
#g: the graph

def prods((u,v),c,g):
    s=g[u][v]['source']
    n=g[u][v]['ns']
    if len(n)==0:
        return 1
    elif c=='h':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==s:
                prod=Decimal(prod)*Decimal(g[h][w]['msdsO_h'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssdO_h'])
    elif c=='s':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==s:
                prod=Decimal(prod)*Decimal(g[h][w]['msdsO_s'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssdO_s'])
    return prod

##############################################################################
#product of messages to dest

#inputs:
#(u,v): an edge of a graph.
#c: the message sent (h for honest, s for sybil)
#g: the graph

def prodd((u,v),c,g):
    d=g[u][v]['dest']
    n=g[u][v]['nd']
    if len(n)==0:
        return 1
    elif c=='h':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==d:
                prod=Decimal(prod)*Decimal(g[h][w]['msdsO_h'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssdO_h'])
    elif c=='s':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==d:
                prod=Decimal(prod)*Decimal(g[h][w]['msdsO_s'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssdO_s'])
    return prod    

###############################################################################    
#product of all messages for a certain node.

#inputs:
#u: a node in the graph
#c: the message sent (h for honest, s for sybil)
#g: the graph

def prodnode(u,c,g):
    n=g.node[u]['neighbours']
    if c=='h':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==u:
                prod=Decimal(prod)*Decimal(g[h][w]['msds_h'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssd_h'])
    elif c=='s':
        prod=1
        for (h,w) in n:
            if g[h][w]['source']==u:
                prod=Decimal(prod)*Decimal(g[h][w]['msds_s'])
            else:
                prod=Decimal(prod)*Decimal(g[h][w]['mssd_s'])
    return prod
    