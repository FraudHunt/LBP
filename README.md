# LBP
Loopy Belief Propagation (LBP) approach to formulate the problem of detecting fraudulent user accounts as a network classification task.

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
