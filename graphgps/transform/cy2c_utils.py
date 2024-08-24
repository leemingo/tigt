
import torch
import numpy as np
import torch.nn as nn
import torch_geometric.transforms as T
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops,remove_self_loops

def make_cycle_adj_speed_nosl(raw_list_adj,data):
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : 
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
            #start = time.time()
            #N_V=len(nodes)                
            for i in nodes:
                SUM_CYCLE_ADJ[i,nodes]=1   
            SUM_CYCLE_ADJ[nodes,nodes]=0
            #print('3. time',time.time()-start)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ


# def max_node_dataset(dataset):
#     aa=[]
#     for i in range(len(dataset)):
#         data=dataset[i]
#         aa.append(data.num_nodes)
#     max_node=np.max(aa) 
#     return max_node

# max_node = max([x.num_nodes for x in dataset])
# new_dataset = make_NEWDATA(dataset, max_node)

def make_NEWDATA(data,max_node,cy2c_self=True, cy2c_same_attr=True, cy2c_trans=False):
    # SUB_ADJ=[]
    # RAW_SUB_ADJ=[]
    # NEWDATA=[]    
    # for i, data in enumerate(dataset):
        # data=dataset[i]
    v1=data.edge_index[0,:]
    v2=data.edge_index[1,:]
    #print(torch.max(v1))
    adj = torch.zeros((max_node,max_node))
    adj[v1,v2]=1
    adj=adj.numpy()
    (adj==adj.T).all()
    list_feature=(data.x)
    list_adj=(adj)       

    #print(dataset[i])
    _, _, _, _, sum_sub_adj = make_cycle_adj_speed_nosl(list_adj,data)

    # if i % 100 == 0:
    #     print(i)

    #_sub_adj=np.array(sub_adj)

    if len(sum_sub_adj)>0:    
        new_adj=np.stack((list_adj,sum_sub_adj),0)
    else :
        sum_sub_adj=np.zeros((1, list_adj.shape[0], list_adj.shape[1]))
        new_adj=np.concatenate((list_adj.reshape(1, list_adj.shape[0], list_adj.shape[1]),sum_sub_adj),0)

    #SUB_ADJ.append(new_adj)
    SUB_ADJ=new_adj
    # data=dataset[i]
    edge_index=data.edge_index
    check1=torch.sum(edge_index[0]-np.where(SUB_ADJ[0]==1)[0])+torch.sum(edge_index[1]-np.where(SUB_ADJ[0]==1)[1])
    if check1 != 0 :
        print('error')

    cycle_index=torch.stack((torch.LongTensor(np.where(SUB_ADJ[1]!=0)[0]), torch.LongTensor(np.where(SUB_ADJ[1]!=0)[1])),1).T.contiguous()
    
    if cy2c_self == True:
    #0903 test
        cycle_index, _ = remove_self_loops(cycle_index)
        cycle_index, _ = add_self_loops(cycle_index) 
        cycle_attr = torch.ones(cycle_index.shape[1]).long()
    else:
        cycle_attr = torch.ones(cycle_index.shape[1]).long()
    
    if cy2c_same_attr == True:
        pos_edge_attr = torch.ones(edge_index.shape[1]).long()
    elif cy2c_same_attr == False:
        pos_edge_attr = torch.zeros(edge_index.shape[1]).long()

    if cy2c_trans == True : 
        cycle_index, _ = remove_self_loops(cycle_index)
        old_length = cycle_index.shape[1]
        cycle_index, _ = add_self_loops(cycle_index)
        new_length = cycle_index.shape[1]
        added_self_loops = new_length - old_length
        cycle_attr = torch.ones(new_length).long()
        if added_self_loops > 0:
            cycle_attr[-added_self_loops:] = 0  
            
    return cycle_index, cycle_attr, pos_edge_attr
    # return dataset
