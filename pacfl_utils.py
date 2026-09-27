from typing import Iterable 
import copy
import numpy as np

def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x
            
def calculating_adjacency(clients_idxs, U): 
    nclients = len(clients_idxs)
    sim_mat = np.zeros([nclients, nclients])
    for idx1 in range(nclients):
        for idx2 in range(nclients):
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])
            mul = np.clip(U1.T @ U2, a_min=-1.0, a_max=1.0)
            sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi
    return sim_mat

def hierarchical_clustering(A, thresh=1.5, linkage='average'):
    label_assg = {i: i for i in range(A.shape[0])}
    
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A, np.inf)
        step += 1
        ind = np.unravel_index(np.argmin(A, axis=None), A.shape)

        if A[ind[0], ind[1]] > thresh:
            break
        else:
            np.fill_diagonal(A, 0)
            if linkage == 'maximum':
                Z = np.maximum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'minimum':
                Z = np.minimum(A[:,ind[0]], A[:,ind[1]])
            elif linkage == 'average':
                Z = (A[:,ind[0]] + A[:,ind[1]])/2
            
            A[:,ind[0]]=Z
            A[:,ind[1]]=Z
            A[ind[0],:]=Z
            A[ind[1],:]=Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            if type(label_assg[ind[0]]) == list: 
                label_assg[ind[0]].append(label_assg[ind[1]])
            else: 
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k, v in label_assg.items():
                kk = k - 1 if k > ind[1] else k 
                temp.append((kk, v))
            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) == int: 
            clusters.append([label_assg[k]])
            
    return clusters