import numpy as np
import scipy
import math
import statistics
from scipy.stats import skew as sp_skewness
import copy
import sklearn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix

def rolling_window(array, n):
    """Create a rolling window from an array.

    An extra axis is added to efficiently compute statistics over. Use
    ``axis=-1`` to remove the extra axis.

    Parameters
    ----------
    array : ndarray
        The input array.
    n : int
        Window length.

    Returns
    -------
    window : array
        The length-n windows of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from axopy.features.util import rolling_window
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> rolling_window(x, 2)
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5]])
    >>> rolling_window(x, 3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> rolling_window(x, 2)
    array([[[1, 2],
            [2, 3],
            [3, 4]],
    <BLANKLINE>
           [[5, 6],
            [6, 7],
            [7, 8]]])

    References
    ----------
    .. [1] https://mail.scipy.org/pipermail/numpy-discussion/2010-December/054392.html # noqa
    """
    shape = array.shape[:-1] + (array.shape[-1] - n + 1, n)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array,
                                           shape=shape,
                                           strides=strides)

def get_rms(emg,window_len,window_step):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    rms=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            rms[0,j+idx_window*Nchannel]=np.sqrt(np.mean(np.square(emg_window_channel)))
        idx_window=idx_window+1
    return rms

def get_wl(emg,window_len,window_step):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    wl=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel = emg_window[:, j]
            wl[0, j + idx_window * Nchannel] = np.sum(np.absolute(np.diff(emg_window_channel)))
        idx_window=idx_window+1
    return wl

def get_ssc(emg,window_len,window_step,thresh):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    ssc=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel = emg_window[:, j]
            diffs = np.diff(emg_window_channel)
            adj_diffs = rolling_window(np.absolute(diffs), 2)
            ssc[0, j + idx_window * Nchannel]=np.sum(np.logical_and(np.diff(np.signbit(diffs)),np.max(adj_diffs, axis=-1) > thresh))
        idx_window = idx_window + 1
    return ssc


def get_zc(emg,window_len,window_step,thresh):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    zc=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel = emg_window[:, j]
            zc[0, j + idx_window * Nchannel]=np.sum(np.logical_and(np.diff(np.signbit(emg_window_channel)),np.absolute(np.diff(emg_window_channel)) > thresh))
        idx_window = idx_window + 1
    return zc


def get_mav(emg,window_len,window_step):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    mav=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel = emg_window[:, j]
            mav[0, j + idx_window * Nchannel]=np.mean(np.absolute(emg_window_channel))
        idx_window = idx_window + 1
    return mav



def get_vcf(emg,window_len,window_step,fs):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    vcf=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            nfft_value = int(np.power(2, np.ceil(math.log(window_len, 2))))
            f,pxx=scipy.signal.welch(emg_window_channel, fs, window='hamming', nperseg=window_len, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=- 1, average='mean')
            sm0 = np.sum(pxx * pow(f, 0))
            sm1 = np.sum(pxx * pow(f, 1))
            sm2 = np.sum(pxx * pow(f, 2))
            vcf[0,j+idx_window*Nchannel]=sm2/sm0-pow((sm1/sm0),2)
        idx_window=idx_window+1
    return vcf

def get_mdf(emg,window_len,window_step,fs):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    mdf=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            nfft_value = int(np.power(2, np.ceil(math.log(window_len, 2))))
            f,pxx=scipy.signal.welch(emg_window_channel, fs, window='hamming', nperseg=window_len, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=- 1, average='mean')
            tmp = 1 / 2 * (sum(pxx))
            pxx_cumsum = np.cumsum(pxx)
            diff_pxx_cumsum = abs(pxx_cumsum - tmp)
            minIdx = np.argmin(diff_pxx_cumsum)
            mdf[0,j+idx_window*Nchannel]=f[minIdx]
        idx_window=idx_window+1
    return mdf

def get_mnf(emg,window_len,window_step,fs):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    mnf=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            nfft_value = int(np.power(2, np.ceil(math.log(window_len, 2))))
            f,pxx=scipy.signal.welch(emg_window_channel, fs, window='hamming', nperseg=window_len, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=- 1, average='mean')
            mnf[0, j + idx_window * Nchannel] = np.sum(f * pxx) / np.sum(pxx)
        idx_window=idx_window+1
    return mnf

def get_pkf(emg,window_len,window_step,fs):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    pkf=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            nfft_value = int(np.power(2, np.ceil(math.log(window_len, 2))))
            f,pxx=scipy.signal.welch(emg_window_channel, fs, window='hamming', nperseg=window_len, noverlap=None, nfft=nfft_value, detrend=False, return_onesided=True, scaling='density', axis=- 1, average='mean')
            maxIdx = np.argmax(pxx)
            pkf[0,j+idx_window*Nchannel]=f[maxIdx]
        idx_window=idx_window+1
    return pkf

def get_skw(emg,window_len,window_step):
    Nsample=np.size(emg,0)
    Nchannel=np.size(emg,1)
    Nwindow=int(np.round((Nsample-window_len)/window_step+1))
    skw=np.zeros([1,Nchannel*Nwindow],dtype=float)
    idx_window=0
    for i in range(window_len,Nsample+1,window_step):
        emg_window=emg[i-window_len:i,:]
        for j in range(Nchannel):
            emg_window_channel=emg_window[:,j]
            skewness_ = sp_skewness(emg_window_channel, axis=-1, bias=True, nan_policy='propagate')
            skw[0,j+idx_window*Nchannel]=skewness_
        idx_window=idx_window+1
    return skw



def subject_wise_norm(feature,label,label_subject,num_subsample):
    label_subject_unique=np.unique(label_subject)
    feature_norm=np.zeros([np.size(feature,0),np.size(feature,1)])
    label_unique=np.unique(label)
    for subject in label_subject_unique:
        idx = (label_subject == subject)
        idx_select=list()
        for class_ in label_unique:
            idx_class = (label_subject==subject) & (label==class_)
            idx_class = np.where(idx_class==True)[0][-num_subsample:]
            idx_select.append(idx_class)
        idx_select=np.hstack(idx_select)
        feature_stats_calculate=feature[idx_select,:]
        mean_val = np.mean(feature_stats_calculate, axis=0)
        std_val=np.std(feature_stats_calculate,axis=0)
        idx_zero=std_val==0
        std_val[idx_zero]=1
        feature_subject=feature[idx,:]
        feature_norm[idx,:]=(feature_subject-mean_val)/std_val
    return feature_norm

def findLeafIdHeight(childrenLeft,childrenRight,cutPredictor,cutPoint,input):
    nodeID=0
    height=0
    while (childrenLeft[nodeID]+childrenRight[nodeID]>0):
        predictorID = cutPredictor[nodeID]
        if(input[predictorID]<=cutPoint[nodeID]):
            nodeID=childrenLeft[nodeID]
        else:
            nodeID=childrenRight[nodeID]
        height = height + 1
    return nodeID,height


def get_sklearn_rf_parameters(mdl):
    trees = mdl.estimators_
    classes=mdl.classes_
    children_left_all=list()
    children_right_all = list()
    feature_index_all=list()
    threshold_all=list()
    value_all=list()
    for tree in trees:
        children_left_all.append(tree.tree_.children_left)
        children_right_all.append(tree.tree_.children_right)
        feature_index_all.append(tree.tree_.feature)
        threshold_all.append(tree.tree_.threshold)
        value_all.append(tree.tree_.value)
    parameters = {
        "left": children_left_all,
        "right": children_right_all,
        "feature": feature_index_all,
        "threshold": threshold_all,
        "value": value_all,
        "classes": classes
    }
    return parameters

def append(mdl1,mdl2):
    mdl = {
        "left": mdl1['left']+mdl2['left'],
        "right": mdl1['right']+mdl2['right'],
        "feature": mdl1['feature']+mdl2['feature'],
        "threshold": mdl1['threshold']+mdl2['threshold'],
        "value": mdl1['value']+mdl2['value'],
        "classes": mdl1['classes']
    }
    return mdl


def prune(mdl,feature_all,label_all):
    classes=mdl['classes']
    nTree = len(mdl['left'])
    Nexample = np.size(feature_all, 0)
    CutPredictorAll=mdl['feature']
    CutPointAll = mdl['threshold']
    ChildrenLeftAll=mdl['left']
    ChildrenRightAll=mdl['right']
    ValueAll=mdl['value']
    nodeNumMax = 0
    for k in range(nTree):
        nodeNum=len(mdl['left'][k])
        if(nodeNum>nodeNumMax):
            nodeNumMax=nodeNum

    nodePruneFlag = lil_matrix((nTree, nodeNumMax)) # sparse matrix nodePruneFlag will save if a node is pruned
    nodePruneLabel = lil_matrix((nTree, nodeNumMax)) # sparse matrix nodePruneLabel will save the new predicted label of a new leaf node after pruning

    for m in range(nTree):
        ChildrenLeft=ChildrenLeftAll[m]
        ChildrenRight=ChildrenRightAll[m]
        CutPredictor=CutPredictorAll[m]
        CutPoint=CutPointAll[m]
        value=ValueAll[m]
        leafID=np.zeros([Nexample,])  # variable to save the ID of leaf that each calibration sample falls in
        height=np.zeros([Nexample,]) # variable to save the height (depth) of each leaf in variable leafID
        feature=copy.deepcopy(feature_all)
        label=copy.deepcopy(label_all)
        for k in range(Nexample):
            leafID[k],height[k]=findLeafIdHeight(ChildrenLeft,ChildrenRight,CutPredictor,CutPoint,feature[k,:])
        height_max=np.max(height)
        leafIDUnique = np.unique(leafID)
        leafSampleLabel = -1*np.ones([nodeNumMax,Nexample+1]) # leafSampleLabel saves the information of samples that fall into each leaf node. Each row represents a node. If a node is not leaf node, the corresponding row will be filled with -1 element. If a node is leaf node, and have n calibration samples falling into the leaf node, the first n elements in this row save the groundtruth labels of these calibration samples, with the (n+1)th element saving the predicted label of the leaf node
        for k in range(len(leafIDUnique)):
            idx_samples_in_leaf=np.where(leafID==leafIDUnique[k])[0] # find the index of samples that fall into the leaf with leafID=leafIDUnique[k]
            leafSampleLabel[int(leafIDUnique[k]), 0:len(idx_samples_in_leaf)]=label[idx_samples_in_leaf] # each row corresponds to a leaf node; elements in each row save labels of all calibration samples that fall into a specific leaf node
            node_statistic = value[int(leafIDUnique[k]), 0, :]
            idx=np.where(node_statistic==np.max(node_statistic))[0][0]
            label_predict=classes[idx]
            leafSampleLabel[int(leafIDUnique[k]), len(idx_samples_in_leaf)] = label_predict  # the last valid element in each row saves the predicted label for a sample that fall into a leaf node
        while (height_max > 0): # iterated inspection starts from the highest node
            idx = np.where(height == height_max)
            idx=idx[0]
            leafIDUnique = np.unique(leafID[idx]) # unique leaf ID with the same height
            height_max = height_max - 1
            if (len(leafIDUnique) == 0): # if there's no node leaf node at the current height, continue to inspect the next height
                continue
            for k in range(len(leafIDUnique)): # if leaf nodes exist at the current height, inspect these leaf nodes one by one
                leaf_tmp = leafIDUnique[k]

                # find the parent node of the inspected leaf node
                tmp1 = np.where(ChildrenLeft == leaf_tmp)
                tmp1=tmp1[0]
                tmp2 = np.where(ChildrenRight == leaf_tmp)
                tmp2=tmp2[0]
                if(len(tmp1)==1):
                    parent_tmp=tmp1[0]
                elif(len(tmp2)==1):
                    parent_tmp = tmp2[0]
                else:
                    continue

                # find other children nodes of the parent node (i.e. the brother node of the inspected leaf node)
                childIdAll=np.array([ChildrenLeft[parent_tmp],ChildrenRight[parent_tmp]])

                # find all children (and children of children) nodes (if the inspected leaf node is pruned, all these bother and children nodes will be pruned together)
                while(1):
                    childId=np.array([ChildrenLeft[childIdAll],ChildrenRight[childIdAll]])
                    childId=np.reshape(childId,[1,-1],order='F')
                    childId=childId.ravel()
                    childId=childId[childId>=0] # new children nodes
                    if(len(childId)>0): # if there are new children nodes
                        childIdAllUpdate = np.unique(np.concatenate((childIdAll, childId),axis=0))
                        if (len(childIdAllUpdate) == len(childIdAll)): # if there's no update (no new children node), jump out the loop
                            break
                        else:
                            childIdAll = childIdAllUpdate
                    else:
                        break

                label_sample_node=np.zeros([0,])
                label_truth_node=np.zeros([0,])
                for childID in childIdAll: # find labels of all calibration samples that fall into the found children nodes
                    tmp1=leafSampleLabel[childID,:]
                    tmp1=tmp1[tmp1>=0]
                    if(len(tmp1)>0):
                        tmp2=tmp1[0:-1] # the last element saves the predicted label of the pre-trained leaf node, while previous elements save the ground truth labels of calibration samples fall into this node
                        label_sample_node=np.concatenate((label_sample_node,tmp2),axis=0)
                        label_truth_node=np.concatenate((label_truth_node,tmp1[-1]*np.ones([len(tmp1)-1,])),axis=0)
                acc_node=np.mean(label_sample_node==label_truth_node) # validation accuracy of calibration samples in all found children nodes
                label_prune=statistics.mode(label_sample_node) # the new predicted label if the pruning operation is performed (i.e. the most frequently appearing label)
                acc_prune = np.mean(label_sample_node == label_prune) # the new validation accuracy of calibration samples in all found children nodes, if the pruning operation is performed
                if(acc_prune>acc_node): # the pruning operation is performed only if pruning improves the accuracy.
                    nodePruneFlag[m, parent_tmp] = 2  # 2 denotes this node is pruned, and set as a new leaf node
                    nodePruneLabel[m, parent_tmp] = label_prune+1 # avoid zero label in a sparse matrix. if the label is n, the value is encoded as n+1, which will be decoded back to n later.
                    nodePruneFlag[m, childIdAll] = 1  # 1 denotes this node is pruned, and no longer a valid node in the tree
                    nodePruneLabel[m, childIdAll] = label_prune+1 # avoid zero label in a sparse matrix. if the label is n, the value is encoded as n+1, which will be decoded back to n later.
                    leafSampleLabel[parent_tmp,0:len(label_sample_node)]=label_sample_node # update the information of the new leaf node (the pruned node), samples that previously fell into its children nodes now all fall into the new leaf node
                    leafSampleLabel[parent_tmp,len(label_sample_node)]=label_prune # the new predicted label of the new leaf node after pruning
                    leafSampleLabel[childIdAll,:]=0*leafSampleLabel[childIdAll,:]-1 # samples that previously fell into the children nodes of the pruned node are cleared
                    height_update=np.zeros([len(height)+1,])
                    height_update[0:-1]=height
                    height_update[-1]=height_max
                    height=height_update
                    leafID_update=np.zeros([len(leafID)+1,])
                    leafID_update[0:-1]=leafID
                    leafID_update[-1]=parent_tmp
                    leafID=leafID_update
                    ChildrenLeft[parent_tmp]=-1 # all nodes related to the pruning operation now have no children nodes
                    ChildrenLeft[childIdAll]=-1
                    ChildrenRight[parent_tmp]=-1
                    ChildrenRight[childIdAll]=-1
    mdl_update=model_update_prune(mdl, nodePruneFlag, nodePruneLabel)
    return mdl_update


def model_update_prune(mdl,nodePruneFlag,nodePruneLabel):
    classes = mdl['classes']
    nTree=len(mdl['left'])
    for i in range(nTree):

        nodePruneFlagTree=nodePruneFlag[i,:]
        idx_prune_parents=(nodePruneFlagTree==2).toarray().ravel()
        idx_prune_children = (nodePruneFlagTree == 1).toarray().ravel()
        idx_prune=(nodePruneFlagTree>0).toarray().ravel()

        children_left=mdl['left'][i]
        children_right=mdl['right'][i]
        feature=mdl['feature'][i]
        threshold=mdl['threshold'][i]
        value=mdl['value'][i]

        n_node=len(children_left)
        n_classes=np.sum(np.sum(value[children_left==-1,:,:],axis=0)>0)

        children_left[idx_prune[0:n_node]]=-1
        children_right[idx_prune[0:n_node]] = -1
        feature[idx_prune[0:n_node]] = -2
        threshold[idx_prune[0:n_node]] = -2
        value[idx_prune_children[0:n_node],:,:] = np.zeros((1,n_classes))
        idx=[k for k in range(n_node) if idx_prune_parents[k]]
        for k in idx:
            label_prune_predict=nodePruneLabel[i,k] - 1 # to avoid zero label in a sparse matrix, the label n is encoded as n+1. Here the label was decoded back to n
            value_prune_predict=np.zeros((1,n_classes))
            idx_predict=np.where(classes==label_prune_predict)[0][0]
            value_prune_predict[0,idx_predict]=1
            value[k,:,:]=value_prune_predict
    return mdl



def get_predictions(mdl,features):
    classes=mdl['classes']
    Ntree=len(mdl['left'])
    features=np.reshape(features,(np.size(features,0),-1),'F')
    if(np.size(features,1)==1):
        features = np.reshape(features, (1,np.size(features, 0)), 'F')
    n_sample=np.size(features,axis=0)
    score_predict=np.zeros((n_sample,int(np.max(mdl['classes']+1))))
    for idx_tree in range(Ntree):
        parameter_left = mdl['left'][idx_tree]
        parameter_right = mdl['right'][idx_tree]
        parameter_feature = mdl['feature'][idx_tree]
        parameter_threshold = mdl['threshold'][idx_tree]
        parameter_value = mdl['value'][idx_tree]
        for idx_sample in range(n_sample):
            idx_node=0
            feature=features[idx_sample,:]
            while(parameter_left[idx_node]>0):
                left_flag = feature[parameter_feature[idx_node]] < parameter_threshold[idx_node]
                if(left_flag):
                    idx_node = parameter_left[idx_node]
                else:
                    idx_node = parameter_right[idx_node]
            label_tree=int(classes[np.argmax(parameter_value[idx_node,0,:])])
            score_predict[idx_sample,label_tree]=score_predict[idx_sample,label_tree]+1
    label_predict=np.argmax(score_predict,axis=1)
    if(len(label_predict)==1):
        label_predict=label_predict[0]
    idx=np.isin(np.array(range(np.size(score_predict,1))),mdl['classes'])
    score=score_predict[:,idx]
    score=np.divide(score,np.reshape(np.sum(score,1),(np.size(score,0),1)))
    return label_predict, score



def findGroupCenter(feature,label):
    label_unique=np.unique(label)
    feature_group_center=np.zeros((len(label_unique),np.size(feature,1)))
    idx_group=0
    for label_tmp in label_unique:
        idx=(label==label_tmp)
        feature_group=feature[idx,:]
        feature_group_center[idx_group,:]=np.mean(feature_group,0)
        idx_group=idx_group+1
    return feature_group_center


def buffer_update(max_sample,data_buffer,label_buffer,data_in,label_in):
    label_in = np.reshape(label_in, (np.size(label_in)))
    data_buffer_new=copy.deepcopy(data_buffer)
    label_buffer_new=copy.deepcopy(label_buffer)
    if((len(label_buffer)+len(label_in))<=max_sample):
        data_buffer_new=np.concatenate((data_in, data_buffer_new),axis=0)
        label_buffer_new = np.concatenate((label_in, label_buffer_new), axis=0)
    else:
        size_remain=max_sample-len(label_buffer)
        if(size_remain>0):
            data_buffer_new = np.concatenate((data_in[:size_remain,:], data_buffer_new), axis=0)
            label_buffer_new = np.concatenate((label_in[:size_remain], label_buffer_new), axis=0)
            data_in=np.delete(data_in,np.array(range(size_remain)),axis=0)
            label_in = np.delete(label_in, np.array(range(size_remain)), axis=0)
        for idx in range(len(label_in)):
            label_buffer_new=label_buffer_new.astype(np.int64)
            label_most=np.bincount(label_buffer_new).argmax()
            idx_delete = np.where(label_buffer_new == label_most)[0][-1] # delete the samples in buffer with the most frequencyly appearing class (pseudo-)label
            data_buffer_new = np.delete(data_buffer_new, idx_delete, axis=0)
            label_buffer_new = np.delete(label_buffer_new, idx_delete, axis=0)
            data_buffer_new=np.insert(data_buffer_new,0,data_in[idx,:],axis=0)
            label_buffer_new = np.insert(label_buffer_new, 0, label_in[idx], axis=0)
    return data_buffer_new,label_buffer_new

def manifold_clustering(feature_buffer, label_set_all_classes, min_sample_class, mdl, method='kmeans', embedding=None):
    label_buffer_update, score = get_predictions(mdl, feature_buffer)
    count = list()
    for label_class in label_set_all_classes:
        count.append(np.sum(label_buffer_update == label_class))
    if(np.min(count)<min_sample_class): # clustering-based pseudo-labeling is performed only when the number of samples in each class reaches the threshold (clustering performance relies on a reasonable number of samples). Otherwise, simply use the model predictions as the pseudo-labels
        count=[i if i>0 else np.inf for i in count] # if the number of samples in a class is zero, using a special symbol to denote it (inf in our implementation), so that the number of selected samples for each class can be determined by the second minimal number of samples in a class (>0 samples).
        feature_buffer_select = np.zeros((0, np.size(feature_buffer, axis=1))) # the final dataset used in self-calibration contains the same number of samples for each class (a balanced dataset, determined by the pseudo-labels). The number of samples for each class is set to the minimal number of samples in a class.
        label_select_pseudo = np.zeros((0,))
        for label_class in label_set_all_classes:
            idx = (label_buffer_update == label_class)
            if(count[label_set_all_classes.index(label_class)]==np.inf):
                continue
            # make the dataset balanced across classes (according to the pseudo labels)
            feature_class = feature_buffer[idx, :]
            feature_class = feature_class[:int(np.min(count)), :]  # by setting 0 in count as inf, min(count) will find the second-smallest number.
            label_class_ = label_buffer_update[idx]
            label_class_ = label_class_[:int(np.min(count))]
            feature_buffer_select = np.concatenate((feature_buffer_select, feature_class), axis=0)
            label_select_pseudo = np.concatenate((label_select_pseudo, label_class_), axis=0)
    else:
        feature_buffer_select=np.zeros((0,np.size(feature_buffer,axis=1)))
        label_select = np.zeros((0,))
        for label_class in label_set_all_classes:
            idx = (label_buffer_update == label_class)
            feature_class=feature_buffer[idx,:]
            feature_class=feature_class[:np.min(count),:]
            label_class_ = label_buffer_update[idx]
            label_class_=label_class_[:np.min(count)]
            feature_buffer_select=np.concatenate((feature_buffer_select,feature_class),axis=0)
            label_select = np.concatenate((label_select, label_class_), axis=0)
        if(embedding==None):
            feature_clustering_input=copy.deepcopy(feature_buffer_select)
        if(embedding=='tsne'): # add another branch if you extend the functions using other manifold learning algorithms
            feature_clustering_input = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=10).fit_transform(feature_buffer_select)
            scaler = sklearn.preprocessing.StandardScaler().fit(feature_clustering_input)
            feature_clustering_input = scaler.transform(feature_clustering_input)

        if(method=='kmeans'): # add another branch if you extend the functions using other clustering algorithms
            feature_group_center = findGroupCenter(feature_clustering_input, label_select)
            mdl = KMeans(n_clusters=len(label_set_all_classes), init=feature_group_center, random_state=0, n_init=1).fit(feature_clustering_input)
            if(len(np.unique(mdl.labels_))==len(label_set_all_classes)):
                label_select_pseudo = mdl.labels_
                label_max=np.max(label_set_all_classes)
                label_unique=np.unique(label_select_pseudo)
                for idx in range(len(label_unique)):
                    idx_class=label_select_pseudo==label_unique[idx]
                    label_select_pseudo[idx_class]=label_set_all_classes[idx]+label_max+1
                label_select_pseudo=label_select_pseudo-label_max-1
            else:
                label_select_pseudo=copy.deepcopy(label_select)

    return feature_buffer_select, label_select_pseudo, label_buffer_update

def replace(mdl1,mdl2,idx_remain):
    nTree=len(mdl1['left'])
    mdl=copy.deepcopy(mdl1)
    idx_replace=-1
    for i in range(nTree):
        if(np.isin(i, idx_remain)==False):
            idx_replace=idx_replace+1
            mdl['left'][i] = mdl2['left'][idx_replace]
            mdl['right'][i] = mdl2['right'][idx_replace]
            mdl['feature'][i] = mdl2['feature'][idx_replace]
            mdl['threshold'][i] = mdl2['threshold'][idx_replace]
            mdl['value'][i] = mdl2['value'][idx_replace]
    return mdl




