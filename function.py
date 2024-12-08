import numpy as np

"""
The file containt all used functions in the project
"""

def split_sample(X, Y, N_LS):
    # size of learning subset
    avg_len = len(X) // N_LS
    remainder = len(X) % N_LS

    X_splits = []
    Y_splits = []
    start = 0
    for i in range(N_LS):
        #taking count of the reminder
        end = start + avg_len + (1 if i < remainder else 0)
        X_splits.append(X[start:end])
        Y_splits.append(Y[start:end])
        start = end

    return X_splits, Y_splits

def compute_bve(X,y,model,LS_size,LS_number=None):
    """
    This function compute the bias, variance and expected error for a given model
    
    Parameters :
        X, inputs
        Y, outputs
        model, the model on which we want to evaluate
        LS_size, size of the learning subsets 
        LS_number, number of learning subsets
        
    Return:
        B, bias computed as the mean of the square difference between the true outputs and the mean outputs of the models
        V, variance computed as the variance of the outputs of our model
        E, expected error computed as the MSE of the outputs of our model and the true outputs
        
    """
    #Splitting the learning sample into LS_number of subsets
    #range_loop is the the variable on which we iterate
    if LS_number==None:
        LS_number=len(X)//LS_size
        
    X_subsets, y_subsets=split_sample(X,y,LS_number)
    
    #we need to keep track of the predictions
    y_preds=np.empty(LS_number,dtype = 'object')
    #Mean squared error
    mse = np.zeros(len(y),)
    
    for i, subset in enumerate(X_subsets):
        model.fit(X_subsets[i],y_subsets[i])
        #predict on the whole learning sample
        y_preds[i]=model.predict(X)
        #mean squared error calculation
        mse+=(y-y_preds[i])**2
    #we do it once the calcultation are finished
    mse/=LS_number
    
    #we compute the mean of the squarred bias as b= y - E[predictions]
    bias=(y-np.mean(y_preds))**2 
    
    var=np.var(y_preds)
    
    return np.mean(bias), np.mean(var), np.mean(mse)
