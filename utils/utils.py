import numpy as np
import logging
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.special import xlogy 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
import pandas as pd

########################################################################################

class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def performance(y_actual, y_pred):
    """Compute total of true/false positive/negatives"""
    y_actual.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)
    TN, FP, FN, TP = confusion_matrix(y_actual, y_pred).ravel()
    
    if (TP + FN) != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0
    
    # Specificity/true negative rate
    if (TN + FP) != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0
    
    # Accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # Balanced accuracy
    B_ACC = (TPR + TNR) / 2
    
    return TPR, TNR, ACC, B_ACC

########################################################################################

def conditions_TF(Mstep): 
    """Compute whether case is true/false positive/negative"""
    if (Mstep['manipulated']==Mstep['z']==1): 
        return "TP"
    elif (Mstep['manipulated']==Mstep['z']==0): 
        return "TN"
    elif (Mstep['z']==1 and Mstep['manipulated']!=Mstep['z']):
        return "FP"    
    else: #the false negatives are the other ones.
        return "FN"  
    
########################################################################################

def p_logp(p):
    if p == 0:
        return 0
    else:
        return p * np.log(p)

########################################################################################

def ent_y(pred_y):
    """ calculates the entropy value for the predicted label probabilities of one individual record

    Args:
        pred_y (list-like): predicted label probabilities

    Returns:
        float: entropy of one record
    """
    entropy = -sum([p_logp(yi) for yi in pred_y])
    return entropy

########################################################################################

def rel_ent_y_noisy(pred, n_label):
    """ calculates the entropy value for the complete set of predictions

    Args:
        pred (array-like): predicted label probabilities
        n_label (int): total number of unique labels

    Returns:
        float: entropy value of the complete set of predictions
    """
    entropy = 1 - sum([ent_y(pred_y) for pred_y in pred]) / (len(pred) * np.log(n_label))
    return entropy

########################################################################################

def eR2_y_gold(pred, n_label):
    """Compute relative cross entropy on the predicted label probabilities"""
    sumlogp = - sum([np.log(p + 1e-20) for p in pred])
    ce = 1 - sumlogp / (len(pred) * np.log(n_label))
    return ce

########################################################################################

def eR2(true, pred):
    """Compute relative cross entropy on the z values"""
    true.reset_index(drop=True, inplace=True)
    pred.reset_index(drop=True, inplace=True)
    log__loss = [-np.log(pred[i] + 1e-20) if true[i]==1 else -np.log(1 - pred[i] + 1e-20) for i in range(0, len(true))] # add 1e-20 to avoid log(0)
    sumll = sum(log__loss)
    ce = 1 - (sumll / (len(true)*np.log(2)))
    return ce

########################################################################################

def get_sd(mean_pc, n):
    """Helper function"""
    return np.sqrt((1/n) * mean_pc * (1 - mean_pc))

########################################################################################

def llh_function(dat):
    """Compute likelihood"""
    dat.probML = dat.probML.astype(float)
    
    ### noisy part
    inclusion = dat['cat'] == "noisy"
    dat_noisy = dat[inclusion] 

    Pyisy = dat_noisy['probML']
    z = dat_noisy['pred_z']
    pi = dat_noisy['pred_pi']
    
    contr_psi = dat_noisy['contr_psi']
    l_noisy = np.log(Pyisy+(contr_psi-Pyisy)*pi+1e-20)
    sum_l_noisy = sum(l_noisy)

    ### labelled part
     
    inclusion = dat['cat'] != "noisy"
    dat_labeled = dat[inclusion] 

    Pyisy = dat_labeled['probML']
    z = dat_labeled['pred_z']
    pi = dat_labeled['pred_pi']
    
    contr_psi = dat_labeled['contr_psi']
    # l_lab_pi = (1-z)*np.log(1-pi) + z * np.log(pi)
    l_lab_pi = xlogy(1 - z, 1 - pi + 1e-20) + xlogy(z, pi + 1e-20)
    # l_lab_ml = (1-z)*np.log(Pyisy) + z * np.log(contr_psi)
    l_lab_ml = xlogy(1 - z, Pyisy + 1e-20) + xlogy(z, contr_psi + 1e-20)
    l_lab = l_lab_pi + l_lab_ml 

    sum_l_lab_pi = sum(l_lab_pi)
    sum_l_lab_ml = sum(l_lab_ml)    
    sum_l_lab = sum(l_lab)

    l_tot = sum_l_noisy + sum_l_lab
    return sum_l_noisy, sum_l_lab_pi, sum_l_lab_ml, sum_l_lab, l_tot

#######################################################################################################################################

def Q_pi_function(dat): 
    """Compute likelihood"""
   
    # Voor noisy en labelled
    tau=dat['tau'] # Voor het lab gedeelte, pred_z nodig, voor noisy pred_tau. Hiervoor tau_new_adj gebruiken waar dit al is bewerkt.
    pi=dat['pred_pi'] #deze is voor label en noisy hetzelfde

    # l_lab_pi = (1-tau)*np.log(1-pi) + tau * np.log(pi)
    l_lab_pi = xlogy(1 - tau, 1 - pi + 1e-20) + xlogy(tau, pi + 1e-20)
    sum_l_lab_pi = sum(l_lab_pi)

    return sum_l_lab_pi

#######################################################################################################################################

def Q_ml_function(dat):
    """Compute likelihood"""
    dat.probML = dat.probML.astype(float)
    
    Pyisy=dat['probML']
    tau=dat['tau'] # Voor het lab gedeelte, pred_z nodig, voor noisy pred_tau. Hiervoor tau_new_adj gebruiken waar dit al is bewerkt.
    contr_psi = dat['contr_psi']
    
    # l_lab_ml = (1-tau)*np.log(Pyisy) + tau * np.log(contr_psi)
    l_lab_ml = xlogy(1-tau, Pyisy+1e-20) +xlogy(tau, contr_psi+1e-20)

    sum_l_lab_ml = sum(l_lab_ml)    

    return sum_l_lab_ml

#######################################################################################################################################

def afkap(X):
    """Compute threshold for z based on tau (=input)"""
    correct = 1 - np.mean(X)
    afkap = np.quantile(X, correct)
    return afkap    

#######################################################################################################################################
  
def X_pi_fit_adjust(X_pi_fit_init, y_pi_fit, X_pi_init):
    X_pi = X_pi_init.copy()
    comb_fit = X_pi_fit_init.copy()
    cols = list(X_pi_fit_init.columns)
    cols.remove("const")
    comb_fit['y'] = y_pi_fit

    for col in cols:
        if comb_fit[comb_fit[col] == 1]['y'].var() < 0.000001:
            comb_fit.drop([col], axis=1, inplace=True) 
            X_pi.drop([col], axis=1, inplace=True)
    comb_fit.drop(['y'], axis=1, inplace=True)
    
    return comb_fit, X_pi

#######################################################################################################################################

def get_errorper(config, error_type):
    special_case = config.get(error_type, None)
    if special_case != None:
        errorper = special_case['errorper']
    else:
        errorper = config['errorper']
    return errorper

#######################################################################################################################################

def select_model(model_name, params):
    calibrate = True
    
    if model_name == 'NB':
        model = MultinomialNB(**params[model_name])
    elif model_name == 'SVM':
        model = SVC(**params[model_name])
    elif model_name == 'RF':
        model = RandomForestClassifier(**params[model_name])
    elif model_name == 'XG':
        model = XGBClassifier(**params[model_name])
    elif model_name == 'LR':
        model = LogisticRegression(**params[model_name])
        calibrate = False
    elif model_name == 'GB':
        model = GradientBoostingClassifier(**params[model_name])
    else:
        model = None

    return model, calibrate, params[model_name]

#######################################################################################################################################

def lq_log(message, to_console=True):
    logging.info(message)
    if to_console:
        print(message)

#######################################################################################################################################
