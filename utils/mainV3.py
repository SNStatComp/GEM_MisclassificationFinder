import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from utils.tau_functions import tau_epsilon, tau_epsilon_init, tau_psi, psi_table, R2_entropy
from utils.utils import MyLabelEncoder, llh_function, afkap, performance, \
                    conditions_TF, eR2, get_sd, rel_ent_y_noisy, eR2_y_gold, Q_pi_function, Q_ml_function, lq_log
from statistics import stdev


class LabelQuality:

    ########################################################################################

    def __init__(self, X, yplus, config):
        """
        Initialization of the LabelQuality object.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix
            Feature matrix. Can be:
            - NumPy array of shape (n_samples, n_features)
            - pandas DataFrame (features in columns)
            - scipy.sparse.csr_matrix for sparse high-dimensional data
        yplus : pandas.DataFrame
            Target labels dataframe of the shape (n_samples, n_columns). 
            Dataframe should contain column names defined self.observedLabel and self.trueLabel
        config: dict
            Configuration settings for the EM algorithm (gold column, true label column, e.g.)
        """

        self._read_config(config)
        self.X = X
        self.yplus = yplus
        self.n_label = yplus[self.observedLabel].nunique()
        self.yplus[self.observedLabel] = self.yplus[self.observedLabel].astype('str') # LSZU DPGS labelencoder gaat niet goed als dtype een float of int is, convert to string
        self.yplus[self.trueLabel] = self.yplus[self.trueLabel].astype('str')
    
        self.le = MyLabelEncoder()

        if self.trueLabel != None:

            yplus_gold = self.yplus[self.yplus[self.gold]==1] # Label encoding op gouden set     
            self.le.fit(pd.concat((pd.concat([yplus_gold[self.observedLabel], yplus_gold[self.trueLabel]]), pd.concat([self.yplus[self.observedLabel], self.yplus[self.trueLabel]]))))
            self.yplus['observedLabel_enc'] = self.le.transform(yplus[self.observedLabel])
            self.yplus['trueLabel_enc'] = self.le.transform(yplus[self.trueLabel])
        else:
            self.le.fit(yplus[self.observedLabel])
            self.yplus['observedLabel_enc'] = self.le.transform(yplus[self.observedLabel])
            self.yplus['trueLabel_enc'] = self.yplus['observedLabel_enc']

        self.Mstep = yplus.copy() # LSZU check of yplus verandert 
        self.Mstep['cat'] = self.Mstep[self.gold].apply(lambda x: 'gold' if x==1 else 'noisy')

        sbilist_gold = self.Mstep[self.Mstep[self.gold]==1][self.observedLabel].unique()
        if self.sample != None:

            train = (yplus[self.gold]==1) | (yplus[self.sample]==1) | (yplus[self.trueLabel].isin(self.labels_exclude))
           
            if self.test_col is None:
                self.index_test = yplus[(yplus[self.gold]==0) & (yplus[self.sample]==0) & (yplus[self.observedLabel].isin(sbilist_gold))& (~yplus[self.observedLabel].isin(self.labels_exclude))].index
            else:
                self.index_test = yplus[(yplus[self.test_col]==1)  & (yplus[self.observedLabel].isin(sbilist_gold))& (yplus[self.sample]==0) & (~yplus[self.observedLabel].isin(self.labels_exclude))].index

            self.Mstep.loc[self.Mstep[self.sample]==1, 'cat'] = 'auditsample'
        else:
         
            train = (yplus[self.gold]==1) | (yplus[self.trueLabel].isin(self.labels_exclude)) # LSZU; test -> op observedlabel, train op truelabel

            if self.test_col is None:
                self.index_test = yplus[(yplus[self.gold]==0) & (yplus[self.observedLabel].isin(sbilist_gold)) & (~yplus[self.observedLabel].isin(self.labels_exclude))].index #alleen items uit testset meenemen voor performance
            else:
                self.index_test = yplus[(yplus[self.test_col]==1) & (yplus[self.observedLabel].isin(sbilist_gold)) & (~yplus[self.observedLabel].isin(self.labels_exclude))].index
           

        self.index_train_em = yplus[train].index
      
        if self.bg_col != None:
            self.le_background = MyLabelEncoder()
            self.Mstep['background_enc'] = self.le_background.fit_transform(self.Mstep[self.bg_col].astype('str'))
            self.dummies = pd.get_dummies(yplus[self.bg_col], prefix=self.bg_col)
            if type(self.X) == sparse.csr_matrix:
                self.X = sparse.hstack([self.X, self.dummies]).tocsr()
            elif isinstance(self.X, pd.DataFrame):
                self.X = pd.concat([self.X, self.dummies], axis=1)
            else:
                self.X = np.hstack([self.X, self.dummies])

        if self.manipulated != None: 
            self.Mstep['manipulated'] = self.yplus[self.manipulated]
        else:
            self.Mstep['manipulated']= np.where((self.Mstep['trueLabel_enc']== self.Mstep['observedLabel_enc']), 0, 1) 
          

    ########################################################################################
    
    def _read_config(self, config):
        """
        Read configuration settings and assign them to instance attributes.

        Parameters
        ----------
        config: dict
            Configuration settings for the EM algorithm (gold column, true label column, e.g.)
            - "bg_col" : str
            - "gold_col" : str
            - "sample_col" : str
            - "observedLabel_col" : str
            - "trueLabel_col" : str
            - "base_clf" : 
            - "calibrate" : 
            - "maxiterations" : int
            - "maxiterations_short" : int
            - "miniterations" : int
            - "performance_col" : str
            - "GHTables" : 
            - "scenarios" : str
            - "wmodel" : str
            - "alpha" : float
            - "tau_choice" : str
            - "threshold" : int
            - "tol" : float
            - "manipulated" : str
            - "test_col" : str
            - "r2_grenswaarde" : float
            - "it_overduidelijkefout" : int
            - "min_teratie" : int
            - "include_odf" : boolean
            - "prior" : boolean
            - "controle" : 
            - "labels_exclude" : list
        """

        self.bg_col = config.get('bg_col', None)
        self.gold = config.get('gold_col', 'gold')
        self.sample = config.get('sample_col', None)
        self.observedLabel = config.get('observedLabel_col', 'label')
        self.trueLabel = config.get('trueLabel_col', None)
        self.base_clf = config['base_clf']
        self.calibrate = config['calibrate']
        self.maxiterations = config.get('maxiterations', 200) 
        self.maxiterations_short = config.get('maxiterations_short', 10) 
        self.miniterations = config.get('miniterations', 10) 
        self.performance_col = config['performance_col']
        self.GHTables = config.get('GHTables', None)
        self.scenarios = config['scenarios'] # Remove? (self.errortypes en self.percentage al weg)
        self.wmodel = config.get('wmodel')  
        self.alpha = config.get('alpha', 1)
        self.tau_choice = config.get('tau_choice', 'psi')
        self.threshold = config.get('threshold', 5)
        self.tol = config.get('tol', 0.0001) # tolerance for changes in likelihood
        self.manipulated = config.get('manipulated', None)
        self.test_col = config.get('test_col', None)
        self.r2_grenswaarde = config.get('r2_grenswaarde', 0.5) 
        self.it_overduidelijkefout = config.get('it_overduidelijkefout', 1)
        self.min_iteratie = config.get('min_iteratie', 0)
        self.include_odf = config.get('include_odf', False)
        self.prior = config.get('prior', None)
        self.controle = config.get('controle', None)
        self.labels_exclude = config.get('labels_exclude', [])


    ########################################################################################

    def _fit_pp(self, model, y_train, X_train, X_test):
        """
        Fit the model and generate prediction probabilities.
        
        Parameters
        ----------
        model : object
            A model instance implementing '.fit()', '.predict()' and '.predict_proba()'.
            For example, a scikit-learn classifier.
        X_train : numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix
            Training feature matrix of shape (n_samples_train, n_features).
        y_train : numpy.ndarray, pandas.Series
            Training target vector of shape (n_samples_train,).
        X_test : numpy.ndarray, pandas.DataFrame or scipy.sparse.csr_matrix
            Test feature matrix of shape (n_samples_test, n_features).

        Returns
        -------
        class_predictions : numpy.ndarray
            Predicted class labels for 'X_test'.
        prob_predictions : numpy.ndarray
            Predicted class probabilities for 'X_test' of shape (n_samples_test, n_classes).
        """

        model.fit(X_train, y_train)
        class_prediction = model.predict(X_test)
        prob_prediction = model.predict_proba(X_test)
        return class_prediction, prob_prediction

    ########################################################################################

    def _probML(self, setseed):
        """
        Fit the base classifier (if it has not been initialized in the configuration) and generate predictions.
        The function trains the model twice:

        1. On the training set ('X_train', 'y_train') to produce out-of-sample predictions.
        2. On the full dataset ('X_full', 'y_full') to produce final class labels and
        class probabilities for all available samples.

        Parameters
        ----------
        setseed : int
            Random seed used for KFold cross-validation. Ensures that the data splits are reproducible
            across runs.

        Returns
        -------
        probML_init_gold : numpy.ndarray
            1-D array containing for each sample the predicted probability of the class selected for that sample.
            The model is trained on training samples and the i-th value is computed as 'probML_init_gold[i] = y_prob[i][y_class[i]]'. 
        probML_init_full : numpy.ndarray
            1-D array containing for each sample the predicted probability of the class selected for that sample.
            The model is trained on the full sample and the i-th value is computed as 'probML_init_full[i] = y_prob[i][y_class[i]]'. 
        """

        obs_popverdeling = (self.yplus['observedLabel_enc'].value_counts() / len(self.yplus['observedLabel_enc'])).sort_index().tolist()
    
        if self.prior is True:               
            if isinstance(self.base_clf, MultinomialNB):
                self.base_clf.class_prior = obs_popverdeling
            
        if self.calibrate:
            model = CalibratedClassifierCV(self.base_clf, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=setseed))
        else:
            model = self.base_clf
        # print(self.yplus.loc[self.index_train_em, 'trueLabel_enc'].unique())

        # Simulaties geeft X als csr matrix, sectie geeft X als numpy array, en droge bonen geeft X als pd dataframe
        if isinstance(self.X, pd.DataFrame):
            self.y_hat_class_gold, y_hat_prob_gold = self._fit_pp(model=model, y_train=self.yplus.loc[self.index_train_em, 'trueLabel_enc'],
                                                                X_train=self.X.iloc[self.index_train_em], X_test=self.X)
        else:
            self.y_hat_class_gold, y_hat_prob_gold = self._fit_pp(model=model, y_train=self.yplus.loc[self.index_train_em, 'trueLabel_enc'],
                                                                X_train=self.X[self.index_train_em], X_test=self.X)
      
        y_hat_class_fullset, y_hat_prob_fullset = self._fit_pp(model=model, y_train=self.yplus['observedLabel_enc'], 
                                                                X_train=self.X, X_test=self.X)

        # self.y_hat_class_gold -> geen self; 
        probML_init_gold = np.array([sublist[select] for sublist, select in zip(y_hat_prob_gold, self.y_hat_class_gold)]) # LSZU: gaat iets mis met sbi hoeveelheid: IndexError: index 33 is out of bounds for axis 0 with size 25 met inladen van nieuwe odf input bestanden
        probML_init_full = np.array([sublist[select] for sublist, select in zip(y_hat_prob_fullset, y_hat_class_fullset)])

        return probML_init_gold, probML_init_full
    
    ########################################################################################

    def initial_tau_multiple(self, seed):
        """
        Initialize the tau probabilities given a specific seed [TODO; add more text with details]

        Parameters
        ----------
        seed : int 
            Random seed used as input for the probML function for reproducibility across runs

        Returns
        -------
        tau_init_list_df : 
            1-D array containing tau values
        """
        
        # probML_init_gold, probML_init_full =  self._probML(seed, prior)
        probML_init_gold, probML_init_full =  self._probML(seed)
        
        if self.sample == None:
            # predicted pi is based on all units (we do not exclude the units of the gold set)
            pi_gold_est = sum([y_hat!=y_true for y_hat, y_true in zip(self.y_hat_class_gold, self.yplus['observedLabel_enc'])]) / len(self.yplus)
            pi_options = pd.DataFrame([{'o1': 0.05, 'o2': 0.1, 'o3': 0.25, 'o4': 0.5, 'o5': pi_gold_est}])
        else:                
            samp = self.Mstep[self.Mstep[self.sample]==1]
            means = samp.groupby('background_enc', group_keys=False)['manipulated'].apply(lambda group_df: group_df.mean()).values
            n_sample = samp.groupby('background_enc', group_keys=False)['manipulated'].apply(lambda group_df: group_df.size).values
            means = [0.01 if x==0 else x for x in means]
            meanssd = [get_sd(item, n) for item, n in zip(means, n_sample)]
            # optie 1 : mean
            # optie 2: means + 2 * sd
            meanplus2 = [m + sd * 2 for m, sd in zip(means,meanssd)]
            # optie 3: means + 2 * sd
            meanminus2 = [m - sd * 2 for m, sd in zip(means,meanssd)]

            pi_options = [[means[item], meanplus2[item], meanminus2[item]] for item in self.Mstep['background_enc']]
            # pi_option2 = [meanplus2[item-1] for item in self.yplus[self.bg_col]]
            # pi_option3 = [meanminus2[item-1] for item in self.yplus[self.bg_col]]

            pi_options = pd.DataFrame(pi_options, columns=['o1', 'o2', 'o3'])
            pi_options[pi_options < 0] = 0.001
            pi_options[pi_options > 1] = 0.999

        w_options = [0.25,0.5,1]

        tau_init_list = []
        for w in w_options:
            weighted_probML_init = w * probML_init_gold + (1-w) * probML_init_full
            for column in pi_options.columns:
                if self.sample == None:
                    pi_opt = pi_options[column].values[0]
                else:
                    pi_opt = pi_options[column]
                d1 = pi_opt * (1 - weighted_probML_init)
                d2 = (1 - pi_opt) * weighted_probML_init
                tau_init_list.append(d1 / (d1 + d2))

        tau_init_list_df = pd.DataFrame(tau_init_list, columns = None)
        tau_init_list_df = np.transpose(tau_init_list_df)
        tau_init_list_df.columns = range(tau_init_list_df.shape[1])

        return tau_init_list_df

    ########################################################################################

    def _wmodel(self, sce): 
        """
        Returns the dataframe with a weight column, computed based on a specific scenario.

        Parameters
        ----------
        sce : str
            Predefined scenario, (e.g. 'sc1', 'sc2').

        Returns 
        -------
        data['weight'] : pandas.series
            Pandas series containing the weights computed from a specific scenario
        """
        # NSCN: let op alleen de weegmodellen bij sc1 + w4b zijn geschikt voor sectie R. De rest is nog niet aangepast.

        # Alleen bij scenario 3 en 4 weging voor gold + audit
        # weegmodel moet wel opgegeven zijn om weging toe te gaan passen 
        if (sce=="sc3" or sce=="sc4") and len(self.wmodel) != 0:
        
            # Arnout: bij def EM() staat:  self.Mstep["X_cat"] = self.Mstep[self.bg_col]
            # Als je dit ook doet en vervolgens "X_cat" gebruikt bij onderstaande code in plaats van "X_cat" 
            # dan heb je het juiste achtergrondkenmerk voor sc3 en sc4.
            self.Mstep["X_cat"] = self.Mstep[self.bg_col]

            # Totale data kopie
            data = self.Mstep[["cat", "observedLabel_enc", "X_cat"]].copy()

            # size per category (gold, audit, noisy) and stratum (=bgcol)
            cat_strat = data.groupby(['cat', 'X_cat']).size().reset_index(name = 'total')
            cat_strat = cat_strat.set_index(['cat', 'X_cat']) #compleet maken van alle cases 
            mux = pd.MultiIndex.from_product([cat_strat.index.levels[0], cat_strat.index.levels[1]], names = ['cat', 'X_cat'])
            cat_strat = cat_strat.reindex(mux, fill_value = 0).reset_index()

            # size per category (gold, audit, noisy) and SBI
         
            cat_SBI = data.groupby(['cat', 'observedLabel_enc']).size().reset_index(name = 'total')  
            cat_SBI = cat_SBI.set_index(['cat', 'observedLabel_enc']) 
            # mux2 = pd.MultiIndex.from_product([cat_SBI.index.levels[0], cat_SBI.index.levels[1]], names = ['cat', 'trueLabel_enc'])
            mux2 = pd.MultiIndex.from_product([cat_SBI.index.levels[0], cat_SBI.index.levels[1]], names = ['cat', 'observedLabel_enc']) 
            cat_SBI = cat_SBI.reindex(mux2, fill_value = 0).reset_index()

            # size per category (gold, audit, noisy), SBI and stratum
        
            # mux3 = pd.MultiIndex.from_product([cat_SBI_strat.index.levels[0], cat_SBI_strat.index.levels[1], cat_SBI_strat.index.levels[2]], names = ['cat', 'trueLabel_enc', 'X_cat'])
            cat_SBI_strat = data.groupby(['cat', 'observedLabel_enc', 'X_cat']).size().reset_index(name = 'total') 
            cat_SBI_strat = cat_SBI_strat.set_index(['cat', 'observedLabel_enc', 'X_cat']) 
            mux3 = pd.MultiIndex.from_product([cat_SBI_strat.index.levels[0], cat_SBI_strat.index.levels[1], cat_SBI_strat.index.levels[2]], names = ['cat', 'observedLabel_enc', 'X_cat']) # DPGS LSZU observed ipv true label
            cat_SBI_strat = cat_SBI_strat.reindex(mux3, fill_value = 0).reset_index()

            # Ngx: size per SBI and stratum (Arnout: kortere code gemaakt)
            # mux4 = pd.MultiIndex.from_product([cat_SBI_strat_pop.index.levels[0], cat_SBI_strat_pop.index.levels[1]], names = ['trueLabel_enc', 'X_cat'])
            at_SBI_strat_pop = cat_SBI_strat.groupby(['observedLabel_enc', 'X_cat']).sum().reset_index() # DPGS LSZU observed ipv true label

            #N+x: aantal eenheden met stratum x in populatie
            #Nu komt alles voor, in andere toepassing ook dit compleet maken
            # pop_stratum = data.groupby(['X_cat']).size().reset_index()
            pop_stratum = cat_strat.groupby(['X_cat']).sum().reset_index() # Arnout gebruik je wel de compleet gemaakte data.

            #n+xA: aantal eenheden met stratum x in auditsteekproef
            audit_stratum = cat_strat.loc[cat_strat["cat"] == "auditsample", ].copy().reset_index(drop = True)
            
            #n+xB: aantal eenheden met stratum x in gouden set
            gold_stratum = cat_strat.loc[cat_strat["cat"] == "gold",].copy().reset_index(drop = True)

            #Ng+: aantal eenheden met SBI=g in populatie
            #Nu komt alles voor, in andere toepassing ook dit compleet maken
            # pop_SBI = data.groupby(['trueLabel_enc']).size().reset_index()
            # pop_SBI = cat_SBI.groupby(['trueLabel_enc']).sum().reset_index() # Arnout: nu gebruik je wel de compleet gemaakte data
            pop_SBI = cat_SBI.groupby(['observedLabel_enc']).sum().reset_index() # Arnout: nu gebruik je wel de compleet gemaakte data # DPGS LSZU observed ipv true label

            #ng+A: aantal eenheden met SBI=g in auditsteekproef
            audit_SBI = cat_SBI.loc[cat_SBI["cat"] == "auditsample", ].copy().reset_index(drop = True)

            #ng+B: aantal eenheden met SBI=g in gouden set 
            gold_SBI = cat_SBI.loc[cat_SBI["cat"] == "gold", ].copy().reset_index(drop = True)

            #ngxA: aantal eenheden met SBI = g en stratum x in auditsteekproef
            audit_SBI_strat = cat_SBI_strat.loc[cat_SBI_strat["cat"] == "auditsample", ].copy().reset_index(drop = True)

            #ngxB: aantal eenheden met SBI = g en stratum x in gouden set
            gold_SBI_strat = cat_SBI_strat.loc[cat_SBI_strat["cat"] == "gold", ].copy().reset_index(drop = True)

            if self.wmodel == "w2": 
                data['weight'] = np.where(data['cat'] == 'auditsample', 0, np.where(data['cat'] == "gold", 1, 0))

            elif self.wmodel == "w3": 
                
                # Gewicht 3 berekenen 
                # Gewicht A voor auditsample met SBI = g en stratum = x
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']
                w3_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w3A': w3_A, 'cat': 'auditsample'})

                # Gewicht B voor gouden set met SBI = g en stratum = x 
                w3_B = pop_SBI['total']/gold_SBI['total']
                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'}) 

                # Gewicht 3 toekennen aan dataset door kolom W3 te maken
                # Als cat = auditsample, dan w3_A koppelen aan X_cat categorieen
                # Als cat = gold, dan w3_B koppelen aan SBI
                # Dit koppelen aan data, later wordt selectie op train gemaakt en dan zit het juiste gewicht er al bij. 
                # index train_em nodig, maar welk object?

                data = pd.merge(data, w3_A_df, on = ['X_cat', 'cat'], how = "left")
                # data = pd.merge(data, w3_B_df, on = ['trueLabel_enc', 'cat'], how = "left")
                data = pd.merge(data, w3_B_df, on = ['observedLabel_enc', 'cat'], how = "left") # DPGS LSZU observed ipv true label

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data[['w3A', 'w3B']] = data[['w3A', 'w3B']].fillna(0)

                data['weight'] = data['w3A'] + data['w3B']

            elif self.wmodel == "w4a":

                w4a = pop_stratum['total']/(audit_stratum['total']+gold_stratum['total'])
                w4a = w4a.replace([np.inf, -np.inf], np.nan) #hier miss niet nodi, maar voor zekerheid
                w4a = w4a.fillna(0)

                w4a_df = pd.DataFrame({'X_cat': pop_stratum['X_cat'], 'weight': w4a})
                data = pd.merge(data, w4a_df, on = ['X_cat'], how = "left")

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data.loc[data['cat'] == "noisy", 'weight'] = 0

            elif self.wmodel == "w4b":

                w4b = pop_SBI['total']/(audit_SBI['total']+gold_SBI['total'])
                w4b = w4b.replace([np.inf, -np.inf], np.nan) #hier miss niet nodi, maar voor zekerheid
                w4b = w4b.fillna(0)

                w4b_df = pd.DataFrame({'observedLabel_enc': pop_SBI['observedLabel_enc'], 'weight': w4b}) 
                data = pd.merge(data, w4b_df, on = ['observedLabel_enc'], how = "left") 

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data.loc[data['cat'] == "noisy", 'weight'] = 0

            elif self.wmodel == "w4c": 

                w4c = cat_SBI_strat_pop['total']/(audit_SBI_strat['total']+gold_SBI_strat['total'])
                w4c = w4c.replace([np.inf, -np.inf], np.nan)
                w4c = w4c.fillna(0)

                w4c_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'weight': w4c}) 
                data = pd.merge(data, w4c_df, on = ['observedLabel_enc', 'X_cat'], how = "left")

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data.loc[data['cat'] == "noisy", 'weight'] = 0

            elif self.wmodel == "w4c_extra": 

                w4c = cat_SBI_strat_pop['total']/(audit_SBI_strat['total']+gold_SBI_strat['total'])
                w4c = w4c.replace([np.inf, -np.inf], np.nan)
                w4c = w4c.fillna(0)

                wg_plus = ((pop_SBI['total'].sum())/25)/pop_SBI['total']
                wg_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'wg_plus': wg_plus}) 

                w4c_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'w4c': w4c}) 
                w4c_df = pd.merge(w4c_df, wg_df, on = 'observedLabel_enc', how = "left")
                w4c_df["weight"] = w4c_df.loc[:,"w4c"]*w4c_df.loc[:,"wg_plus"]

                data = pd.merge(data, w4c_df, on = ['observedLabel_enc', 'X_cat'], how = "left")

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data.loc[data['cat'] == "noisy", 'weight'] = 0
            
            elif self.wmodel == "w5a":
                w5_A = (pop_stratum['total']/2)/audit_stratum['total']
                w5_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'wt_a': w5_A, 'cat': 'auditsample'})

                # dgx_B, w3_B hiervoor nodig
                w3_B = pop_SBI['total']/gold_SBI['total']

                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'}) 
                ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'observedLabel_enc'], how = "left") 
                ngx_b["nd"] = (ngx_b['total']*ngx_b['w3B'])
                
                nd_sum = (ngx_b.groupby("X_cat")["nd"].sum().reset_index(name = 'wg_cat')) #bij SBI is index en data labels gelijk, bij X_cat niet, dus opletten!
                wg_cat = (pop_stratum['total']/2)/nd_sum['wg_cat']
                wg_cat = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'wg_cat': wg_cat})
                ngx_b2 = pd.merge(ngx_b, wg_cat, on = ['X_cat'], how = "left")
                ngx_b2["wt_b"] = ngx_b2["w3B"]*ngx_b2["wg_cat"]

                data = pd.merge(data, w5_A_df, on = ['X_cat', 'cat'], how = "left")
                data = pd.merge(data, ngx_b2, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 
            
                data['weight'] = np.where(data['cat'] == 'auditsample', data['wt_a'], np.where(data['cat'] == "gold", data['wt_b'], 0))

            elif self.wmodel == "w5b":
                
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']
                w3_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w3A': w3_A, 'cat': 'auditsample'})
                ngx_a = pd.merge(audit_SBI_strat, w3_A_df, on = ['cat', 'X_cat'], how = "left")
                ngx_a["nd"] = (ngx_a['total']*ngx_a['w3A'])
                wg_sbi = ((pop_SBI['total']/2)/ngx_a.groupby("observedLabel_enc")["nd"].sum()).reset_index(name = 'wg_sbi') #hoe index naar = trueLabel_enc krijgen? #DPGS LSZU observed ipv true label
                wg_sbi = wg_sbi.replace([np.inf, -np.inf], np.nan) #dit nu toegevoegd, ook bij 5b_extra
                wg_sbi = wg_sbi.fillna(0)              
                ngx_a2 = pd.merge(ngx_a, wg_sbi, left_on = ['observedLabel_enc'], right_on = ['index'], how = "left") #index = trueLabel_enc, betere optie?
                ngx_a2["wt_a"] = ngx_a2["w3A"]*ngx_a2["wg_sbi"]

                # gewicht voor SBI
                weight_SBI = (pop_SBI['total']/2)/gold_SBI['total']          
                w5_sbi = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'wt_b': weight_SBI}) 

                data = pd.merge(data, ngx_a2, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 
                data = pd.merge(data, w5_sbi, on = ['observedLabel_enc'], how = "left") 

                data['weight'] = np.where(data['cat'] == 'auditsample', data['wt_a'], np.where(data['cat'] == "gold", data['wt_b'], 0))

            elif self.wmodel == "w5b_extra":
                # Gewicht 3 berekenen 
                # Gewicht A voor auditsample met SBI = g en stratum = x
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']
                w3_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w3A': w3_A, 'cat': 'auditsample'})

                # Gewicht B voor gouden set met SBI = g en stratum = x 
                w3_B = pop_SBI['total']/gold_SBI['total']
                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'}) 
                
                #w3_A koppelen op X_cat voor audit, W3_B koppelen op SBI voor gold 
                ngx_a = pd.merge(audit_SBI_strat, w3_A_df, on = ['cat', 'X_cat'], how = "left")
                ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'observedLabel_enc'], how = "left") 

                # gewicht voor audit sample
                N_50 = (pop_SBI['total'].sum())/50
                ngx_a["nd"] = (ngx_a['total']*ngx_a['w3A'])
                wg_sbi = (N_50/ngx_a.groupby("observedLabel_enc")["nd"].sum()).reset_index(name = 'wg_sbi') 
                wg_sbi = wg_sbi.replace([np.inf, -np.inf], np.nan)
                wg_sbi = wg_sbi.fillna(0)
                ngx_a2 = pd.merge(ngx_a, wg_sbi, on = ['observedLabel_enc'], how = "left")  
                ngx_a2["wt_a"] = ngx_a2["w3A"]*ngx_a2["wg_sbi"]

                # gewicht voor SBI
                weight_SBI = N_50/gold_SBI['total']
                w5_sbi = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'wt_b': weight_SBI}) 

                data = pd.merge(data, ngx_a2, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 
                data = pd.merge(data, w5_sbi, on = ['observedLabel_enc'], how = "left")  

                data['weight'] = np.where(data['cat'] == 'auditsample', data['wt_a'], np.where(data['cat'] == "gold", data['wt_b'], 0))

            elif self.wmodel == "w5c": 
                w5c_a = (cat_SBI_strat_pop['total']/2)/audit_SBI_strat['total']
                # w5c_a_df = pd.DataFrame({'trueLabel_enc':cat_SBI_strat_pop['trueLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'auditsample', 'w5c_a': w5c_a})
                w5c_a_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'auditsample', 'w5c_a': w5c_a}) # DPGS LSZU observed ipv true label
                w5c_b = (cat_SBI_strat_pop['total']/2)/gold_SBI_strat['total']
                # w5c_b_df = pd.DataFrame({'trueLabel_enc':cat_SBI_strat_pop['trueLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'gold', 'w5c_b': w5c_b})
                w5c_b_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'gold', 'w5c_b': w5c_b})  # DPGS LSZU observed ipv true

                data = pd.merge(data, w5c_a_df, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left")  
                data = pd.merge(data, w5c_b_df, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 

                data['weight'] = np.where(data['cat'] == 'auditsample', data['w5c_a'], np.where(data['cat'] == "gold", data['w5c_b'], 0))

            elif self.wmodel == "w6a":

                # Gewicht 3 berekenen 
                # Gewicht A voor auditsample met SBI = g en stratum = x
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']

                # Gewicht B voor gouden set met SBI = g en stratum = x 
                w3_B = pop_SBI['total']/gold_SBI['total']
                # w3_B_df = pd.DataFrame({'trueLabel_enc':pop_SBI['trueLabel_enc'], 'w3B': w3_B, 'cat': 'gold'})
                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'})  # DPGS LSZU observed ipv true label
                
                #W3_B koppelen op SBI voor gold 
                # ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'trueLabel_enc'], how = "left")
                ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'observedLabel_enc'], how = "left")  # DPGS LSZU observed ipv true label

                #nd berekenen en sommatie over SBI, op basis van B --> ook voor auditsample
                ngx_b["nd"] = (ngx_b['total']*ngx_b['w3B'])
                wg_cat = (ngx_b.groupby("X_cat")["nd"].sum()).reset_index(name = 'wg_cat')
                
                w6a_a = (pop_stratum['total']/(pop_stratum['total']-gold_stratum['total']+wg_cat['wg_cat']))*w3_A
                # Klopt het dat w3_A direct vermenigvuldigd kan worden, geen koppeling met SBI x stratum nodig?
                w6a_a_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w6a_a': w6a_a, 'cat': 'auditsample'})
                
                w6a_b = (pop_stratum['total']/(pop_stratum['total']-gold_stratum['total']+wg_cat['wg_cat']))
                w6a_b_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w6b': w6a_b})
                ngx_b2 = pd.merge(ngx_b, w6a_b_df, on = ["X_cat"], how = "left")
                ngx_b2['wt_b'] = ngx_b2['w3B']*ngx_b2['w6b']
        
                data = pd.merge(data, w6a_a_df, on = ['X_cat', 'cat'], how = "left")
                data = pd.merge(data, ngx_b2, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left")  
                
                data['weight'] = np.where(data['cat'] == 'auditsample', data['w6a_a'], np.where(data['cat'] == "gold", data['wt_b'], 0))
                
            elif self.wmodel == "w6b":

                # Gewicht 3 berekenen 
                # Gewicht A voor auditsample met SBI = g en stratum = x
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']
                w3_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w3A': w3_A, 'cat': 'auditsample'})

                # Gewicht B voor gouden set met SBI = g en stratum = x 
                w3_B = pop_SBI['total']/gold_SBI['total']
                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'})  
                
                #w3_A koppelen op X_cat voor audit, W3_B koppelen op SBI voor gold 
                ngx_a = pd.merge(audit_SBI_strat, w3_A_df, on = ['cat', 'X_cat'], how = "left")
                ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'observedLabel_enc'], how = "left") 

                ngx_a["nd"] = (ngx_a['total']*ngx_a['w3A'])
                wg_sbi = (ngx_a.groupby("observedLabel_enc")["nd"].sum()).reset_index(name = 'wg_sbi')  
                
                w6b_a = (pop_SBI['total']/(wg_sbi['wg_sbi']+pop_SBI['total'])) #deels, later vermenigvuldigen met W3A
                w6b_a_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w6b_a': w6b_a})  

                w6b_b = (pop_SBI['total']/(wg_sbi['wg_sbi']+pop_SBI['total']))*w3_B
                w6b_b_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w6b_b': w6b_b, 'cat': 'gold'})  
               
                # voor auditsample met df rekenen
                ngx_a2 = pd.merge(ngx_a, w6b_a_df, on = ["observedLabel_enc"], how = "left")  
                ngx_a2['wt_a'] = ngx_a2['w3A']*ngx_a2['w6b_a']

                data = pd.merge(data, ngx_a2, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left")  
                data = pd.merge(data, w6b_b_df, on = ['observedLabel_enc', 'cat'], how = "left") 
                
                data['weight'] = np.where(data['cat'] == 'auditsample', data['wt_a'], np.where(data['cat'] == "gold", data['w6b_b'], 0))
                
            elif self.wmodel == "w6c": 
                # Gewicht 3 berekenen 
                # Gewicht A voor auditsample met SBI = g en stratum = x
                w3_A = (pop_stratum['total'] - gold_stratum['total'])/audit_stratum['total']
                w3_A_df = pd.DataFrame({'X_cat':pop_stratum['X_cat'], 'w3A': w3_A, 'cat': 'auditsample'})

                # Gewicht B voor gouden set met SBI = g en stratum = x 
                w3_B = pop_SBI['total']/gold_SBI['total']
                # w3_B_df = pd.DataFrame({'trueLabel_enc':pop_SBI['trueLabel_enc'], 'w3B': w3_B, 'cat': 'gold'})
                w3_B_df = pd.DataFrame({'observedLabel_enc':pop_SBI['observedLabel_enc'], 'w3B': w3_B, 'cat': 'gold'})  # DPGS LSZU observed ipv true label
                
                #w3_A koppelen op X_cat voor audit, W3_B koppelen op SBI voor gold 
                ngx_a = pd.merge(audit_SBI_strat, w3_A_df, on = ['cat', 'X_cat'], how = "left")
                # ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'trueLabel_enc'], how = "left")
                ngx_b = pd.merge(gold_SBI_strat, w3_B_df, on = ['cat', 'observedLabel_enc'], how = "left")  # DPGS LSZU observed ipv true label

                w6c_a = (cat_SBI_strat_pop['total']/((ngx_a['total']*ngx_a['w3A'])+(ngx_b['total']*ngx_b['w3B'])))*ngx_a['w3A']
                w6c_b = (cat_SBI_strat_pop['total']/((ngx_a['total']*ngx_a['w3A'])+(ngx_b['total']*ngx_b['w3B'])))*ngx_b['w3B']

                w6c_a_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'auditsample', 'w6c_a': w6c_a})  # DPGS LSZU observed ipv true label
                w6c_b_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'cat': 'gold', 'w6c_b': w6c_b})  # DPGS LSZU observed ipv true label
                
                data = pd.merge(data, w6c_a_df, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 
                data = pd.merge(data, w6c_b_df, on = ['observedLabel_enc', 'X_cat', 'cat'], how = "left") 

                data['weight'] = np.where(data['cat'] == 'auditsample', data['w6c_a'], np.where(data['cat'] == "gold", data['w6c_b'], 0))

        elif (sce=="sc1") and len(self.wmodel) != 0:

            # Totale data kopie
            data = self.Mstep[["cat","observedLabel_enc"]].copy() #NSCN: voor sectie R moet dit observed label zijn, want true is onbekend!!

            # size per category (gold, audit, noisy) and SBI
            cat_SBI = data.groupby(['cat', 'observedLabel_enc']).size().reset_index(name = 'total')
            cat_SBI = cat_SBI.set_index(['cat', 'observedLabel_enc']) #compleet maken van alle cases 
            mux2 = pd.MultiIndex.from_product([cat_SBI.index.levels[0], cat_SBI.index.levels[1]], names = ['cat', 'observedLabel_enc'])
            cat_SBI = cat_SBI.reindex(mux2, fill_value = 0).reset_index()

            #Ng+: aantal eenheden met SBI=g in populatie
            #Nu komt alles voor, in andere toepassing ook dit compleet maken
            # pop_SBI = data.groupby(['observedLabel_enc']).size().reset_index()
            pop_SBI = cat_SBI.groupby(['observedLabel_enc']).sum().reset_index() # Arnout: nu gebruik je wel de compleet gemaakte data

            #ng+B: aantal eenheden met SBI=g in gouden set 
            gold_SBI = cat_SBI.loc[cat_SBI["cat"] == "gold", ].copy().reset_index(drop = True)

            if self.wmodel == "w4b":

                w4b = pop_SBI['total']/(gold_SBI['total'])
                w4b = w4b.replace([np.inf, -np.inf], np.nan) #hier miss niet nodi, maar voor zekerheid
                w4b = w4b.fillna(0)

                w4b_df = pd.DataFrame({'observedLabel_enc': pop_SBI['observedLabel_enc'], 'weight': w4b})
                data = pd.merge(data, w4b_df, on = ['observedLabel_enc'], how = "left")

                # Alle noisy krijgt een gewicht van 0 van alle wegingen. 
                data.loc[data['cat'] == "noisy", 'weight'] = 0

        elif (sce=="sc2") and len(self.wmodel) != 0:
        
        # Hieronder is alles hetzelfde als bij sc3 en 4, behalve de subsets voor de auditsample, die zijn er niet hier. 
            # Arnout: bij def EM() staat:  self.Mstep["X_cat"] = self.Mstep[self.bg_col]
            # Als je dit ook doet en vervolgens "X_cat" gebruikt bij onderstaande code in plaats van "X_cat" 
            # dan heb je het juiste achtergrondkenmerk voor sc3 en sc4.
            self.Mstep["X_cat"] = self.Mstep[self.bg_col]

            # Totale data kopie
            # data = self.Mstep[["cat","trueLabel_enc", "X_cat"]].copy()
            data = self.Mstep[["cat","observedLabel_enc", "X_cat"]].copy() # DPGS LSZU observed ipv true label

            # size per category (gold, audit, noisy) and stratum (=bgcol)
            cat_strat = data.groupby(['cat', 'X_cat']).size().reset_index(name = 'total')
            cat_strat = cat_strat.set_index(['cat', 'X_cat']) #compleet maken van alle cases 
            mux = pd.MultiIndex.from_product([cat_strat.index.levels[0], cat_strat.index.levels[1]], names = ['cat', 'X_cat'])
            cat_strat = cat_strat.reindex(mux, fill_value = 0).reset_index()

            # size per category (gold, audit, noisy) and SBI
            cat_SBI = data.groupby(['cat', 'observedLabel_enc']).size().reset_index(name = 'total') # DPGS LSZU observed ipv true label
            cat_SBI = cat_SBI.set_index(['cat', 'observedLabel_enc']) #compleet maken van alle cases # DPGS LSZU observed ipv true label
            mux2 = pd.MultiIndex.from_product([cat_SBI.index.levels[0], cat_SBI.index.levels[1]], names = ['cat', 'observedLabel_enc']) # DPGS LSZU observed ipv true label
            cat_SBI = cat_SBI.reindex(mux2, fill_value = 0).reset_index()

            # size per category (gold, audit, noisy), SBI and stratum          
            cat_SBI_strat = data.groupby(['cat', 'observedLabel_enc', 'X_cat']).size().reset_index(name = 'total') # DPGS LSZU observed ipv true label
            cat_SBI_strat = cat_SBI_strat.set_index(['cat', 'observedLabel_enc', 'X_cat']) #compleet maken van alle cases # DPGS LSZU observed ipv true label
            mux3 = pd.MultiIndex.from_product([cat_SBI_strat.index.levels[0], cat_SBI_strat.index.levels[1], cat_SBI_strat.index.levels[2]], names = ['cat', 'observedLabel_enc', 'X_cat']) # DPGS LSZU observed ipv true label
            cat_SBI_strat = cat_SBI_strat.reindex(mux3, fill_value = 0).reset_index()

            # Ngx: size per SBI and stratum (Arnout: kortere code gemaakt)
            # cat_SBI_strat_pop = data.groupby(['trueLabel_enc', 'X_cat']).size().reset_index()
            # cat_SBI_strat_pop = cat_SBI_strat_pop.set_index(['trueLabel_enc', 'X_cat']) #compleet maken van alle cases 
            # mux4 = pd.MultiIndex.from_product([cat_SBI_strat_pop.index.levels[0], cat_SBI_strat_pop.index.levels[1]], names = ['trueLabel_enc', 'X_cat'])
            # cat_SBI_strat_pop = cat_SBI_strat_pop.reindex(mux4, fill_value = 0).reset_index()
            cat_SBI_strat_pop = cat_SBI_strat.groupby(['observedLabel_enc', 'X_cat']).sum().reset_index()  

            #N+x: aantal eenheden met stratum x in populatie
            #Nu komt alles voor, in andere toepassing ook dit compleet maken
            # pop_stratum = data.groupby(['X_cat']).size().reset_index()
            pop_stratum = cat_strat.groupby(['X_cat']).sum().reset_index() # Here we have complete data
            
            #n+xB: aantal eenheden met stratum x in gouden set
            gold_stratum = cat_strat.loc[cat_strat["cat"] == "gold",].copy().reset_index(drop = True)

            #Ng+: aantal eenheden met SBI=g in populatie
            #Nu komt alles voor, in andere toepassing ook dit compleet maken
            pop_SBI = cat_SBI.groupby(['observedLabel_enc']).sum().reset_index() # Here we have complete data

            #ng+B: aantal eenheden met SBI=g in gouden set 
            gold_SBI = cat_SBI.loc[cat_SBI["cat"] == "gold", ].copy().reset_index(drop = True)

            #ngxB: aantal eenheden met SBI = g en stratum x in gouden set
            gold_SBI_strat = cat_SBI_strat.loc[cat_SBI_strat["cat"] == "gold", ].copy().reset_index(drop = True)

            if self.wmodel == "w4a":

                w4a = pop_stratum['total']/(gold_stratum['total'])
                w4a = w4a.replace([np.inf, -np.inf], np.nan) # Perhaps not necessary, added just to be sure
                w4a = w4a.fillna(0)

                w4a_df = pd.DataFrame({'X_cat': pop_stratum['X_cat'], 'weight': w4a})
                data = pd.merge(data, w4a_df, on = ['X_cat'], how = "left")

                # If noisy, weight will be 0
                data.loc[data['cat'] == "noisy", 'weight'] = 0

            elif self.wmodel == "w4b":

                w4b = pop_SBI['total']/(gold_SBI['total'])
                w4b = w4b.replace([np.inf, -np.inf], np.nan) # Perhaps not necessary, added just to be sure
                w4b = w4b.fillna(0)

                w4b_df = pd.DataFrame({'observedLabel_enc': pop_SBI['observedLabel_enc'], 'weight': w4b})
                data = pd.merge(data, w4b_df, on = ['observedLabel_enc'], how = "left")

                # If noisy, weight will be 0
                data.loc[data['cat'] == "noisy", 'weight'] = 0

            elif self.wmodel == "w4c": 

                w4c = cat_SBI_strat_pop['total']/gold_SBI_strat['total']
                w4c = w4c.replace([np.inf, -np.inf], np.nan)
                w4c = w4c.fillna(0)

                w4c_df = pd.DataFrame({'observedLabel_enc':cat_SBI_strat_pop['observedLabel_enc'], 'X_cat': cat_SBI_strat_pop['X_cat'], 'weight': w4c}) 
                data = pd.merge(data, w4c_df, on = ['observedLabel_enc', 'X_cat'], how = "left") 

                # If noisy, weight will be 0
                data.loc[data['cat'] == "noisy", 'weight'] = 0
        else: 

            data = self.Mstep[["cat","observedLabel_enc"]].copy()  
            data['weight'] = 1
            data.loc[data['cat'] == "noisy", 'weight'] = 0

        return(data['weight'])

             
    ########################################################################################

    def EM(self, tau_init, compute_results, sce):
        """
        EM algorithm for optimization.

        Parameters
        ----------
        tau init : numpy.ndarray
            Initial tau
        compute_results : boolean
            boolean indicating if metrics should be computed
        sce : str 
            String containing the specific scenario

        Returns 
        -------
        summary: Summary results file 
        results_df: Full dataframe with complete columns 
        pi_stats: Pi probabilities and fractions
        psi_output: Tau psi transition probabilities
        confusion_matrix: Error confusion matrix 
        """
        y = self.yplus['observedLabel_enc']
        y_true = self.yplus['trueLabel_enc']
        excl_gold = self.Mstep['cat'] != "gold"                                     # True/False lijst 'noisy & audit sample
        excl_fixed = excl_gold
        # ADLN 15082022 gold -> train_em; gold_index -> train_index
        train_em = (self.Mstep['cat'] == "gold") | (self.Mstep['cat'] == 'auditsample') # was 'gold' True/False lijst 'gold' & 'audit sample'
        samp = (self.Mstep['cat'] == 'auditsample')                                 # True/False lijst 'audit sample'
        # ADLN 15082022 added: samplogreg
        samplogreg = samp[excl_gold].copy()                                         # True/False lijst 'audit sample'
        # 1. make selections and define the auxiliary X-variables 
        cat = np.array(self.Mstep['cat'])      

        # calibrate weights, depending on weights scenario
        weights = self._wmodel(sce = sce)

        # EM-algorithm (& tau-init): train set  = noisy & auditsample, testset is complement
        train_index = self.Mstep[train_em].index # gold_index renamed to train_index
        noisy_index = self.Mstep[~train_em].index 

        if not compute_results:
            tau_init = pd.Series(np.where(self.Mstep['cat'] != "noisy", self.Mstep['manipulated'], tau_init))

        tau_old = tau_init.copy()
        weight_tau_old = 1 - ((1 - tau_old) ** self.alpha)
        self.Mstep['z_old'] = tau_old
        
        if self.bg_col == None:
            self.Mstep['X_cat'] = 1
            X_pi_init = pd.DataFrame(np.array(np.ones(len(self.yplus), dtype=int)).reshape(-1, 1))
        else:
            X_pi_init = self.dummies[self.dummies.columns[:-1]]
            X_pi_init = sm.add_constant(X_pi_init)
            X_pi_init = X_pi_init.astype(int)
            self.Mstep["X_cat"] = self.Mstep[self.bg_col]

        # 2. determine the fixed transition probabilities
        if self.tau_choice == 'eps':
            y_list_Gh, y_list_OGh, sizeHg_Gh, sizeOHg_OGh, epsilon = \
                tau_epsilon_init(y, self.GHTables, self.le)

        summary = pd.DataFrame(columns=["Accuracy", "Balanced_accuracy",
                                        "TPR", "TNR", 'TPR_sample', 'TNR_sample',
                                        "R2", 'R2_y_gold', 'relEnt_y_noisy',
                                        "LLH_NOISY", "LLH_LAB_PI", "LLH_LAB_ML", "LLH_LAB", "LLH", "switches",
                                        "a", "Q_pi", "Q_ml", "D_r"])
        
        X_pi = X_pi_init.copy()            
        y_pi = tau_init.copy()
        print('y_pi:', y_pi, '\n')

        # ADLN 15082022: reset the y_pi value of the (gold and) auditsample units for first fit.
        # NSCN: condi en choices worden gematcht, dus bij gold = 0, noisy = y_pi en bij auditsample manipulated.
        condi = [(self.Mstep['cat'] == "gold"), (self.Mstep['cat'] =="noisy"), (self.Mstep['cat'] == "auditsample")] 
        choices = [0, y_pi, self.Mstep['manipulated']]
        y_pi_adj = np.select(condi, choices)
        y_pi_adj = pd.Series(y_pi_adj)
        print('y_pi_adj:', y_pi_adj, '\n')

        X_pi_fit = X_pi[excl_fixed].copy() # excl_gold: noisy set and gold set LSZU: excl_fixed
        y_pi_fit = y_pi_adj[excl_fixed].copy() # LSZU excl fixed

        # audit sample eenheden is tau_init nog niet aangepast.

        if self.bg_col != None:
            # create structure to administer th

            self.yplus['background_enc'] = self.Mstep['background_enc'].copy() 
            y_pi_fit_labels = self.yplus.loc[excl_fixed, 'background_enc'].copy() 
            fixed_tau_admin = {}
            bg_labels = self.yplus['background_enc'].unique() # from bg_col

            for l in bg_labels:
                fixed_tau_admin[l] = {
                    'label_indices': y_pi_fit_labels[y_pi_fit_labels == l].index,
                    'y_pi_fit_index': None,
                    'fixed_tau': None,
                    'weighted_fixed_fixed': None
                     }


        if compute_results:
            max_iter = self.maxiterations
        else:
            max_iter = self.maxiterations_short

        gold_set = set(self.Mstep[self.Mstep[self.gold]==1]['trueLabel_enc']) # LSZU DPGS gold niet herkent, dus naar self.gold

        if self.include_odf == True:
            self.Mstep['subtiele_fout'] = ((self.Mstep['observedLabel_enc'] != self.Mstep['trueLabel_enc']) & (self.Mstep['trueLabel_enc'].isin(gold_set))& (self.Mstep.index.isin(self.index_test))).astype(int)
            self.Mstep['overduidelijke_fout'] = ((self.Mstep['observedLabel_enc'] != self.Mstep['trueLabel_enc']) & (~self.Mstep['trueLabel_enc'].isin(gold_set))& (self.Mstep.index.isin(self.index_test))).astype(int)
        
            self.Mstep['echte_geenfout'] = np.where((self.Mstep[self.controle] == 'Ja') & (self.Mstep[self.observedLabel] == self.Mstep[self.trueLabel]) & (self.Mstep.index.isin(self.index_test)),1,0 )
            self.Mstep['echte_onbekend'] = np.where((~self.Mstep.index.isin(self.index_test)),1,0)        

        else:
            self.Mstep['echte_geenfout'] = np.where((self.Mstep[self.controle] == 'Ja') & (self.Mstep[self.observedLabel] == self.Mstep[self.trueLabel]) & (self.Mstep.index.isin(self.index_test)),1,0 )
            self.Mstep['echte_fout'] = np.where((self.Mstep[self.controle] == 'Ja') & (self.Mstep[self.observedLabel] == self.Mstep[self.trueLabel]) & (self.Mstep.index.isin(self.index_test)),0,1)

        # Start optimisation iterations
        for it in range(1, max_iter + 1):

            samplogreg = samp[excl_fixed].copy()  # samplogreg without overduidelijke fout
            ## add rows to y_pi_fit to avoid singular matrix in LR
            if self.bg_col != None:
                # X_pi_fit, X_pi = X_pi_fit_adjust(X_pi_fit_init, y_pi_fit, X_pi_init)
                for l in bg_labels:
                  
                    print(it,l)
       
                    y_tmp = y_pi_fit[fixed_tau_admin[l]['label_indices']]
                    if (y_tmp.var() < 0.000001) and (fixed_tau_admin[l]['fixed_tau']==None):
                        sel_index = y_tmp[~samplogreg].index[0] # ADLN 15082022: old code y_tmp.index[0]
                        if y_pi_fit[sel_index] < 0.5:
                            fixed_tau = y_pi_fit[sel_index] + 0.0001
                        else:
                            fixed_tau = y_pi_fit[sel_index] - 0.0001
                        
                        y_pi_fit[sel_index] = fixed_tau
                        fixed_tau_admin[l]['y_pi_fit_index'] = sel_index
                        fixed_tau_admin[l]['fixed_tau'] = fixed_tau
                        weighted_fixed_tau = 1 - ((1 - fixed_tau) ** self.alpha)
                        fixed_tau_admin[l]['weighted_fixed_tau'] = weighted_fixed_tau
                    
            #### M STEP ####
            # Maximize: NB en Logistic model    
            if y_pi_fit.var() >= 0.000001:
                updatedmodel = sm.Logit(y_pi_fit,X_pi_fit)
                fitted_LR = updatedmodel.fit(method='bfgs', maxiter=100, disp=False, skip_hessian=True, gtol= 1e-2)

            if y_pi_fit.var() >= 0.000001:
                self.Mstep['pi_est'] = fitted_LR.predict(X_pi)
            else: 
                # this is an exception when there is no bg variable involved,
                # we could have taken the same solution as with bg variable
                lq_log("y_pivar < 0.000001", to_console=False)
                if y_pi_fit.mean() < 0.01:
                    self.Mstep['pi_est'] = 0.0005
                if y_pi_fit.mean() > 0.99:
                    self.Mstep['pi_est'] = 0.9995
                    
            # self.Mstep['pi_est'] is used in the llh. That should have no contribution of the gold set.
            # the next iteration of log.regression is based on tau_new_adj and then also the 
            # units with cat = "audit sample" are corrected.
            condi = [(cat == "gold"), (cat !="gold")]
            choices = [0, self.Mstep['pi_est']]
            self.Mstep['pi_est'] = np.select(condi, choices)

            a = 0.5 # set to be able to write a to the summary file for the first iteration (value written: a * 2) 
            Q_pi = None
            Q_ml = None
            d_r = None
            Q_ml_old = None
            Q_pi_old = None
            
            if it > 1:
                llhdat_pi = {"tau":self.Mstep['tau_new_adj'], "pred_pi":self.Mstep['pi_est']}
                llhdat_pi = pd.DataFrame(llhdat_pi)

                llhdat_pi_old = {"tau":self.Mstep['tau_new_adj'], "pred_pi":pi_est_old}
                llhdat_pi_old = pd.DataFrame(llhdat_pi_old)

                Q_pi = Q_pi_function(llhdat_pi) 
                Q_pi_old = Q_pi_function(llhdat_pi_old)

                d_r = Q_pi - Q_pi_old
            
            self.base_clf.class_prior = None
            self.base_clf.fit(self.X, y, sample_weight = np.array(1 - weight_tau_old))

            if self.calibrate:
                calibrator = CalibratedClassifierCV(estimator = self.base_clf, cv = 'prefit', method = 'sigmoid')     
                if isinstance(self.X, pd.DataFrame):
                    calibrator.fit(self.X.iloc[self.index_train_em], y_true[self.index_train_em], sample_weight = weights[self.index_train_em])
                else:
                    calibrator.fit(self.X[self.index_train_em], y_true[self.index_train_em], sample_weight = weights[self.index_train_em]) # here all weights are 1.
                y_hat_probM_tilde = calibrator.predict_proba(self.X)
            else:
                y_hat_probM_tilde = self.base_clf.predict_proba(self.X)

            if it > 1:  
                llhdat_ml_old = pd.DataFrame({"probML":probML_old, "contr_psi": contr_psi, "tau": self.Mstep['tau_new_adj']})
                Q_ml_old = Q_ml_function(llhdat_ml_old)
                
                
                a = 1.0
                doorgaan = True

                while doorgaan:
                    y_hat_probM = (1-a)*y_hat_probM_old + a*y_hat_probM_tilde
                    
                    if self.tau_choice == 'eps':
                        
                        self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                            tau_epsilon(y_hat_probM, y, y_list_Gh,  sizeHg_Gh, y_list_OGh, sizeOHg_OGh, epsilon,
                                        self.Mstep['pi_est'])

                    else:

                        self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                            tau_psi(y_hat_probM, y, self.Mstep['pi_est'], 'observedLabel_enc') 


                    llhdat_ml = pd.DataFrame({"probML":probML, "contr_psi": contr_psi,"tau": self.Mstep['tau_new_adj']})
                    Q_ml = Q_ml_function(llhdat_ml)

                    doorgaan = (Q_ml < Q_ml_old - d_r) and (a > 0.01)
                    lq_log(f'it = {it}, a = {a * 2}')
                    
                    if a <= 0.01:
                        a = 0.0
                        y_hat_probM = y_hat_probM_old
                         
                        if self.tau_choice == 'eps':
                            self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                                tau_epsilon(y_hat_probM, y, y_list_Gh,  sizeHg_Gh, y_list_OGh, sizeOHg_OGh, epsilon,
                                            self.Mstep['pi_est'])
                        else:
                            self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                                tau_psi(y_hat_probM, y, self.Mstep['pi_est'], 'observedLabel_enc') 

                        llhdat_ml = pd.DataFrame({"probML":probML, "contr_psi": contr_psi,"tau": self.Mstep['tau_new_adj']})
                        Q_ml = Q_ml_function(llhdat_ml)

                    else:
                        a = a / 2

                if not compute_results:
                    lq_log(f'it = {it}, a = {a * 2}', to_console=False)
                    
            else:
                y_hat_probM = y_hat_probM_tilde
                
                if self.tau_choice == 'eps':
                    self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                        tau_epsilon(y_hat_probM, y, y_list_Gh,  sizeHg_Gh, y_list_OGh, sizeOHg_OGh, epsilon,
                                    self.Mstep['pi_est'])
                else:
                    self.Mstep['tau_new'], contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh = \
                        tau_psi(y_hat_probM, y, self.Mstep['pi_est'], 'observedLabel_enc') 

            # Save for next iteration
            y_hat_probM_old = y_hat_probM
            probML_old = probML.copy()

            self.Mstep['weight_tau_new'] = 1 - ((1 - self.Mstep['tau_new'])**self.alpha) # self.alpha op 1, moet aan elkaar gelijk zijn
            
            #### CORRECT PARAMETERS #### (make sure z = 0 for gold set and 1 for audit sample)
            ### set gold set and audit sample to true z ###

            probs_list = y_hat_probM.tolist() #nscn: extra output toevoegen om probML te vergelijken met overduidelijke fouten
            self.Mstep['entropy_R2'] = [R2_entropy(x) for x in probs_list]

            # Condition for overduidelijke fout, when to adjust. 
            if self.include_odf == True:
                if (it > self.min_iteratie):
                    if (it <= self.it_overduidelijkefout + self.min_iteratie):# Andere conditie
                        print('iteratie:', it, self.it_overduidelijkefout)
                        self.Mstep['pred_overduidelijke_fout'] = np.where(self.Mstep['entropy_R2'] <= self.r2_grenswaarde, 1, 0)
                elif it<=self.min_iteratie:
                    self.Mstep['pred_overduidelijke_fout'] = 0

            # Adjust accordingly
            if self.include_odf == True:
                condi = [(self.Mstep['overduidelijke_fout']==1), (cat == "gold"), (cat =="noisy"), (cat == "auditsample")]
                choices = [1, 0, self.Mstep['tau_new'], self.Mstep['manipulated']] # LSZU: hier komt iets bij, als self.Mstep['overduidelijke_fout] == 1 dan self.Mstep['tau_new_adj] == 1, else niks doen Done
            else:
                condi = [(cat == "gold"), (cat =="noisy"), (cat == "auditsample")]
                choices = [0, self.Mstep['tau_new'], self.Mstep['manipulated']] # LSZU: hier komt iets bij, als self.Mstep['overduidelijke_fout] == 1 dan self.Mstep['tau_new_adj] == 1, else niks doen

            self.Mstep['tau_new_adj'] = np.select(condi, choices)

            # for units with cat=="noisy" tau_new_adj can be overruled (nearly same value)            
            if self.bg_col != None:
                for l in fixed_tau_admin.keys():
                    if fixed_tau_admin[l]['fixed_tau'] != None:
                        index = fixed_tau_admin[l]['y_pi_fit_index']
                        self.Mstep.loc[index, 'tau_new_adj'] = fixed_tau_admin[l]['fixed_tau']
            if self.include_odf == True:
                choices = [1,0, self.Mstep['weight_tau_new'], self.Mstep['manipulated']] # LSZU; wat is weight_tau
            else:
                choices = [0, self.Mstep['weight_tau_new'], self.Mstep['manipulated']]
            self.Mstep['weight_tau_new_adj'] = np.select(condi, choices)

            # for units with cat=="noisy" weight_tau_new_adj can be overruled (nearly same value)            
            if self.bg_col != None:
                for l in fixed_tau_admin.keys():
                    if fixed_tau_admin[l]['fixed_tau'] != None:
                        index = fixed_tau_admin[l]['y_pi_fit_index']
                        # ADLN 15082022 old code given below
                        # self.Mstep.loc[index, 'weighted_tau_new_adj'] = fixed_tau_admin[l]['weighted_fixed_tau']
                        self.Mstep.loc[index, 'weight_tau_new_adj'] = fixed_tau_admin[l]['weighted_fixed_tau']
                                    
            ### Z ###
            self.Mstep['z'] = self.Mstep['tau_new_adj'] 
            X_cat_afkap = self.Mstep[excl_fixed].groupby('X_cat')['tau_new_adj'].apply(afkap).to_dict() 
            self.Mstep['afkap_z'] = self.Mstep['X_cat'].map(X_cat_afkap)
            # the units with cat = noisy z value is derived. For cat = gold and cat = auditsample tau_new_adj is already elem {0,1}  
            self.Mstep.loc[((self.Mstep['tau_new_adj'] >= self.Mstep['afkap_z']) & (self.Mstep['cat']== "noisy")), 'z'] = 1 # LSZU: en subtiele fout, dit Mstep['predicted_subtiele_fout] ==1, question: what is de else? 0? 
            if self.include_odf == True:
                self.Mstep['pred_subtiele_fout'] = np.where(((self.Mstep['tau_new_adj'] >= self.Mstep['afkap_z']) & (self.Mstep['cat']== "noisy") & (self.Mstep['pred_overduidelijke_fout'] != 1)), 1, 0) # unsure if correct
        
                # LSZU: pred overduidelijke fout in z
            
                self.Mstep['z'] = np.where(self.Mstep['pred_overduidelijke_fout']==1, 1, self.Mstep['z'])
            
                self.Mstep.loc[((self.Mstep['pred_subtiele_fout']==0) & (self.Mstep['pred_overduidelijke_fout'] ==0) & (self.Mstep['cat']== "noisy")) , 'z'] = 0 
                # als self.mstep['predcited_overduidelijke_fout] ==1, dan self.Mstep[z]==1 , anders afblijven done

                self.Mstep['pred_geenfout'] = np.where(self.Mstep['z'] == 0, 1, 0)
            else:
                self.Mstep.loc[(((self.Mstep['tau_new_adj'] < self.Mstep['afkap_z']) | (self.Mstep['afkap_z']==0)) & (self.Mstep['cat']== "noisy")) , 'z'] = 0 
                self.Mstep['pred_geenfout'] = np.where(self.Mstep['z'] == 0, 1, 0)
                self.Mstep['pred_fout'] = np.where(self.Mstep['z'] == 0, 0, 1)


            ### TIJDELIJKE PRINT ###
            if it>1:
                self.Mstep['pi_est_old'] = pi_est_old
            else:
                self.Mstep['pi_est_old'] = None
                
            Mstep_performance = self.Mstep.loc[self.index_test]
            pi_stats = Mstep_performance.groupby(self.performance_col)[['manipulated', 'pi_est_old', 'pi_est', 'afkap_z', 'z']].mean()
            lq_log(f"pi_stats = {pi_stats}")	        
            #### END CORRECT PARAMETERS ####
            pi_est_old = self.Mstep['pi_est'].copy()
            
            ### LLH ###
            llhdat = pd.DataFrame({
                "probML":probML, "contr_psi": contr_psi, "cat":self.Mstep['cat'],
                "pred_z":self.Mstep['z'], "pred_pi":self.Mstep['pi_est']
                })
        
            LLH_NOISY, LLH_LAB_PI, LLH_LAB_ML, LLH_LAB, LLH_TOT = llh_function(llhdat)

            ### MEASURES ###
            if compute_results:
                Mstep_performance = self.Mstep.loc[self.index_test]
          
                TPR, TNR, ACC, B_ACC = performance(Mstep_performance['manipulated'], Mstep_performance['z']) #werkelijk vs voorspeld

                if self.sample != None:
                    z_samp = self.Mstep.loc[samp, 'tau_new'].copy()
                    z_samp_afkap = self.Mstep.loc[samp, 'X_cat'].map(X_cat_afkap)
                    z_samp[(z_samp >= z_samp_afkap)] = 1
                    z_samp[(z_samp < z_samp_afkap)] = 0
                    TPR_sample, TNR_sample, _, _ = performance(self.Mstep.loc[samp, 'manipulated'], z_samp)
                else:
                    TPR_sample, TNR_sample = None, None

                R2 = eR2(Mstep_performance['manipulated'], Mstep_performance['tau_new_adj']) # same as tau_new since for the noisy set not adjustments are made.

                eR2_y = eR2_y_gold(probML[train_index], self.n_label)
                relEnt_y = rel_ent_y_noisy(y_hat_probM[noisy_index], self.n_label)
                perc_switched = sum(self.Mstep['z']!= self.Mstep['z_old'])/len(self.Mstep['z'])

                summary = pd.concat([summary, pd.DataFrame.from_records([{
                    "Accuracy": ACC, "Balanced_accuracy": B_ACC,
                    "TPR": TPR, "TNR": TNR, 'TPR_sample': TPR_sample, 'TNR_sample': TNR_sample,
                     "R2": R2, 'R2_y_gold': eR2_y, 'relEnt_y_noisy': relEnt_y, 
                    "LLH_NOISY": LLH_NOISY, "LLH_LAB_PI": LLH_LAB_PI, "LLH_LAB_ML": LLH_LAB_ML,
                    "LLH_LAB": LLH_LAB, "LLH": LLH_TOT, "switches": perc_switched,
                    "a": a*2, "Q_ml": Q_ml, "Q_ml_old": Q_ml_old, "Q_pi": Q_pi, "Q_pi_old": Q_pi_old, "D_r": d_r}])
                    ], ignore_index=True)

                if it > 1:
                    diff = summary.iloc[-1]['LLH'] - summary.iloc[-2]['LLH']
                else:
                    diff = 0.0
                
                # Log iteration! 
                lq_log(f'it: {it}, a = {a * 2}, diff = {diff}', to_console=False)

                if (it > self.miniterations) and (diff < self.tol): 
                    break
                
            #### UPDATE PARAMETERS #### (old location of update parameters)
            weight_tau_old = self.Mstep['weight_tau_new_adj'].copy()
            self.Mstep['z_old'] = self.Mstep['z'].copy()
            y_pi = self.Mstep['tau_new_adj'].copy()

            # excl_fixed moet hier geupdated worden
            if self.include_odf == True:
                excl_fixed = ((self.Mstep['cat'] != 'gold') & (self.Mstep['overduidelijke_fout'] == 0))
            else:
                excl_fixed = ((self.Mstep['cat'] != 'gold'))
            y_pi_fit = y_pi[excl_fixed].copy()
            X_pi_fit = X_pi[excl_fixed].copy()

        ## SAVE STATS/RESULTS PER UNIT ## 
        if compute_results:
            self.Mstep['TF'] = self.Mstep.apply(conditions_TF, axis=1) 
            probs_list = y_hat_probM.tolist() 
            self.Mstep['entropy_R2'] = [R2_entropy(x) for x in probs_list]
            self.Mstep['Max'] = np.max(probs_list, axis = 1)
            self.Mstep['SD'] = [stdev(x) for x in probs_list]
            self.Mstep['Range'] = np.ptp(probs_list, axis = 1)

            # Separate dataframes depending on settings used 
            if self.include_odf == True:
                results_df = pd.DataFrame({
                    "CbsPersoonIdentificatie": self.Mstep['CbsPersoonIdentificatie'], "geregistreerde_SBI": self.Mstep['geregistreerde_sbi_5_ingedikt'],
                    "correcte_SBI": self.Mstep['correcte_sbi_5_ingedikt'], "gecontroleerd": self.Mstep['gecontroleerd'], "testset": self.Mstep['testset'], 
                    "manipulated": self.Mstep['manipulated'],
                    "iteration": it, "cat": self.Mstep['cat'], "pred_tau": self.Mstep['tau_new_adj'],
                    "weight_tau": self.Mstep['weight_tau_new_adj'], "pred_z": self.Mstep["z"], "pred_pi" : self.Mstep['pi_est'],
                    "classification" : self.Mstep["TF"], "probML" : probML,
                    "contr_psi": contr_psi, 'c_weight': weights, 'entropy_R2': self.Mstep['entropy_R2'], 'Max': self.Mstep['Max'],
                    'SD': self.Mstep['SD'], 'Range': self.Mstep['Range'],
                    'pred_overduidelijke_fout' : self.Mstep['pred_overduidelijke_fout'], 'echte_overduidelijke_fout' : self.Mstep['overduidelijke_fout'],
                    'pred_subtiele_fout' : self.Mstep['pred_subtiele_fout'], 'echte_subtiele_fout' : self.Mstep['subtiele_fout'], 
                    'pred_geenfout' : self.Mstep['pred_geenfout'], 'echte_geenfout' : self.Mstep['echte_geenfout'],
                    'echte_onbekend' : self.Mstep['echte_onbekend'] 
                    }) 
                
                columns = ['pred_overduidelijke_fout', 'pred_subtiele_fout', 'pred_geenfout','echte_overduidelijke_fout', 'echte_subtiele_fout', 'echte_geenfout']
            
                confusion_df = results_df[results_df.index.isin(self.index_test)][columns].copy()
                confusion_df['predicted'] = confusion_df[['pred_overduidelijke_fout', 'pred_subtiele_fout', 'pred_geenfout']].idxmax(axis=1)
                confusion_df['true'] = confusion_df[['echte_overduidelijke_fout', 'echte_subtiele_fout', 'echte_geenfout']].idxmax(axis=1)

                confusion_matrix = pd.crosstab(confusion_df['true'], confusion_df['predicted'], rownames=['True'], colnames = ['Predicted'])
     
            else:

                input_columns = self.Mstep.columns.tolist()
                results_df = self.Mstep[input_columns].copy()
                
                temp_df = pd.DataFrame({
                 "probML" : probML, "pred_tau": self.Mstep['tau_new_adj'],  "pred_pi" : self.Mstep['pi_est'], "weight_tau": self.Mstep['weight_tau_new_adj'],
                 "iteration": it, "cat": self.Mstep['cat'], "pred_z": self.Mstep["z"], "classification" : self.Mstep["TF"],
                 "probML_sumGh":probML_sumGh, "probML_sumOGh": probML_sumOGh, "contr_sumGh": contr_sumGh,
                 "contr_sumOGh": contr_sumOGh, "contr_psi": contr_psi, 'c_weight': weights,
                })

                results_df = pd.concat([results_df, temp_df], axis=1)

                columns = ['pred_geenfout', 'pred_fout', 'echte_geenfout', 'echte_fout']
                confusion_df = results_df[results_df.index.isin(self.index_test)][columns].copy()

                confusion_df['predicted'] = confusion_df[['pred_geenfout', 'pred_fout']].idxmax(axis=1)
                confusion_df['true'] = confusion_df[['echte_geenfout', 'echte_fout']].idxmax(axis=1)

                confusion_matrix = pd.crosstab(confusion_df['true'], confusion_df['predicted'], rownames=['True'], colnames = ['Predicted'])
                    
            Mstep_performance = self.Mstep.loc[self.index_test]
            pi_stats = Mstep_performance.groupby(self.performance_col)[['manipulated', 'pi_est', 'afkap_z', 'z']].mean()
            pi_stats.columns = ["Actual pi per PC", "Predicted pi PC", "Tau_afkapz", "z_avg"]

            # if psi, write out probability of change 
            if self.tau_choice == 'psi':
                psi_output = psi_table(y_hat_probM, y, 'observedLabel_enc')

            else: 
                psi_output = None

            return summary, results_df, pi_stats, psi_output, confusion_matrix
        else:
            return LLH_TOT

    ########################################################################################
