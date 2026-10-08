import pandas as pd
import numpy as np
from scipy.stats import entropy

########################################################################################

def tau_epsilon_init(y, GHTables, LabelEncoder=None):  # LSZU DPGS labelencoder als input meegegeven
#      """
#      determine the fixed transition probabilities
#      """
    Gh = pd.read_csv(GHTables['Gh'], sep=';')
    Hg = pd.read_csv(GHTables['Hg'], sep=';')
    OGh = pd.read_csv(GHTables['OGh'], sep=';')
    OHg = pd.read_csv(GHTables['OHg'], sep=';')

    if LabelEncoder==None:
        numberread = Gh["N_G_h"] 
        # create baselist: each row is observed NACE (h), rows is set corresponding G(h)
        baselist_Sbis = list() 

        for row in range(len(Gh)):
            list_Sbis = list(map(int, Gh.iloc[row, 3:3+numberread[row]]))
            baselist_Sbis.append(list_Sbis)

        sizeHg = Hg['N_H_g']
        y_list_Gh = [baselist_Sbis[i] for i in y]
        sizeHg_Gh = [list(sizeHg[i]) for i in y_list_Gh]
        

        # create baselist: each row is observed NACE (h), rows is set corresponding OG(h)
        baselist_Sbis = list() 

        for row in range(len(OGh)):
            list_Sbis = list(map(int, OGh.iloc[row, 3:3+numberread[row]]))
            baselist_Sbis.append(list_Sbis)

        sizeOHg = OHg['N_H_g']
        epsilon = np.mean(OHg["P_H_g"])/100 # avg proportion of transitions from g to h not included
        
        # list of OG(h) for the observed NACE codes
        y_list_OGh = [baselist_Sbis[i] for i in y]

        # obtain the size OHg for each g in the array y_list_OGh
        sizeOHg_OGh = [list(sizeOHg[i]) for i in y_list_OGh]  

        return y_list_Gh, y_list_OGh, sizeHg_Gh, sizeOHg_OGh, epsilon
    else:
        mapping = dict(zip(LabelEncoder.classes_, LabelEncoder.transform(LabelEncoder.classes_)))
        
        matrices_list = [Gh, Hg, OGh, OHg]
        for i in range(len(matrices_list)):
            print(matrices_list[i]['SBIcode'])
            matrices_list[i]['SBIcode'] = LabelEncoder.transform(matrices_list[i]['SBIcode'].astype('str'))
        
            print(matrices_list[i]['SBIcode'])
            for j in matrices_list[i].columns[3:]:
                # LSZU DPGS convert these columns to int except NA
                matrices_list[i][j] = matrices_list[i][j].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)
                matrices_list[i][j] = matrices_list[i][j].map(mapping)

            matrices_list[i] = matrices_list[i].sort_values(by="SBIcode").reset_index(drop=True)
            print(matrices_list[i])
            print(Gh)
        Gh = matrices_list[0]
        Hg = matrices_list[1]
        OGh = matrices_list[2]
        OHg = matrices_list[3]
        
        numberread = Gh["N_G_h"] # true NACE codes g belonging to h

        # create baselist: each row is observed NACE (h), rows is set corresponding G(h)
        baselist_Sbis = list() 

        for row in range(len(Gh)):
            list_Sbis = list(map(int, Gh.iloc[row, 3:3+numberread[row]]))
            baselist_Sbis.append(list_Sbis)

        sizeHg = Hg['N_H_g']
        
        # list of G(h) for the observed NACE codes
        y_list_Gh = [baselist_Sbis[i] for i in y]
        
        #  LSZU DPGS, veranderd door de veranderde sortering door de mapping. Index pakken werkt niet meer
        # y_list_Gh = [baselist_Sbis[Gh[Gh['SBIcode']==i].index[0]] for i in y]

        
        # obtain the size Hg for each g in the array y_list_Gh
        sizeHg_Gh = [list(sizeHg[i]) for i in y_list_Gh]

        # sizeHg_Gh = [list(sizeHg)[Gh[Gh['SBIcode']==i].index[0]] for i in y_list_Gh]
        
        numberread = OGh["N_G_h"] # true NACE codes g belonging to h

        # create baselist: each row is observed NACE (h), rows is set corresponding OG(h)
        baselist_Sbis = list() 

        for row in range(len(OGh)):
            list_Sbis = list(map(int, OGh.iloc[row, 3:3+numberread[row]]))
            baselist_Sbis.append(list_Sbis)

        sizeOHg = OHg['N_H_g']
        epsilon = np.mean(OHg["P_H_g"])/100 # avg proportion of transitions from g to h not included
        
        # list of OG(h) for the observed NACE codes
        y_list_OGh = [baselist_Sbis[i] for i in y]
        # # LSZU DPGS veranderd door de sortering
        # y_list_OGh = [baselist_Sbis[OGh[OGh['SBIcode']==i].index[0]] for i in y]


        # obtain the size OHg for each g in the array y_list_OGh
        sizeOHg_OGh = [list(sizeOHg[i]) for i in y_list_OGh]  
        # sizeOHg_OGh = [list(sizeOHg)[OGh[OGh['SBIcode']==i].index[0]] for i in y_list_OGh]  
        

        return y_list_Gh, y_list_OGh, sizeHg_Gh, sizeOHg_OGh, epsilon

########################################################################################

def tau_epsilon(y_hat_probM, y, y_list_Gh, sizeHg_Gh, y_list_OGh, sizeOHg_OGh, epsilon, pi_values):

    # probability of the registered NACE code (y)
    probML = np.array([sublist[select] for sublist, select in zip(y_hat_probM, y)])
    
    probML_Gh = [sublist[select] for sublist, select in zip(y_hat_probM, y_list_Gh)]
    probML_sumGh = [sum(i) for i in probML_Gh]

    # compute the contribution per h of the Gh
    contr_Gh = [probML_Gh[i] / sizeHg_Gh[i] for i in range(len(probML_Gh))] 
    contr_sumGh = [sum(i) for i in contr_Gh]

    # list of predicted probabilities for OGh (non-Gh, (O = other): the low contributors)
    probML_OGh = [sublist[select] for sublist, select in zip(y_hat_probM, y_list_OGh)]
    probML_sumOGh = [sum(i) for i in probML_OGh]

    # compute the contribution per h of the Gh
    contr_OGh = [probML_OGh[i] / sizeOHg_OGh[i] for i in range(len(probML_OGh))]
    contr_sumOGh = [sum(i) for i in contr_OGh]

    # type:pandas.series + type:list gives a pandas series.
    # oherwise a list comprehension would have been needed
    contr_psi = [(1 - epsilon)*a + epsilon*b for a,b in zip(contr_sumGh, contr_sumOGh)]

    d1m = pi_values * contr_psi
    d2 = (1 - pi_values) * probML

    tau_new = d1m / (d2 + d1m)
    
    # NSCN: ook hier return van psi, zodat dit matcht met tau_psi
    return tau_new, contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh

########################################################################################

def tau_psi(y_hat_probM, y, pi_values, label_new):
    
    # Dataframe met waargenomen SBI, h
    h = pd.DataFrame(y)
    h = h.reset_index(drop = True)
    
    # Dataframe met voorspelde SBI, g
    g = pd.DataFrame(y_hat_probM)
    
    # Samenvoegen van h en g
    hg = h.merge(g, left_index = True, right_index = True)
    n_cat = hg[label_new].nunique()
    
    # Nu M_gh maken, waarbij alle eenheden worden opgeteld per waargenomen SBI.
    M_gh = hg.groupby([label_new]).sum()

    # Dan de diagonaal op nul zetten, omdat we niet kijken naar h=g
    np.fill_diagonal(M_gh.values, 0)
    
    # Nu de som nemen over alle g's, de voorspelde SBI's
    col_sum = M_gh.sum(axis=0, skipna=True)
    
    # Alle kolomtotalen van g moeten optellen tot 1, dus alle aantallen delen door de kolomsom.
    psi_list = [M_gh.iloc[:,x]/col_sum[x] for x in range(0,n_cat)]
    #Dataframe komt er verkeerd om uit, dus tranpose nemen om het weer kloppend te maken (misschien moet dit even anders nog?)
    psi = pd.DataFrame(psi_list).T
    
    # HG - nieuwe colnames
    column_indices = [*range(1,n_cat+1)]
    new_names = [str(x) + "_prob" for x in range(0,n_cat)]
    old_names = hg.columns[column_indices]
    hg.rename(columns=dict(zip(old_names, new_names)), inplace = True)

    # PSI - nieuwe colnames
    new_names_psi = [str(x) + "_psi" for x in range(0,n_cat)]
    psi.columns = new_names_psi

    # Voeg dan op basis van de waargenomen SBI, h, de juiste verwisselingskansen toe per eenheid 
    # _x=probML
    # _y=verwisselingskans
    psi_gh = hg.merge(psi, on = label_new, how = 'left')

    # Vermenigvuldigen probML x psi per SBI binnen de eenheid
    # G bevat alle kansen, psi_gh alle verwisselingskansen voor alle eenheden 
    psis = psi_gh.loc[:,'0_psi':f'{n_cat-1}_psi']
    psi_v = pd.DataFrame(g.values*psis.values)
     
    # Som nemen over de vermenigvuldigde waarden, contr_psi en meteen toevoegen aan psi_gh
    psi_gh['contr_psi'] = psi_v.sum(axis=1)
    
    # contr_psi vermenigvuldigen met de pi_values
    # in welk format komen de pi_values eruit, in een array? 
    # nu voor de zekerheid array van maken
    psi_gh['pi_values'] = np.array(pi_values)
    psi_gh['d1'] = psi_gh['contr_psi']*psi_gh['pi_values']
    
    # d2 = (1-pi_values) * probML 
    psi_gh['probML'] = np.array([sublist[select] for sublist, select in zip(y_hat_probM, y)])

    psi_gh['d2']= (1-psi_gh['pi_values'])*psi_gh['probML']
    
    psi_gh['tau_new'] = psi_gh['d1'] / (psi_gh['d2'] + psi_gh['d1'])
    
    tau_new = psi_gh['tau_new']
    contr_psi = psi_gh['contr_psi']
    probML = psi_gh['probML']
    probML_sumGh = [None] * len(contr_psi)
    probML_sumOGh = [None] * len(contr_psi)
    contr_sumGh = [None] * len(contr_psi)
    contr_sumOGh = [None] * len(contr_psi)
    
    # Voor nu ook even return doen van de onderdelen uit de oude tau, zodat we beide opties gemakkelijk kunnen afwisselen
    return tau_new, contr_psi, probML, probML_sumGh, probML_sumOGh, contr_sumGh, contr_sumOGh

########################################################################################

def psi_table(y_hat_probM, y, label_new):
    
    # Dataframe met waargenomen SBI, h
    h = pd.DataFrame(y)
    h = h.reset_index(drop = True)
    
    # Dataframe met voorspelde SBI, g
    g = pd.DataFrame(y_hat_probM)
    
    # Samenvoegen van h en g
    hg = h.merge(g, left_index = True, right_index = True)
    n_cat = hg[label_new].nunique()
    
    # Nu M_gh maken, waarbij alle eenheden worden opgeteld per waargenomen SBI.
    M_gh = hg.groupby([label_new]).sum()

    # Dan de diagonaal op nul zetten, omdat we niet kijken naar h=g
    np.fill_diagonal(M_gh.values, 0)
    
    # Nu de som nemen over alle g's, de voorspelde SBI's
    col_sum = M_gh.sum(axis=0, skipna=True)
    
    # Alle kolomtotalen van g moeten optellen tot 1, dus alle aantallen delen door de kolomsom.
    psi_list = [M_gh.iloc[:,x]/col_sum[x] for x in range(0,n_cat)]
    #Dataframe komt er verkeerd om uit, dus tranpose nemen om het weer kloppend te maken (misschien moet dit even anders nog?)
    psi = pd.DataFrame(psi_list).T
    
    # HG - nieuwe colnames
    column_indices = [*range(1,n_cat+1)]
    new_names = [str(x) + "_prob" for x in range(0,n_cat)]
    old_names = hg.columns[column_indices]
    hg.rename(columns=dict(zip(old_names, new_names)), inplace = True)

    # PSI - nieuwe colnames
    new_names_psi = [str(x) + "_psi" for x in range(0,n_cat)]
    psi.columns = new_names_psi

    return psi

def R2_entropy(probabilities):
    
    ent = entropy(probabilities, base=2)
    
    max_ent = np.log2(len(probabilities))
    
    r2_ent = 1 - (ent/max_ent)
    
    return r2_ent

########################################################################################