import pandas as pd
import numpy as np
import pickle
import toml
import sys, getopt
import datetime
import logging
import re
from utils.mainV3 import LabelQuality
from utils.utils import get_errorper, select_model, lq_log
from utils.sample import create_sample
from utils.create_input_files import add_selection_columns

import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4"

########################################################################################
def start(config, X, yplus, clf_name, run, setseed, pathR):

    alpha = config.get('alpha', 1)
    tau_choice = config.get('tau_choice', 'psi')
    errortypes = config['errortypes']
    errorper = config['errorper']
    observedLabel = config['observedLabel']
    scenarios = config['scenarios']
    threshold = config.get('threshold', 5)
    wmodel = config.get('wmodel')
    prior_ = config.get('prior')
    incl_odf = config.get('include_odf')
    r2_grens = config.get('r2_grenswaarde')
    it_odf = config.get('it_overduidelijkefout')
    X1 = config.get('bg_col', 'X1')
    sc_lim = config.get('sc_lim', 'sc_lim')

    for sce in scenarios:
        if sce == 'sc1':
            config['bg_col'] = None
            config['sample_col'] = None
            config['n_sample'] = None
        elif sce == 'sc2':
            config['bg_col'] = X1
            config['sample_col'] = None
            config['n_sample'] = None
        elif sce == 'sc3':
            config['bg_col'] = sc_lim
            config['sample_col'] = f'sample_{sc_lim}'
        elif sce == 'sc4':
            config['bg_col'] = X1
            config['sample_col'] = f'sample_{X1}'
        
        for gold_col in config['gold_cols']:
            config['gold_col'] = gold_col
            for err in errortypes:
                config['errortype'] = err
                errorper = get_errorper(config, err)
                for per in errorper:
                    config['percentage'] = per 
                    if tau_choice == 'eps':
                        errortype = err if per == '' else ''
                        pattern = f"{'' if errortype=='' else errortype + '_'}{threshold}"
                        config['GHTables'] = {
                            'Gh': f"data/GhTable_simulation_{pattern}.csv",
                            'Hg': f"data/HgTable_simulation_{pattern}.csv",
                            'OGh': f"data/OGhTable_simulation_{pattern}.csv",
                            'OHg': f"data/OHgTable_simulation_{pattern}.csv"
                        }


                    lq_log(f'scenario: {sce}, gold: {gold_col}, errortype: {err}, percentage: {per}')
                    config['observedLabel_col'] = f'{observedLabel}{per}{err}'
                    # print(f'TEST: label_{per}{err}')
                    lq = LabelQuality(X, yplus, config)
                    ## TRAIN MODEL FOR INITIALISATION
                    # all initialisation strategies are describe in the file "sbikwaliteit/Starts_EM.docx" 
                    # if sce == "sc1":
                    tau_init_list_sc1 = pd.DataFrame()
                       
                 
                    if "sc1" not in scenarios:
                        # tau_init_list_sc1 = pd.DataFrame()
                        run_nr = [int(num) for num in re.findall(r"\d+", run)][0]
                        per_str = '' if per == '' else f'{str(100 + per)[1:]}perc_'
                        filename_sc1 = f'{per_str}{err}_{gold_col.capitalize()}_sc1_{tau_choice}_{alpha}_{clf_name}_{run}_{wmodel}_prior{prior_}_{config["log"]}_odf{incl_odf}_{r2_grens}_{it_odf}.csv'
                        # filename_sc1 = '_Gold100_sc1_psi_1_NB_Seed1_w4b_priorFalse_dry_beans_odfFalse_None_None.csv' # LSZU DPGS veranderen naar automatisch inlezen met datum
                        results_sc1 = pd.read_csv(f'{pathR}/results_df/{filename_sc1}', sep=',')
                        tau_init_list_sc1 = results_sc1[['pred_tau']].copy()
                    
                    ############################################################
                    # Obtain results
                    ############################################################
                    tau_init_files = config.get('tau_init', {})
                    if len(tau_init_files) == 0:
                        tau_init_list = lq.initial_tau_multiple(setseed)
                        tau_init_list = pd.concat([tau_init_list, tau_init_list_sc1], axis=1, sort=False)
                        tau_init_list.columns = range(tau_init_list.shape[1])
                        
                        #tau_init_list.to_csv('data/tau_init_list_test.csv', sep=';', index=False)
                    else:
                        tau_init_list = pd.DataFrame()
                        for i, key in enumerate(tau_init_files.keys()):
                            results_df = pd.read_csv(tau_init_files[key], sep=',')
                            tau_init = results_df[['pred_tau']]
                            tau_init.columns = [key]
                            tau_init_list = pd.concat([tau_init_list, tau_init], axis=1)

                    list_bestLLH = []

                    lq_log('processing initial tau_init')
                    
                    for l in tau_init_list.columns:
                        lq_log(f'\ttau_init case: {l}', to_console=False)
                        tau_init = tau_init_list[l]
                        # for the audit sample & gold set the initial tau to the true value.
                        bestLLHs = lq.EM(tau_init=tau_init, compute_results=False, sce = sce)
                        list_bestLLH.append(bestLLHs)

                    n_tau_init = min(len(tau_init_list.columns), 3)
                    LLHs = pd.DataFrame({"LLH":list_bestLLH, "it":tau_init_list.columns}) 
                    # save LLH's and which startoption they belong to
                    bestit = LLHs.sort_values(by="LLH", axis=0, ascending=False)['it'][:n_tau_init] 
                    #select three best
                    tau_init_df = tau_init_list[bestit]
                   

                    lq_log('processing best tau_init (max 3)')

                    summary, results_df, pi_stats, df, psi_output, confusion = [None] * n_tau_init, [None] * n_tau_init, [None] * n_tau_init, [None] * n_tau_init, [None] * n_tau_init, [None] * n_tau_init
                    for i in range(n_tau_init):
                        lq_log(f'\tbest tau_init case: {i}', to_console=False)
                        print(tau_init_df.iloc[:,i])
                        tau_init = tau_init_df.iloc[:,i]
                        # summary[i], results_df[i], pi_stats[i], psi_output[i], confusion[i] =lq.EM(tau_init=tau_init, compute_results=True, sce = sce)
                        list_temp =lq.EM(tau_init=tau_init, compute_results=True, sce = sce)
                        if len(list_temp)==5:
                            summary[i], results_df[i], pi_stats[i], psi_output[i], confusion[i] = list_temp
                        else:
                            summary[i], results_df[i], pi_stats[i], psi_output[i] = list_temp

                        # Extra confusion matrix output!!
                        df[i] = float(summary[i].iloc[-1,:][["LLH"]].iloc[0])

                    i_max = np.argmax(df)

                    if sce == 'sc1':
                        tau_init_list_sc1 = results_df[i_max][['pred_tau']].copy()
                    
                    if len(tau_init_files) > 0:
                        tau_init_file = f'_{tau_init_df.columns[i_max]}'
                    else:
                        tau_init_file = ''

                    per_str = '' if per == '' else f'{str(100 + per)[1:]}perc_'
                    file_name = f'{per_str}{err}_{gold_col.capitalize()}_{sce}_{tau_choice}_{alpha}_{clf_name}_{run}{tau_init_file}_{wmodel}_prior{prior_}_{config["log"]}_odf{incl_odf}_{r2_grens}_{it_odf}.csv'
                    summary[i_max].to_csv(f'{pathR}/Summary/{file_name}', index=False)
                    results_df[i_max].to_csv(f'{pathR}/results_df/{file_name}', index=False)
                    pi_stats[i_max].to_csv(f'{pathR}/pi_stats/{file_name}', index=False)
                    if len(list_temp) == 5:
                        confusion[i_max].to_csv(f'{pathR}/confusion/{file_name}', index=False)

                    if tau_choice == 'psi':
                        psi_output[i_max].to_csv(f'{pathR}/psi/{file_name}', index=True)

########################################################################################

if __name__ == '__main__':
    run = None

    current_path = ""
    pathR = current_path + 'results'
    pathData = current_path + 'data'
    
    models = toml.load('config/models.toml')
    pathLog = current_path + "log/"

    for j in [1]:
      
        # # dry beans
        opts = [('-r', '1'), ('-c', current_path + 'config/dry_beans.toml'), ('-m', 'NB'), ('-l', 'dry_beans')] 
        args = []

        if len(opts) != 4:
            print("usage: python scenarios.py -r <number> -c <filename> -m <classifier> -l <logname>")
            sys.exit(2)

        for opt, arg in opts:
            if opt in ["-r", "--run"]:
                run_no = int(arg)
            elif opt in ["-c", "--config"]:
                try:
                    config = toml.load(arg)
                except:
                    print(f'file {arg} not found')
                    sys.exit(2)
            elif opt in ["-m", "--model"]:
                if arg in models.keys():
                    base_clf, calibrate, clf_params = select_model(arg, models)
                    clf_name = arg
                    if base_clf == None:
                        print(f'classifier must be one of {models.keys()}')
                        sys.exit(2)                
                else:
                    print(f'classifier must be one of {models.keys()}')
                    sys.exit(2)
            elif opt in ["-l", "log="]:
                log = arg
            else:
                print("usage: scenarios.py -r <number> -c <filename> -m <classifier> -l <logname>")
                sys.exit(2)
        
        if run_no in range(1, len(config['seeds']) + 1):
            run = f"Seed{run_no}"
        else:
            print(f"number must be an integer between 1 and {len(config['seeds']) + 1}")
            sys.exit(2)


        # uncomment for drybeans
        with open(f'{current_path}/dry_beans/features.pickle', 'rb') as f:
            X = pickle.load(f)
        yplus = pd.read_csv(f'{current_path}/dry_beans/labels_1_test.csv',  sep=';') # Separator in sectie R moet naar ; (is nu ,), in simulaties al ;
       

        logging.basicConfig(filename=f'{pathLog}{log}.log', level=logging.DEBUG, filemode='w')

        config['base_clf'] = base_clf
        config['calibrate'] = calibrate
        config['log'] = log

        for prior_i in [False, True]: 
            for r2_grenswaarde_j in [0.1, 0.2, 0.3, 0.4, 0.5]:
                config['prior'] = prior_i
                config['r2_grenswaarde'] = r2_grenswaarde_j

                seed = config['seeds'][run_no - 1]

                lq_log(f"run={run}, seed={seed}")
                lq_log(config)
                lq_log(f'Classifier: {clf_name}')
                lq_log(f'Parameters: {clf_params}')
                lq_log(f'Calibrate: {calibrate}')

                # for i in ["eps", "psi"]:
                #     config['prior'] = False
                #     config["tau_choice"] = i
                start_time = datetime.datetime.now().replace(microsecond=0)
                print(config)
                start(config, X, yplus, clf_name, run, seed, pathR)
                end_time = datetime.datetime.now().replace(microsecond=0)

                lq_log(f'start: {start_time}')
                lq_log(f'stop: {end_time}')
                lq_log(f'time elapsed: {end_time - start_time}')
