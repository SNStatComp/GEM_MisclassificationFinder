import pandas as pd

def create_sample(yplus, index_noisy, config):
    # mapping_file = config.get('mapping_file', None)

    # if mapping_file != None:
    #     mapping = pd.read_csv(mapping_file, sep=';')
    #     bg_cols = mapping.columns
    #     yplus = yplus.merge(mapping, on=bg_cols[0], how='left')
    # else:

    bg_cols = [config['bg_col']]
    sample = yplus.loc[index_noisy]

    for bg_col in bg_cols:
        yplus[f'sample_{bg_col}'] = 0
        sample = sample.groupby(bg_col, group_keys=False).apply(lambda group_df: group_df.sample(config['n_sample']))
        yplus.loc[sample.index, f'sample_{bg_col}'] = 1
    
    return yplus