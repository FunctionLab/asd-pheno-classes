import statistics
from math import sqrt
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle as rick
import matplotlib.pyplot as plt


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.05):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))
    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')
    return mean, confidence_interval


def remove_indels(variants):
    variants['ref'] = variants['ref'].astype('str')
    variants['alt'] = variants['alt'].astype('str')
    mask = (variants['ref'].str.len() == 1) & (variants['alt'].str.len() == 1)
    variants = variants.loc[mask]
    return variants


def load_and_label_DNVs():
    pro_vars = pd.read_csv('/mnt/home/nsauerwald/ceph/SPARK/hareResults/out/HAT_DNVcalldataframe_positionfiles.txt',
                            sep='\t', header=0, index_col=0,
                            names=['spid', 'chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    sib_vars = pd.read_csv('/mnt/home/nsauerwald/ceph/SPARK/hareSiblings/out/HareSibling_DNVcalldataframe_positionfiles.txt', 
                            sep='\t', header=0, index_col=0,
                            names=['spid', 'chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    
    ### QC
    sib_vars = sib_vars[sib_vars['spid'] != 'SP0255010']  # exclude outlier sample
    # exclude samples with dnv count os >=3SD above the mean
    counts_pro = pro_vars.groupby('spid').count().to_dict()['chr']
    counts_sib = sib_vars.groupby('spid').count().to_dict()['chr']
    pro_vars['count'] = pro_vars['spid'].map(counts_pro) 
    sib_vars['count'] = sib_vars['spid'].map(counts_sib)  
    total_count_list = list(pro_vars['count']) + list(sib_vars['count'])
    cutoff = np.mean(total_count_list) + 3*np.std(total_count_list) 
    pro_vars = pro_vars[pro_vars['count'] <= cutoff]  
    sib_vars = sib_vars[sib_vars['count'] <= cutoff] 

    num_probands = len(pro_vars['spid'].unique())
    num_siblings = len(sib_vars['spid'].unique())

    var_to_spid = defaultdict(list)
    var_to_asd = defaultdict(list)

    for i in range(len(pro_vars)):
        spid, _, _, id, _, _, _, _, _, _ = pro_vars.iloc[i, :]
        var_to_spid[id].append(spid)
        var_to_asd[id].append(1)

    for i in range(len(sib_vars)):
        spid, _, _, id, _, _, _, _, _, _ = sib_vars.iloc[i, :]
        var_to_spid[id].append(spid)
        var_to_asd[id].append(0)

    # remove non-singleton variants
    count_sibling = 0
    count_proband = 0
    variant_to_label = dict()
    count = 0
    for var, asd_list in var_to_asd.items():
        if len(asd_list) > 1: # non-singleton: remove
            count += 1
            continue
        assert len(asd_list) == 1 
        if asd_list[0] == 0:  # sibling variant
            variant_to_label[var] = 0
            count_sibling += 1
        else:  # proband variant
            variant_to_label[var] = 1
            count_proband += 1

    with open(f'variant_labels/SPARK_de_novo_variant_to_label_singleton_3sd.pkl', 'wb') as handle:
        rick.dump(variant_to_label, handle)

    return variant_to_label


if __name__ == "__main__":
    
    variant_to_label = load_and_label_DNVs()