import sys
from os import listdir
from os.path import isfile, join
import statistics
from math import sqrt
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle as rick
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from pyliftover import LiftOver
from class_genetics import remove_indels

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
    '''remove variants from file where ref or alt are indels (>1bp).'''
    variants['ref'] = variants['ref'].astype('str')
    variants['alt'] = variants['alt'].astype('str')
    mask = (variants['ref'].str.len() == 1) & (variants['alt'].str.len() == 1)
    variants = variants.loc[mask]
    return variants

def load_and_label_DNVs():
    # load proband and sibling de novo variants into one dictionary mapping variant ID to list of SPIDs
    pro_vars = pd.read_csv('/mnt/home/nsauerwald/ceph/SPARK/hareResults/out/HAT_DNVcalldataframe_positionfiles.txt',
                           sep='\t', header=0, index_col=0,
                           names=['spid', 'chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    sib_vars = pd.read_csv(
        '/mnt/home/nsauerwald/ceph/SPARK/hareSiblings/out/HareSibling_DNVcalldataframe_positionfiles.txt', sep='\t',
        header=0, index_col=0,
        names=['spid', 'chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    
    ### QC
    # EXCLUDE SAMPLES WITH HIGH DNV COUNT
    sib_vars = sib_vars[sib_vars['spid'] != 'SP0255010']  # exclude outlier sample
    # exclude samples with >=3SD DNVs
    counts_pro = pro_vars.groupby('spid').count().to_dict()['chr']
    counts_sib = sib_vars.groupby('spid').count().to_dict()['chr']
    pro_vars['count'] = pro_vars['spid'].map(counts_pro)  # add count column
    sib_vars['count'] = sib_vars['spid'].map(counts_sib)  # add count column
    total_count_list = list(pro_vars['count']) + list(sib_vars['count'])
    cutoff = np.mean(total_count_list) + 3*np.std(total_count_list) # cutoff is 3 SD above the average mutation count
    pro_vars = pro_vars[pro_vars['count'] <= cutoff]  # filter - remove samples with >= DNVs
    sib_vars = sib_vars[sib_vars['count'] <= cutoff]  # filter - remove samples with >= DNVs
    print('mean count for probands')
    print(np.mean(pro_vars['count']))
    print('mean count for siblings')
    print(np.mean(sib_vars['count']))

    # get number of probands and siblings
    print('number of probands:')
    print(len(pro_vars['spid'].unique()))
    num_probands = len(pro_vars['spid'].unique())
    print('number of siblings:')
    print(len(sib_vars['spid'].unique()))
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

    # REMOVE NON-SINGLETONS
    count_sibling = 0
    count_proband = 0
    variant_to_label = dict()
    count = 0
    for var, asd_list in var_to_asd.items():
        if len(asd_list) > 1: # means that multiple people have variant, so remove non-singletons
            count += 1
            continue
        assert len(asd_list) == 1 # assert singleton before continuing
        if asd_list[0] == 0:  # sibling variant, mark with 0
            variant_to_label[var] = 0
            count_sibling += 1
        else:  # ASD-only variant, mark with 1
            variant_to_label[var] = 1
            count_proband += 1

    with open(f'variant_labels/SPARK_de_novo_variant_to_label_singleton_3sd.pkl', 'wb') as handle:
        rick.dump(variant_to_label, handle)

    return variant_to_label

if __name__ == "__main__":
    
    variant_to_label = load_and_label_DNVs()
    
    labels_df = pd.DataFrame(
        [(k, val) for k, val in variant_to_label.items()],
        columns=['id', 'label']
    )

    # expand 'id' 
    labels_df[['chr', 'pos', 'ref', 'alt']] = labels_df['id'].str.split('_', expand=True)
    labels_df = labels_df.drop(['label'], axis=1)
    labels_df = remove_indels(labels_df)

    # merge with seqweaver predictions to get labeled dataframe with DIS
    # First for ASD genes only
    dis_df = pd.read_csv(
        '/mnt/home/alitman/ceph/seqweaver_spark_ASD_genes_output/merged_spark_seqweaver_only_DIS_ASD_genes.tsv',
        sep='\t', index_col=None)  # merged chromosome seqweaver predictions ASD genes only DIS
    dis_df = dis_df.rename({'dis':'disease_impact_score'}, axis='columns') # rename for consistency
    merged_label_dis = pd.merge(dis_df[['id', 'disease_impact_score']], labels_df, on='id')  # merge labels and DIS on column 'id'
    merged_label_dis.to_csv('/mnt/home/alitman/ceph/DNVs_processed/DNVs_ASD_genes_labeled_singleton_qc3sd_DIS.tsv', sep='\t', index=False)  # save to file
    
    # Now for all genes
    dis_df_all_genes = pd.read_csv(
        '/mnt/home/alitman/ceph/seqweaver_spark_all_genes_output/merged_seqweaver_all_genes_onlyDIS.tsv',
        sep='\t', index_col=None) # merged chromosome seqweaver predictions all genes only DIS
    dis_df_all_genes = dis_df_all_genes[dis_df_all_genes['pos'] != 'pos']
    dis_df_all_genes['disease_impact_score'] = pd.to_numeric(dis_df_all_genes['disease_impact_score'])
    merged_label_dis_all_genes = pd.merge(dis_df_all_genes[['id', 'disease_impact_score']], labels_df, on='id')  # merge labels and DIS on column 'id'
    merged_label_dis_all_genes.to_csv('/mnt/home/alitman/ceph/DNVs_processed/DNVs_all_genes_labeled_singleton_qc3sd_DIS.tsv', sep='\t', index=False)  # save to file