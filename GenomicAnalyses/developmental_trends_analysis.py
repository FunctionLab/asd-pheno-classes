import pandas as pd
import numpy as np
import pickle as rick
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from utils import load_dnvs, get_trend_celltype_gene_sets


def make_gene_trend_figure(fdr=0.05):
    # run function or load precomputed data
    # gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets() 
    
    with open('data/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('data/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('data/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('data/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # select LoF consequences
    consequences = [
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 
        'splice_donor_variant', 'start_lost', 'stop_lost', 
        'transcript_ablation'
        ]
   
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs 
    # with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of participants in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # compute enrichments for each gene set
    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['LoF'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['LoF']
    
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro[
            'mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['gene_set&consequence'].sum().tolist() 
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # get p-values comparing each class to sibs using a t-test
        sibs_rest_of_sample = class0 + class1 + class2 + class3
        class0_pval = ttest_ind(
            class0, sibs, alternative='greater')[1]
        class1_pval = ttest_ind(
            class1, sibs, alternative='greater')[1]
        class2_pval = ttest_ind(
            class2, sibs, alternative='greater')[1]
        class3_pval = ttest_ind(
            class3, sibs, alternative='greater')[1]
        sibs_pval = ttest_ind(
            sibs, sibs_rest_of_sample, alternative='greater')[1]
        all_pros_pval = ttest_ind(
            all_pros_data, sibs, alternative='greater')[1]

        uncorrected_pvals = [-np.log10(class0_pval), -np.log10(class1_pval),
                                -np.log10(class2_pval), -np.log10(class3_pval),
                                -np.log10(sibs_pval), -np.log10(all_pros_pval)]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, 
                                   class3_pval, sibs_pval, all_pros_pval], 
                                   method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + 
                          np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + 
                                                          num_class2 + num_class3 + 
                                                          num_sibs)
        background = np.sum(sibs)/num_sibs # sibling background
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'unadjusted': uncorrected_pvals[0],
                                  'Fold Enrichment': class0_fe, 'cluster': 0, 'vs.': 'siblings',
                                  'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'unadjusted': uncorrected_pvals[1], 
                                  'Fold Enrichment': class1_fe, 'cluster': 1, 'vs.': 'siblings',
                                  'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'unadjusted': uncorrected_pvals[2], 
                                  'Fold Enrichment': class2_fe, 'cluster': 2, 'vs.': 'siblings',
                                  'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'unadjusted': uncorrected_pvals[3], 
                                  'Fold Enrichment': class3_fe, 'cluster': 3, 'vs.': 'siblings',
                                  'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'unadjusted': uncorrected_pvals[5], 
                                    'Fold Enrichment': all_pros_fe, 'cluster': -1, 'vs.': 'siblings',
                                    'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'unadjusted': uncorrected_pvals[4], 
                                'Fold Enrichment': sibs_fe, 'cluster': -2, 'vs.': 'all other probands',
                                'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, 
                                       class0_df, class1_df, class2_df, class3_df], 
                                       axis=0)
    
    order = ['variable', 'trend', 'cluster', 'vs.', 'unadjusted', 'value', 'Fold Enrichment']
    validation_subset = validation_subset[order]

    validation_subset.to_csv('../supp_tables/Supp_Table_12_siblings.csv')

    sorted_val = validation_subset.groupby('variable').max().sort_values('value', ascending=False)
    rm_gene_sets = sorted_val[sorted_val['value'] < -np.log10(fdr)].index
    validation_subset = validation_subset[~validation_subset['variable'].isin(rm_gene_sets)]
    
    order_of_variables = ['Principal_excitatory_neuron_down',
                            'Inhibitory_interneuron_MGE_down',
                            'Inhibitory_interneuron_CGE_down',
                            'Glia_down',
                            'Principal_excitatory_neuron_trans_down',
                            'Inhibitory_interneuron_MGE_trans_down',
                            'Inhibitory_interneuron_CGE_trans_down',
                            'Glia_trans_down',
                            'Principal_excitatory_neuron_trans_up',
                            'Inhibitory_interneuron_MGE_trans_up',
                            'Inhibitory_interneuron_CGE_trans_up',
                            'Glia_trans_up',
                            'Principal_excitatory_neuron_up',
                            'Inhibitory_interneuron_MGE_up',
                            'Inhibitory_interneuron_CGE_up',
                            'Glia_up'
                            ]
    validation_subset['variable'] = pd.Categorical(validation_subset['variable'], 
                        categories=order_of_variables, ordered=True)
    validation_subset = validation_subset.sort_values(['variable', 'cluster'])
    
    colors = ['white', 'purple', '#27AAE1', '#39B54A', '#FBB040', '#EE2A7B']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map(
        {-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    validation_subset['color'] = validation_subset['cluster'].map(
        {-2: 'white', -1: 'mediumorchid', 0: '#FBB040', 1: '#EE2A7B', 
        2: '#39B54A', 3: '#27AAE1'})
    validation_subset['Cluster'] = validation_subset['cluster'].map(
        {-2: 'Siblings', -1: 'All Probands', 0: 'Moderate Challenges', 
         1: 'Broadly Impacted', 2: 'Social/Behavioral', 3: 'Mixed ASD with DD'})
    
    # plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(5.7,9))
    for _, row in validation_subset.iterrows():
        if row['value'] < -np.log10(fdr):
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*230, c='black', linewidth=2.5, 
                        edgecolors=row['color'], alpha=0.9)
        else:
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*270, c=row['color'])

    for i in range(2, 14, 3): # add legend sizes
        plt.scatter([], [], s=(i)*270, c='white', label=str(i))
    plt.legend(scatterpoints=1, labelspacing=2.6, title='Fold Enrichment', 
               title_fontsize=23, fontsize=18, loc='upper left', 
               bbox_to_anchor=(1, 1))
    plt.yticks(fontsize=17)
    plt.xticks(fontsize=17, rotation=35, ha='right')
    yticklabels = ['Glia', 'Inhibitory Interneuron MGE', 
                  'Inhibitory Interneuron MGE', 
                  'Principal Excitatory Neuron', 
                  'Principal Excitatory Neuron', 
                  'Glia', 'Inhibitory Interneuron CGE', 
                  'Inhibitory Interneuron MGE', 
                  'Principal Excitatory Neuron'][::-1]
    plt.yticks(ticks=range(len(yticklabels)), labels=yticklabels, fontsize=17)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.savefig(
        f'figures/WES_gene_trends_dnLoF_analysis.png', 
        bbox_inches='tight', 
        dpi=900
        )
    plt.close()


def get_stats_class2_baseline():
    # run function or load precomputed data
    #gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets() 
    
    with open('data/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('data/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('data/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('data/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # select LoF consequences
    consequences = [
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 
        'splice_donor_variant', 'start_lost', 'stop_lost', 
        'transcript_ablation'
        ]
   
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs 
    # with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of participants in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # compute enrichments for each gene set
    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['LoF'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['LoF']
    
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro[
            'mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['gene_set&consequence'].sum().tolist() 
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class3

        # get p-values comparing each class to sibs using a t-test
        class0_pval = min(ttest_ind(
            class0, class2, alternative='less')[1], ttest_ind(class0, class2, alternative='greater')[1])
        class1_pval = min(ttest_ind(
            class1, class2, alternative='less')[1], ttest_ind(class1, class2, alternative='greater')[1])
        class3_pval = min(ttest_ind(
            class3, class2, alternative='less')[1], ttest_ind(class3, class2, alternative='greater')[1])
        siblings_pval = min(ttest_ind(
            sibs, class2, alternative='less')[1], ttest_ind(sibs, class2, alternative='greater')[1])
        all_pros_pval = min(ttest_ind(
            all_pros_data, class2, alternative='less')[1], ttest_ind(all_pros_data, class2, alternative='greater')[1])
        
        symbols = ['^', '^', 'v', 'v']

        uncorrected_pvals = [-np.log10(class0_pval), -np.log10(class1_pval),
                                -np.log10(class3_pval), 
                                -np.log10(siblings_pval), -np.log10(all_pros_pval)]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, 
                                   class3_pval, siblings_pval, all_pros_pval], 
                                   method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class3_pval = -np.log10(corrected[2])
        sibs_pval = -np.log10(corrected[3])
        all_pros_pval = -np.log10(corrected[4])

        background = np.sum(class2)/num_class2        
        class0_fe = max((np.sum(class0)/num_class0)/background, background/(np.sum(class0)/num_class0))
        class1_fe = max((np.sum(class1)/num_class1)/background, background/(np.sum(class1)/num_class1))
        class3_fe = max((np.sum(class3)/num_class3)/background, background/(np.sum(class3)/num_class3))
        sibs_fe = max((np.sum(sibs)/num_sibs)/background, background/(np.sum(sibs)/num_sibs))
        all_pros_fe = max((np.sum(all_pros_data)/all_spids)/background, background/(np.sum(all_pros_data)/all_spids))

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'unadjusted': uncorrected_pvals[0],
                                  'Fold Enrichment': class0_fe, 'cluster': 0, 'vs.': 'Social/Behavioral',
                                  'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'unadjusted': uncorrected_pvals[1], 
                                  'Fold Enrichment': class1_fe, 'cluster': 1, 'vs.': 'Social/Behavioral',
                                  'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'unadjusted': uncorrected_pvals[2], 
                                  'Fold Enrichment': class3_fe, 'cluster': 3, 'vs.': 'Social/Behavioral', 
                                  'trend': trend}, index=[0])
        siblings_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'unadjusted': uncorrected_pvals[3],
                                    'Fold Enrichment': sibs_fe, 'cluster': -2, 'vs.': 'Social/Behavioral',
                                    'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'unadjusted': uncorrected_pvals[4],
                                    'Fold Enrichment': all_pros_fe, 'cluster': -1, 'vs.': 'Social/Behavioral',
                                    'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, siblings_df, all_pros_df,
                                    class0_df, class1_df, class3_df,
                                    ], 
                                       axis=0)
    
    order = ['variable', 'trend', 'cluster', 'vs.', 'unadjusted', 'value', 'Fold Enrichment']
    validation_subset = validation_subset[order]

    validation_subset.to_csv('../supp_tables/Supp_Table_12_class2_baseline.csv')
    

def get_stats_class3_baseline():
    # run function or load precomputed data
    #gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets() 
    
    with open('data/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('data/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('data/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('data/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # select LoF consequences
    consequences = [
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 
        'splice_donor_variant', 'start_lost', 'stop_lost', 
        'transcript_ablation'
        ]
   
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs 
    # with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of participants in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # compute enrichments for each gene set
    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['LoF'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['LoF']
    
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro[
            'mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['gene_set&consequence'].sum().tolist() 
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2

        # get p-values comparing each class to sibs using a t-test
        class0_pval = min(ttest_ind(
            class0, class3, alternative='less')[1], ttest_ind(class0, class3, alternative='greater')[1])
        class1_pval = min(ttest_ind(
            class1, class3, alternative='less')[1], ttest_ind(class1, class3, alternative='greater')[1])
        class2_pval = min(ttest_ind(
            class2, class3, alternative='less')[1], ttest_ind(class2, class3, alternative='greater')[1])
        siblings_pval = min(ttest_ind(
            sibs, class3, alternative='less')[1], ttest_ind(sibs, class3, alternative='greater')[1])
        all_pros_pval = min(ttest_ind(
            all_pros_data, class3, alternative='less')[1], ttest_ind(all_pros_data, class3, alternative='greater')[1])
        
        symbols = ['^', '^', 'v', 'v']

        uncorrected_pvals = [-np.log10(class0_pval), -np.log10(class1_pval),
                                -np.log10(class2_pval), 
                                -np.log10(siblings_pval), -np.log10(all_pros_pval)]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, 
                                   class2_pval, siblings_pval, all_pros_pval], 
                                   method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        sibs_pval = -np.log10(corrected[3])
        all_pros_pval = -np.log10(corrected[4])

        background = np.sum(class3)/num_class3       
        class0_fe = max((np.sum(class0)/num_class0)/background, background/(np.sum(class0)/num_class0))
        class1_fe = max((np.sum(class1)/num_class1)/background, background/(np.sum(class1)/num_class1))
        class2_fe = max((np.sum(class2)/num_class2)/background, background/(np.sum(class2)/num_class2))
        sibs_fe = max((np.sum(sibs)/num_sibs)/background, background/(np.sum(sibs)/num_sibs))
        all_pros_fe = max((np.sum(all_pros_data)/all_spids)/background, background/(np.sum(all_pros_data)/all_spids))

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'unadjusted': uncorrected_pvals[0],
                                  'Fold Enrichment': class0_fe, 'cluster': 0, 'vs.': 'Mixed ASD with DD',
                                  'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'unadjusted': uncorrected_pvals[1], 
                                  'Fold Enrichment': class1_fe, 'cluster': 1,  'vs.': 'Mixed ASD with DD',
                                  'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'unadjusted': uncorrected_pvals[2], 
                                  'Fold Enrichment': class2_fe, 'cluster': 2,  'vs.': 'Mixed ASD with DD', 
                                  'trend': trend}, index=[0])
        siblings_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'unadjusted': uncorrected_pvals[3],
                                    'Fold Enrichment': sibs_fe, 'cluster': -2, 'vs.': 'Mixed ASD with DD', 
                                    'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'unadjusted': uncorrected_pvals[4],
                                    'Fold Enrichment': all_pros_fe, 'cluster': -1, 'vs.': 'Mixed ASD with DD',
                                    'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, siblings_df, all_pros_df,
                                    class0_df, class1_df, class2_df,
                                    ], 
                                       axis=0)
    
    order = ['variable', 'trend', 'cluster', 'vs.', 'unadjusted', 'value', 'Fold Enrichment']
    validation_subset = validation_subset[order]

    validation_subset.to_csv('../supp_tables/Supp_Table_12_class3_baseline.csv')
    

def get_stats_class0_baseline():
    # run function or load precomputed data
    #gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets() 
    
    with open('data/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('data/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('data/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('data/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # select LoF consequences
    consequences = [
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 
        'splice_donor_variant', 'start_lost', 'stop_lost', 
        'transcript_ablation'
        ]
   
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs 
    # with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of participants in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # compute enrichments for each gene set
    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['LoF'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['LoF']
    
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro[
            'mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['gene_set&consequence'].sum().tolist() 
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class2 + class1 + class3

        # get p-values comparing each class to sibs using a t-test
        class2_pval = min(ttest_ind(
            class2, class0, alternative='less')[1], ttest_ind(class2, class0, alternative='greater')[1])
        class1_pval = min(ttest_ind(
            class1, class0, alternative='less')[1], ttest_ind(class1, class0, alternative='greater')[1])
        class3_pval = min(ttest_ind(
            class3, class0, alternative='less')[1], ttest_ind(class3, class0, alternative='greater')[1])
        siblings_pval = min(ttest_ind(
            sibs, class0, alternative='less')[1], ttest_ind(sibs, class0, alternative='greater')[1])
        all_pros_pval = min(ttest_ind(
            all_pros_data, class0, alternative='less')[1], ttest_ind(all_pros_data, class0, alternative='greater')[1])
        
        symbols = ['^', '^', 'v', 'v']

        uncorrected_pvals = [-np.log10(class2_pval), -np.log10(class1_pval),
                                -np.log10(class3_pval), 
                                -np.log10(siblings_pval), -np.log10(all_pros_pval)]
        
        # multiple testing correction
        corrected = multipletests([class2_pval, class1_pval, 
                                   class3_pval, siblings_pval, all_pros_pval], 
                                   method='fdr_bh')[1]
        class2_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class3_pval = -np.log10(corrected[2])
        sibs_pval = -np.log10(corrected[3])
        all_pros_pval = -np.log10(corrected[4])

        background = np.sum(class0)/num_class0       
        class2_fe = max((np.sum(class2)/num_class2)/background, background/(np.sum(class2)/num_class2))
        class1_fe = max((np.sum(class1)/num_class1)/background, background/(np.sum(class1)/num_class1))
        class3_fe = max((np.sum(class3)/num_class3)/background, background/(np.sum(class3)/num_class3))
        sibs_fe = max((np.sum(sibs)/num_sibs)/background, background/(np.sum(sibs)/num_sibs))
        all_pros_fe = max((np.sum(all_pros_data)/all_spids)/background, background/(np.sum(all_pros_data)/all_spids))

        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'unadjusted': uncorrected_pvals[0],
                                  'Fold Enrichment': class2_fe, 'cluster': 2, 'vs.': 'Moderate Challenges',
                                  'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'unadjusted': uncorrected_pvals[1], 
                                  'Fold Enrichment': class1_fe, 'cluster': 1, 'vs.': 'Moderate Challenges',
                                  'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'unadjusted': uncorrected_pvals[2], 
                                  'Fold Enrichment': class3_fe, 'cluster': 3,  'vs.': 'Moderate Challenges',
                                  'trend': trend}, index=[0])
        siblings_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'unadjusted': uncorrected_pvals[3],
                                    'Fold Enrichment': sibs_fe, 'cluster': -2, 'vs.': 'Moderate Challenges',
                                    'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'unadjusted': uncorrected_pvals[4],
                                    'Fold Enrichment': all_pros_fe, 'cluster': -1, 'vs.': 'Moderate Challenges',
                                    'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, siblings_df, all_pros_df,
                                    class2_df, class1_df, class3_df,
                                    ], 
                                       axis=0)
    
    order = ['variable', 'trend', 'cluster', 'vs.', 'unadjusted', 'value', 'Fold Enrichment']
    validation_subset = validation_subset[order]

    validation_subset.to_csv('../supp_tables/Supp_Table_12_class0_baseline.csv')
    

def get_stats_class1_baseline():
    # run function or load precomputed data
    #gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets() 
    
    with open('data/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('data/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('data/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('data/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # select LoF consequences
    consequences = [
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 
        'splice_donor_variant', 'start_lost', 'stop_lost', 
        'transcript_ablation'
        ]
   
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs 
    # with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of participants in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # compute enrichments for each gene set
    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['LoF'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['LoF']
    
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro[
            'mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['gene_set&consequence'].sum().tolist() 
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class2 + class0 + class3

        # get p-values comparing each class to sibs using a t-test
        class2_pval = min(ttest_ind(
            class2, class1, alternative='less')[1], ttest_ind(class2, class1, alternative='greater')[1])
        class0_pval = min(ttest_ind(
            class0, class1, alternative='less')[1], ttest_ind(class0, class1, alternative='greater')[1])
        class3_pval = min(ttest_ind(
            class3, class1, alternative='less')[1], ttest_ind(class3, class1, alternative='greater')[1])
        siblings_pval = min(ttest_ind(
            sibs, class1, alternative='less')[1], ttest_ind(sibs, class1, alternative='greater')[1])
        all_pros_pval = min(ttest_ind(
            all_pros_data, class1, alternative='less')[1], ttest_ind(all_pros_data, class1, alternative='greater')[1])
        
        symbols = ['^', '^', 'v', 'v']

        uncorrected_pvals = [-np.log10(class2_pval), -np.log10(class0_pval),
                                -np.log10(class3_pval), 
                                -np.log10(siblings_pval), -np.log10(all_pros_pval)]
        
        # multiple testing correction
        corrected = multipletests([class2_pval, class0_pval, 
                                   class3_pval, siblings_pval, all_pros_pval], 
                                   method='fdr_bh')[1]
        class2_pval = -np.log10(corrected[0])
        class0_pval = -np.log10(corrected[1])
        class3_pval = -np.log10(corrected[2])
        sibs_pval = -np.log10(corrected[3])
        all_pros_pval = -np.log10(corrected[4])

        background = np.sum(class1)/num_class1      
        class2_fe = max((np.sum(class2)/num_class2)/background, background/(np.sum(class2)/num_class2))
        class0_fe = max((np.sum(class0)/num_class0)/background, background/(np.sum(class0)/num_class0))
        class3_fe = max((np.sum(class3)/num_class3)/background, background/(np.sum(class3)/num_class3))
        sibs_fe = max((np.sum(sibs)/num_sibs)/background, background/(np.sum(sibs)/num_sibs))
        all_pros_fe = max((np.sum(all_pros_data)/all_spids)/background, background/(np.sum(all_pros_data)/all_spids))

        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'unadjusted': uncorrected_pvals[0],
                                  'Fold Enrichment': class2_fe, 'cluster': 2, 'vs.': 'Broadly Impacted',
                                  'trend': trend}, index=[0])
        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'unadjusted': uncorrected_pvals[1], 
                                  'Fold Enrichment': class0_fe, 'cluster': 0,  'vs.': 'Broadly Impacted',
                                  'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'unadjusted': uncorrected_pvals[2], 
                                  'Fold Enrichment': class3_fe, 'cluster': 3, 'vs.': 'Broadly Impacted',
                                  'trend': trend}, index=[0])
        siblings_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'unadjusted': uncorrected_pvals[3],
                                    'Fold Enrichment': sibs_fe, 'cluster': -2,  'vs.': 'Broadly Impacted',
                                    'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'unadjusted': uncorrected_pvals[4],
                                    'Fold Enrichment': all_pros_fe, 'cluster': -1, 'vs.': 'Broadly Impacted',
                                    'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, siblings_df, all_pros_df,
                                    class2_df, class0_df, class3_df,
                                    ], 
                                       axis=0)
    
    order = ['variable', 'trend', 'cluster', 'vs.', 'unadjusted', 'value', 'Fold Enrichment']
    validation_subset = validation_subset[order]

    validation_subset.to_csv('../supp_tables/Supp_Table_12_class1_baseline.csv')
    

def create_supp_table():
    class0 = pd.read_csv('../supp_tables/Supp_Table_12_class0_baseline.csv', index_col=0)
    class1 = pd.read_csv('../supp_tables/Supp_Table_12_class1_baseline.csv', index_col=0)
    class2 = pd.read_csv('../supp_tables/Supp_Table_12_class2_baseline.csv', index_col=0)
    class3 = pd.read_csv('../supp_tables/Supp_Table_12_class3_baseline.csv', index_col=0)

    combined = pd.concat([class0, class1, class2, class3], axis=0)

    # sort by variable
    combined = combined.sort_values('variable')
    combined.to_csv('../supp_tables/Supp_Table_12_combined_class_baseline.csv')


if __name__ == "__main__":
    make_gene_trend_figure()
    get_stats_class0_baseline()
    get_stats_class1_baseline()
    get_stats_class2_baseline()
    get_stats_class3_baseline()
    
    create_supp_table()
