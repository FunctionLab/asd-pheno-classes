import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from utils import load_dnvs, get_trend_celltype_gene_sets


def make_gene_trend_figure(fdr=0.05):
    gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets()
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # subset to target Consequences 
    consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    validation_subset = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF']

        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        sibs_rest_of_sample = class0 + class1 + class2 + class3
        class0_pval = ttest_ind(class0, sibs, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, alternative='greater')[1]
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])

        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]) < -np.log10(fdr):
            continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 3, 'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 2, 'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 1, 'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 0, 'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': -1, 'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -2, 'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, class0_df, class1_df, class2_df, class3_df], axis=0)

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
    validation_subset['variable'] = pd.Categorical(validation_subset['variable'], categories=order_of_variables, ordered=True)
    validation_subset = validation_subset.sort_values(['variable', 'cluster'])
    
    colors = ['black', 'purple', '#27AAE1', '#39B54A', '#FBB040', '#EE2A7B']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'mediumorchid', 0: '#FBB040', 1: '#EE2A7B', 2: '#39B54A', 3: '#27AAE1'})
    validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Moderate Challenges', 1: 'Broadly Impacted', 2: 'Social/Behavioral', 3: 'Mixed ASD with DD'})
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7,12))
    for _, row in validation_subset.iterrows():
        if row['value'] < -np.log10(fdr):
            plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*210, c='white', linewidth=2.5, edgecolors=row['color'], alpha=0.9)
        else:
            plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*230, c=row['color']) #

    # legend for bubble size
    for i in range(2, 16, 3):
        plt.scatter([], [], s=(i)*230, c='dimgray', label=str(i+1))
    plt.legend(scatterpoints=1, labelspacing=2.8, title='Fold Enrichment', title_fontsize=23, fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=20, rotation=35, ha='right')
    yticklabels = ['Inhibitory Interneuron MGE', 'Inhibitory Interneuron MGE', 'Principal Excitatory Neuron',
                    'Principal Excitatory Neuron', 'Glia', 'Inhibitory Interneuron CGE', 'Inhibitory Interneuron MGE', 'Principal Excitatory Neuron'][::-1]
    plt.yticks(ticks=range(8), labels=yticklabels, fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    plt.savefig(f'figures/WES_gene_trends_dnLoF_analysis.png', bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == "__main__":
    make_gene_trend_figure()
