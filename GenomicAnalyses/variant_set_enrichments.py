import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, multipletests
import pickle as rick

from utils import load_dnvs, get_gene_sets


def compute_variant_set_proportions():
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['lof_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_pro['mis_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['mis_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    gene_sets, gene_set_names = get_gene_sets()

    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    gene_sets_to_keep = ['all_genes']
    
    # dnLoF + dnMis 
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['lof_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['lof_consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags'] 
        dnvs_sibs['lof_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        dnvs_pro['mis_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['mis_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['mis_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['mis_consequence'] * dnvs_sibs['am_class']

        class0_lof = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class0_mis = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)] + zero_class0
        class1_lof = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class1_mis = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)] + zero_class1
        class2_lof = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class2_mis = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)] + zero_class2
        class3_lof = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class3_mis = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)] + zero_class3
        sibs_lof = dnvs_sibs.groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        sibs_mis = dnvs_sibs.groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)] + zero_sibs['count'].astype(int).tolist()
        
        props.append(np.sum(sibs)/num_sibs)
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)

        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))

        pvals = []
        pvals.append(stats.ttest_ind(class0, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class1, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class2, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class3, sibs, equal_var=False, alternative='greater').pvalue)
        pvals = multipletests(pvals, method='fdr_bh')[1]
        pvals = {i: pval for i, pval in enumerate(pvals)}
        break 
        
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
    for i in range(len(x_values)):
        ax[0].errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Count per offspring', fontsize=16)
    ax[0].set_xticks(x_values)
    ax[0].tick_params(labelsize=16, axis='y')
    ax[0].set_title('High-impact de novo variants', fontsize=21)
    ax[0].set_axisbelow(True)
    for axis in ['top','bottom','left','right']:
        ax[0].spines[axis].set_linewidth(1.5)
        ax[0].spines[axis].set_color('black')
    ax[0].grid(color='gray', linestyle='-', linewidth=0.5)

    # add significance stars to plot
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx+1]
        se_value = stds[grpidx+1]
        ypos = y_position + se_value - 0.001
        if p_value < 0.01:
            ax[0].annotate('***', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.05:
            ax[0].annotate('**', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.1:
            ax[0].annotate('*', xy=(x_position, ypos), ha='center', size=20)
    
    # inhLoF + inhMis
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited.pkl', 'rb') as f: # _90patho
        spid_to_num_missense = rick.load(f)

    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' 

    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in spid_to_class}
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in sibling_list}

    gene_sets, gene_set_names = get_gene_sets()

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    gene_set_to_index = {gene_set: i for i, gene_set in enumerate(gene_set_names)}
    gene_sets_to_keep = ['all_genes']
    gene_set_indices = [gene_set_to_index[gene_set] for gene_set in gene_sets_to_keep]
    
    props = []
    stds = []
    for i in gene_set_indices:
        class0_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class0_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 0]
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)]
        class1_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class1_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 1]
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)]
        class2_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class2_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 2]
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)]
        class3_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        class3_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 3]
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)]
        all_pros_data = class0 + class1 + class2 + class3
        sibs_lof = [v[i] for k, v in sibs_to_num_ptvs.items()]
        sibs_mis = [v[i] for k, v in sibs_to_num_missense.items()]
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)]

        props.append(np.sum(sibs)/num_sibs)
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)

        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))

        # hypothesis testing
        pvals = []
        pvals.append(stats.ttest_ind(class0, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class1, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class2, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class3, sibs, equal_var=False, alternative='greater').pvalue)
        pvals = multipletests(pvals, method='fdr_bh')[1]
        pvals = {i: pval for i, pval in enumerate(pvals)}
        break
    
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
    for i in range(len(x_values)):
        ax[1].errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Count per offspring', fontsize=16)
    ax[1].set_xticks(x_values)
    ax[1].tick_params(labelsize=16, axis='y')
    ax[1].set_title('High-impact rare inherited variants', fontsize=21)
    ax[1].set_axisbelow(True)
    for axis in ['top','bottom','left','right']:
        ax[1].spines[axis].set_linewidth(1.5)
        ax[1].spines[axis].set_color('black')
    ax[1].grid(color='gray', linestyle='-', linewidth=0.5)

    # add significance stars to plot
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx+1]
        se_value = stds[grpidx+1]
        ypos = y_position + se_value-0.05
        if p_value < 0.01:
            ax[1].annotate('***', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.05:
            ax[1].annotate('**', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.1:
            ax[1].annotate('*', xy=(x_position, ypos), ha='center', size=20)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig('figures/WES_LoF_combined_Mis_props_scatter.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    compute_variant_set_proportions()
