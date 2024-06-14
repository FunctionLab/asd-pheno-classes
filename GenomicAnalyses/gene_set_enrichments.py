import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
import pickle as rick
from statsmodels.stats.multitest import multipletests

from utils import load_dnvs, get_gene_sets


def volcano_LoF():
    '''
    Produce volcano plot for LoF variants.
    '''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
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

    FE = []
    pvals = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = [] # keep track of colors for plot

    gene_set_order = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes']
    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]

    for gene_set in gene_set_order:
        dnvs_pro[f'{gene_set}&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] 
        dnvs_sibs[f'{gene_set}&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF']

        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist() # these probands have no DNVs
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # pvalue comparing each class to sibs using a t-test
        background = (np.sum(sibs))/(num_sibs)
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)

        shape = shape_list.pop(0)
        shapes += [shape, shape, shape, shape]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        colors += ref_colors
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All genes', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='v', color='w', label='LoF-Intolerant', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='p', color='w', label='FMRP targets', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='ASD risk', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='d', color='w', label='SFARI', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='P', color='w', label='Satterstrom', markerfacecolor='gray', markersize=10),
                        Line2D([0], [0], marker='s', color='w', label='Brain-expressed', markerfacecolor='gray', markersize=10)]
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('dnLoF', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=1, linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/WES_volcano_plot_dnLoF.png', bbox_inches='tight', dpi=600)
    plt.close()


def volcano_missense():
    '''
    Produce volcano plot for de novo missense variants.
    '''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
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

    FE = []
    pvals = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = []

    gene_set_order = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes']
    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]
    for gene_set in gene_set_order:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']

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
        all_pros_data = class0 + class1 + class2 + class3
        
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs

        background = (np.sum(sibs))/(num_sibs)
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]

        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)

        shape = shape_list.pop(0)
        shapes += [shape, shape, shape, shape]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        colors += ref_colors

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('dnMis', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/WES_volcano_plot_dnMis.png', bbox_inches='tight', dpi=600)
    plt.close()


def volcano_inherited():
    '''
    Produce volcano plot for rare inherited LoF and missense variants.
    '''
    with open('data/spid_to_num_lof_rare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('data/spid_to_num_missense_rare_inherited.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)

    gfmm_labels = pd.read_csv('../PhenotypeValidations/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0)
    sibling_list = '../PhenotypeValidations/data/WES_5392_siblings_spids.txt'
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

    FE = []
    pvals = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = []

    gene_set_to_index = {gene_set: i for i, gene_set in enumerate(gene_set_names)}
    gene_set_order = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes']
    gene_set_indices = [gene_set_to_index[gene_set] for gene_set in gene_set_order]
    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]
    
    for i in gene_set_indices:
        gene_set = gene_set_names[i]
        class0 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]

        # pvalues comparing each class to sibs using a t-test
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]

        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        background = (np.sum(sibs))/(num_sibs)
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)

        shape = shape_list.pop(0)
        shapes += [shape, shape, shape, shape]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        colors += ref_colors

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhLoF', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/WES_volcano_plot_inhLoF.png', bbox_inches='tight', dpi=600)
    plt.close()

    # rare inherited missense variants
    FE = []
    pvals = []
    tick_labels = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = []

    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]

    for i in gene_set_indices:
        gene_set = gene_set_names[i]
        class0 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 3]
        sibs = [v[i] for k, v in sibs_to_num_missense.items()]

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]

        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        background = (np.sum(sibs))/(num_sibs)
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)

        shape = shape_list.pop(0)
        shapes += [shape, shape, shape, shape]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        colors += ref_colors
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhMis', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for i, txt in enumerate(tick_labels):
        ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/WES_volcano_plot_inhMis.png', bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == '__main__':
    volcano_LoF()
    volcano_missense()
    volcano_inherited()
