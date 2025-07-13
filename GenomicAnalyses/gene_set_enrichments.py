import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
import pickle as rick
from statsmodels.stats.multitest import multipletests
import seaborn as sns

from utils import load_dnvs, get_gene_sets


def volcano_missense():
    '''
    Produce gene set volcano plot for de novo missense variants.
    '''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_missense = ['missense_variant', 'inframe_deletion', 
                             'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    
    gene_sets, gene_set_names = get_gene_sets()

    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + \
        zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    qvals = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = []

    supp_table = pd.DataFrame()

    gene_set_order = ['all_genes', 'lof_genes', 'fmrp_genes', 
                      'asd_risk_genes', 'sfari_genes1', 'satterstrom', 
                      'brain_expressed_genes']
    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]
    for gene_set in gene_set_order:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * \
                                           dnvs_pro['consequence'] * \
                                           dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * \
                                            dnvs_sibs['consequence'] * \
                                            dnvs_sibs['am_class']

        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[
            zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
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

        class0_pval = ttest_ind(
            class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(
            class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(
            class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(
            class3, sibs, equal_var=False, alternative='greater')[1]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]

        corrected = multipletests(
            [class0_pval, class1_pval, class2_pval, class3_pval], 
            method='fdr_bh')[1]
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
        qvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        colors += ref_colors

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if qvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], qvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], qvals[i], c='white', s=90, marker=shapes[i], 
                       linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('dnMis', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(
        'figures/WES_volcano_plot_dnMis.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()


def volcano_inherited():
    '''
    Produce volcano plot for rare inherited LoF and missense variants.
    '''
    # WES V3
    with open('data/spid_to_num_lof_rare_inherited_gnomad_wes_v3.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('data/spid_to_num_missense_rare_inherited_gnomad_wes_v3.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)

    gfmm_labels = pd.read_csv(
        '../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', 
        index_col=False, 
        header=0
        )
    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() 
                        if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() 
                            if k in spid_to_class}
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() 
                        if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() 
                            if k in sibling_list}

    gene_sets, gene_set_names = get_gene_sets()

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    FE = []
    pvals = []
    ref_colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    colors = []

    gene_set_to_index = {gene_set: i for i, gene_set in \
                         enumerate(gene_set_names)}
    gene_set_order = ['all_genes', 'lof_genes', 'fmrp_genes', 
                      'asd_risk_genes', 'sfari_genes1', 'satterstrom', 
                      'brain_expressed_genes']
    gene_set_indices = [gene_set_to_index[gene_set] for \
                        gene_set in gene_set_order]
    shapes = []
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shape_list = shape_list[0:len(gene_set_order)]
    
    for i in gene_set_indices:
        gene_set = gene_set_names[i]
        class0 = [v[i] for k, v in pros_to_num_ptvs.items() 
                  if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_ptvs.items() 
                  if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_ptvs.items() 
                  if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_ptvs.items() 
                  if spid_to_class[k] == 3]
        sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]

        # pvalues comparing each class to sibs using a t-test
        class0_pval = ttest_ind(
            class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(
            class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(
            class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(
            class3, sibs, equal_var=False, alternative='greater')[1]

        corrected = multipletests(
            [class0_pval, class1_pval, class2_pval, class3_pval], 
            method='fdr_bh')[1]
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
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], 
                       linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhLoF', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(
        'figures/WES_volcano_plot_inhLoF.png', 
        bbox_inches='tight', dpi=600)
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
        class0 = [v[i] for k, v in pros_to_num_missense.items() 
                  if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_missense.items() 
                  if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_missense.items() 
                  if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_missense.items() 
                  if spid_to_class[k] == 3]
        sibs = [v[i] for k, v in sibs_to_num_missense.items()]

        class0_pval = ttest_ind(
            class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(
            class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(
            class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(
            class3, sibs, equal_var=False, alternative='greater')[1]

        corrected = multipletests(
            [class0_pval, class1_pval, class2_pval, class3_pval], 
            method='fdr_bh')[1]
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
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], 
                       linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhMis', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', 
               linestyle='--', linewidth=1)
    for i, txt in enumerate(tick_labels):
        ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(
        'figures/WES_volcano_plot_inhMis.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()


def gene_set_bubble_plot(fdr=0.05):
    # run function or load precomputed data
    gene_sets, gene_set_names = get_gene_sets()
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
    shape_list = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shapes = []
    for gene_set in ['all_genes', 'lof_genes', 'fmrp_genes', 
                      'asd_risk_genes', 'sfari_genes1', 'satterstrom', 
                      'brain_expressed_genes'][::-1]:
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
        shapes += ['s', 's', 's', 's']
        
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

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval,
                                  'Fold Enrichment': class0_fe, 'cluster': 0, 'vs.': 'sibs',
                                  }, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval,
                                  'Fold Enrichment': class1_fe, 'cluster': 1, 'vs.': 'sibs',
                                  }, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval,
                                  'Fold Enrichment': class2_fe, 'cluster': 2, 'vs.': 'sibs',
                                  }, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval,
                                  'Fold Enrichment': class3_fe, 'cluster': 3, 'vs.': 'sibs'
                                  }, index=[0])
        validation_subset = pd.concat([validation_subset, 
                                       class0_df, class1_df, class2_df, class3_df], 
                                       axis=0)
    
    # duplicate 'value' and rename to 'p' for one
    validation_subset['p'] = validation_subset['value']
    
    # correct for multiple testing
    validation_subset['value'] = -np.log10(multipletests(
        validation_subset['value'], method='fdr_bh')[1])

    validation_subset.to_csv('../supp_tables/Supp_Table_gene_sets_sibs_baseline.csv')

    colors = ['black', '#27AAE1', '#39B54A', '#FBB040', '#EE2A7B']
    validation_subset['marker'] = validation_subset['cluster'].map(
        {-1: 'x', 0: '^', 1: '^', 2: '^', 3: '^'})
    validation_subset['color'] = validation_subset['cluster'].map(
        {-1: 'black', 0: '#FBB040', 1: '#EE2A7B', 
        2: '#39B54A', 3: '#27AAE1'})
    validation_subset['Cluster'] = validation_subset['cluster'].map(
        {-1: 'Siblings', 0: 'Moderate Challenges', 
         1: 'Broadly Impacted', 2: 'Social/Behavioral', 3: 'Mixed ASD with DD'})
    
    # plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7,12))
    for _, row in validation_subset.iterrows():
        if row['value'] < -np.log10(fdr):
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*250, c='white', linewidth=2.5, 
                        edgecolors='black', alpha=0.9, marker='o')
        else:
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*280, c=row['color'], marker='o')

    for i in range(3, 16, 3): # add legend sizes
        plt.scatter([], [], s=(i)*280, c='dimgray', label=str(i))
    plt.legend(scatterpoints=1, labelspacing=2.6, title='Fold Enrichment\nover siblings', 
               title_fontsize=23, fontsize=22, loc='upper left', 
               bbox_to_anchor=(1, 1))
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24, rotation=35, ha='right')
    plt.xlim(-0.5, 3.5)
    plt.ylabel('Gene Set', fontsize=29)
    yticklabels = ['All genes', 
                   'LoF-intolerant genes', 
                   'FMRP target genes', 
                   'ASD risk genes', 
                   'SFARI genes (1)', 'Satterstrom genes', 
                   'Brain expressed genes'][::-1]
    plt.yticks(ticks=range(len(yticklabels)), labels=yticklabels, fontsize=22)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.title('dnLoF enrichment', fontsize=32)
    plt.savefig(
        f'figures/gene_set_analysis_dnLoF_bubble_plot.png', 
        bbox_inches='tight', 
        dpi=900
        )
    plt.close()


def gene_set_bubble_plot_proband_baseline(fdr=0.1):
    # run function or load precomputed data
    gene_sets, gene_set_names = get_gene_sets()
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
    for gene_set in ['all_genes', 'lof_genes', 'fmrp_genes', 
                      'asd_risk_genes', 'sfari_genes1', 'satterstrom', 
                      'brain_expressed_genes'][::-1]:
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
        class0_pval = min(ttest_ind(
            class0, class1+class2+class3, alternative='less')[1], \
            ttest_ind(class0, class1+class2+class3, alternative='greater')[1])
        class1_pval = min(ttest_ind(
            class1, class0+class2+class3, alternative='greater')[1], \
            ttest_ind(class1, class0+class2+class3, alternative='less')[1])
        class2_pval = min(ttest_ind(
            class2, class0+class1+class3, alternative='less')[1], \
            ttest_ind(class2, class0+class1+class3, alternative='greater')[1])
        class3_pval = min(ttest_ind(
            class3, class0+class1+class2, alternative='greater')[1], \
            ttest_ind(class3, class0+class1+class2, alternative='less')[1])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + 
                          np.sum(class3))/(num_class0 + num_class1 + 
                                                          num_class2 + num_class3)
        background_class0 = (np.sum(class1) + np.sum(class2) + np.sum(class3))/(num_class1 + num_class2 + num_class3)
        background_class1 = (np.sum(class0) + np.sum(class2) + np.sum(class3))/(num_class0 + num_class2 + num_class3)
        background_class2 = (np.sum(class0) + np.sum(class1) + np.sum(class3))/(num_class0 + num_class1 + num_class3)
        background_class3 = (np.sum(class0) + np.sum(class1) + np.sum(class2))/(num_class0 + num_class1 + num_class2)

        background = np.sum(sibs)/num_sibs # sibling background
        class0_fe = (np.sum(class0)/num_class0)/background_class0
        class1_fe = (np.sum(class1)/num_class1)/background_class1
        class2_fe = (np.sum(class2)/num_class2)/background_class2
        class3_fe = (np.sum(class3)/num_class3)/background_class3
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        
        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval,
                                  'Fold Enrichment': class0_fe, 'cluster': 0, 'vs.': 'all other probands',
                                  }, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval,
                                  'Fold Enrichment': class1_fe, 'cluster': 1, 'vs.': 'all other probands',
                                  }, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval,
                                  'Fold Enrichment': class2_fe, 'cluster': 2, 'vs.': 'all other probands',
                                  }, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval,
                                  'Fold Enrichment': class3_fe, 'cluster': 3, 'vs.': 'all other probands',
                                  }, index=[0])
        validation_subset = pd.concat([validation_subset, 
                                       class0_df, class1_df, class2_df, class3_df], 
                                       axis=0)
    
    # duplicate 'value' and rename to 'p' for one
    validation_subset['p'] = validation_subset['value']
    
    # correct for multiple testing
    validation_subset['value'] = -np.log10(multipletests(
        validation_subset['value'], method='fdr_bh')[1])

    validation_subset.to_csv('../supp_tables/Supp_Table_gene_sets_probands_baseline.csv')

    colors = ['black', '#27AAE1', '#39B54A', '#FBB040', '#EE2A7B']
    validation_subset['marker'] = validation_subset['cluster'].map(
        {-1: 'x', 0: 'v', 1: '^', 2: 'v', 3: '^'})
    validation_subset['color'] = validation_subset['cluster'].map(
        {-1: 'black', 0: '#D4AF37', 1: '#2C3E50', 
        2: '#D4AF37', 3: '#2C3E50'})
    validation_subset['Cluster'] = validation_subset['cluster'].map(
        {-1: 'Siblings', 0: 'Moderate Challenges', 
         1: 'Broadly Impacted', 2: 'Social/Behavioral', 3: 'Mixed ASD with DD'})
    
    # plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7,12))
    for _, row in validation_subset.iterrows():
        if row['value'] < -np.log10(fdr):
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*1070, c='white', linewidth=2.5, 
                        edgecolors='black', alpha=0.9, marker='o')
        else:
            plt.scatter(row['Cluster'], row['variable'], 
                        s=row['Fold Enrichment']*1100, c=row['color'], marker='o')

    for i in range(1, 5, 1): # add legend sizes
        plt.scatter([], [], s=(i)*1000, c='dimgray', label=str(i))
    plt.legend(scatterpoints=1, labelspacing=2.6, title='Fold Enrichment\n over probands', 
               title_fontsize=23, fontsize=19, loc='upper left', 
               bbox_to_anchor=(1, 1))
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24, rotation=35, ha='right')
    plt.xlim(-0.7, 3.7)
    plt.ylabel('Gene Set', fontsize=29)
    yticklabels = ['All genes', 
                   'LoF-intolerant genes', 
                   'FMRP target genes', 
                   'ASD risk genes', 
                   'SFARI genes (1)', 'Satterstrom genes', 
                   'Brain expressed genes'][::-1]
    plt.yticks(ticks=range(len(yticklabels)), labels=yticklabels, fontsize=22)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    plt.title('dnLoF enrichment', fontsize=32)
    plt.savefig(
        f'figures/gene_set_analysis_dnLoF_bubble_plot_proband_baseline.png', 
        bbox_inches='tight', 
        dpi=900
        )
    plt.close()


def get_class_label_map():
    return {
        0: "Moderate Challenges",
        1: "Broadly Impacted",
        2: "Social/Behavioral",
        3: "Mixed ASD with DD"
    }


def compute_stats_class_baseline(baseline_class: int, out_csv: str):
    gene_sets, gene_set_names = get_gene_sets()
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    class_labels = get_class_label_map()

    consequences = {
        'stop_gained', 'frameshift_variant', 'splice_acceptor_variant',
        'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation'
    }

    for df in [dnvs_pro, dnvs_sibs]:
        df['consequence'] = df['Consequence'].apply(lambda x: int(x in consequences))

    for name, genes in zip(gene_set_names, gene_sets):
        for df in [dnvs_pro, dnvs_sibs]:
            df[name] = df['name'].apply(lambda x: int(x in genes))

    class_sizes = {
        i: dnvs_pro[dnvs_pro['class'] == i]['spid'].nunique()
        + zero_pro[zero_pro['mixed_pred'] == i]['spid'].nunique()
        for i in range(4)
    }

    validation_subset = pd.DataFrame()

    for gene_set in reversed(gene_set_names):
        dnvs_pro['gene_set&consequence'] = (
            dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF']
        )
        dnvs_sibs['gene_set&consequence'] = (
            dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF']
        )

        class_data = {
            i: dnvs_pro[dnvs_pro['class'] == i]
                .groupby('spid')['gene_set&consequence'].sum()
                .tolist()
            + zero_pro[zero_pro['mixed_pred'] == i]['count'].astype(int).tolist()
            for i in range(4)
        }

        # exclude baseline from comparison set
        other_classes = [i for i in range(4) if i != baseline_class]

        rows = []
        baseline_data = class_data[baseline_class]
        baseline_mean = np.sum(baseline_data) / class_sizes[baseline_class]

        for i in other_classes:
            comp_data = class_data[i]
            pval = min(
                ttest_ind(comp_data, baseline_data, alternative='less')[1],
                ttest_ind(comp_data, baseline_data, alternative='greater')[1],
            )
            comp_mean = np.sum(comp_data) / class_sizes[i]
            fold_enrichment = max(comp_mean / baseline_mean, baseline_mean / comp_mean)
            row = {
                'variable': gene_set,
                'value': pval,
                'Fold Enrichment': fold_enrichment,
                'cluster': class_labels[i],
                'vs.': class_labels[baseline_class],
            }
            rows.append(row)

        validation_subset = pd.concat(
            [validation_subset, pd.DataFrame(rows)], axis=0, ignore_index=True
        )

    # duplicate 'value' and rename to 'p'
    validation_subset['p'] = validation_subset['value']

    # correct for multiple testing
    validation_subset['value'] = -np.log10(
        multipletests(validation_subset['value'], method='fdr_bh')[1]
    )

    validation_subset.to_csv(out_csv, index=False)


def create_supp_table():
    # combine the supp tables produced, remove duplicate comparisons, and correct across all comparisons
    class1 = pd.read_csv('../supp_tables/Supp_Table_gene_sets_class1_baseline.csv', index_col=0)
    class2 = pd.read_csv('../supp_tables/Supp_Table_gene_sets_class2_baseline.csv', index_col=0)
    class0 = pd.read_csv('../supp_tables/Supp_Table_gene_sets_class0_baseline.csv', index_col=0)
    sibs_baseline = pd.read_csv('../supp_tables/Supp_Table_gene_sets_sibs_baseline.csv', index_col=0)
    proband_baseline = pd.read_csv('../supp_tables/Supp_Table_gene_sets_probands_baseline.csv', index_col=0)

    # concatenate tables
    combined = pd.concat([sibs_baseline, proband_baseline, class1, class2, class0], axis=0)
    
    combined['fdr'] = multipletests(combined['p'], method='fdr_bh')[1]
    
    # print all rows
    combined = combined[['variable', 'cluster', 'vs.', 'p', 'fdr', 'Fold Enrichment']]
    combined.to_csv('../supp_tables/Supp_Table_gene_sets_all.csv')


if __name__ == '__main__':
    gene_set_bubble_plot()
    gene_set_bubble_plot_proband_baseline()
    volcano_missense()
    volcano_inherited()
    
    for i in range(4):
        compute_stats_class_baseline(baseline_class=i, out_csv=f'../supp_tables/Supp_Table_gene_sets_class{i}_baseline.csv')

    create_supp_table()
