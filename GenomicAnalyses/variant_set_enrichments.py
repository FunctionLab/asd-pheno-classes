import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, sem
from statsmodels.stats.multitest import multipletests
import pickle as rick
import scipy.stats as st

from utils import load_dnvs, get_gene_sets, get_star_labels, draw_lines_and_stars


def compute_variant_set_proportions():
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()

    consequences_missense = ['missense_variant', 'inframe_deletion', 
                             'inframe_insertion', 'protein_altering_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 
                        'splice_acceptor_variant', 'splice_donor_variant', 
                        'start_lost', 'stop_lost', 'transcript_ablation']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['lof_consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof_consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)
    dnvs_pro['mis_consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['mis_consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    
    gene_sets, gene_set_names = get_gene_sets()

    # for each gene set, annotate dnvs_pro and dnvs_sibs with 
    # gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(
            lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[
        dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[
            zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[
        dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[
            zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[
        dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[
            zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[
        dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[
            zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    gene_sets_to_keep = ['all_genes'] # we want all genes for this analysis
    
    supp_table = pd.DataFrame()
    class_names = ['Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD/DD']

    # dnLoF + dnMis 
    props = []
    stds = []
    confidence_intervals = []
    pvals = []
    fold_changes = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['lof_gene_set&consequence'] = dnvs_pro[gene_set] * \
            dnvs_pro['lof_consequence'] * dnvs_pro['LoF']
        dnvs_sibs['lof_gene_set&consequence'] = dnvs_sibs[gene_set] * \
            dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF']
        dnvs_pro['mis_gene_set&consequence'] = dnvs_pro[gene_set] * \
            dnvs_pro['mis_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['mis_gene_set&consequence'] = dnvs_sibs[gene_set] * \
            dnvs_sibs['mis_consequence'] * dnvs_sibs['am_class']

        class0_lof = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['lof_gene_set&consequence'].sum().tolist()
        class0_mis = dnvs_pro[dnvs_pro['class'] == 0].groupby(
            'spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[
            zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)] + zero_class0
        class1_lof = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['lof_gene_set&consequence'].sum().tolist()
        class1_mis = dnvs_pro[dnvs_pro['class'] == 1].groupby(
            'spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[
            zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)] + zero_class1
        class2_lof = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['lof_gene_set&consequence'].sum().tolist()
        class2_mis = dnvs_pro[dnvs_pro['class'] == 2].groupby(
            'spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class2 = zero_pro[
            zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)] + zero_class2
        class3_lof = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['lof_gene_set&consequence'].sum().tolist()
        class3_mis = dnvs_pro[dnvs_pro['class'] == 3].groupby(
            'spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[
            zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)] + zero_class3
        sibs_lof = dnvs_sibs.groupby(
            'spid')['lof_gene_set&consequence'].sum().tolist()
        sibs_mis = dnvs_sibs.groupby(
            'spid')['mis_gene_set&consequence'].sum().tolist()
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)] + \
                zero_sibs['count'].astype(int).tolist()
        
        # compute means
        groups = {
            'sibs': sibs,
            'class0': class0,
            'class1': class1,
            'class2': class2,
            'class3': class3
        }
        
        for data in groups.values():
            props.append(np.mean(data))  # equivalent to sum / n
            stds.append(sem(data))
            confidence_intervals.append(
                st.t.interval(confidence=0.95, df=len(data) - 1, loc=np.mean(data), scale=sem(data))
            )
    
        classes = [class0, class1, class2, class3]
        sizes = [num_class0, num_class1, num_class2, num_class3]
        
        # compare each class to sibs
        for cls_data, cls_size in zip(classes, sizes):
            pvals.append(ttest_ind(cls_data, sibs, equal_var=False, alternative='greater').pvalue)
            fold_changes.append((np.sum(cls_data) / cls_size) / (np.sum(sibs) / num_sibs))
        
        # case-group pairwise comparisons
        comparison_pairs = [(1, 0), (1, 2), (1, 3), (0, 2), (3, 2), (0, 3)]
        for i, j in comparison_pairs:
            pvals.append(ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue)
            fold_changes.append((np.sum(classes[i]) / sizes[i]) / (np.sum(classes[j]) / sizes[j]))
    
        uncorrected_pvals = pvals
        pvals = multipletests(pvals, method='fdr_bh')[1]

        # add to supp table - one row per comparison
        index_group1 = [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], 
                        class_names[1], class_names[1], class_names[0], class_names[3], class_names[0]]
        index_vs = ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[2], 
                    class_names[3], class_names[2], class_names[2], class_names[3]]
        for i in range(10):
            supp_table = supp_table.append({
                'variant type': 'dnLoF + dnMis',
                'group1': index_group1[i],
                'vs.': index_vs[i],
                'p': uncorrected_pvals[i],
                'fdr': pvals[i],
                'fold change': fold_changes[i]
            }, ignore_index=True)
        
        pvals = {i: pval for i, pval in enumerate(pvals)} 
      
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', '#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    
    # plot with 95% confidence intervals
    for i in range(len(x_values)):
        lower_err = y_values[i] - confidence_intervals[i][0] 
        upper_err = confidence_intervals[i][1] - y_values[i] 
        yerr = np.array([[lower_err], [upper_err]])

        ax[0].errorbar(
            x_values[i], y_values[i], yerr=yerr, 
            fmt='o', color=colors[i], markersize=20)

    ax[0].set_xlabel('')
    ax[0].set_ylabel('Count per offspring', fontsize=16)
    ax[0].set_xticks(x_values)
    ax[0].tick_params(labelsize=16, axis='y')
    ax[0].set_title('High-impact de novo variants', fontsize=17)
    ax[0].set_axisbelow(True)
    y_min, y_max = ax[0].get_ylim()
    ax[0].set_ylim([y_min, y_max * 1.2])
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
    for axis in ['top','bottom','left','right']:
        ax[0].spines[axis].set_linewidth(1.5)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].grid(color='gray', linestyle='-', linewidth=0.5)
    ax[0].set_xticklabels(['']*len(x_values))

    # add significance to plot
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        ypos = confidence_intervals[grpidx+1][1] - 0.007
        if p_value < 0.01:
            ax[0].annotate('***', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')
        elif p_value < 0.05:
            ax[0].annotate('**', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')
        elif p_value < 0.1:
            ax[0].annotate('*', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')

    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }

    custom_pvalues = list(pvals.values())[4:7]
    star_labels = get_star_labels(custom_pvalues, custom_thresholds)
    pairs = [(1, 2), (2, 3), (2, 4)] 
    y_positions = [0.6, 0.635, 0.67] 
    draw_lines_and_stars(ax[0], pairs, y_positions, star_labels)

    # inhLoF + inhMis
    # gnomAD-filtered rare inherited variants from 
    # WES V3 
    with open('data/spid_to_num_lof_rare_inherited_gnomad_wes_v3.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('data/spid_to_num_missense_rare_inherited_gnomad_wes_v3.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)

    gfmm_labels = pd.read_csv('../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv')
    gfmm_labels = gfmm_labels.rename(
        columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() 
                        if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() 
                            if k in spid_to_class}

    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
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

    gene_set_to_index = {gene_set: i for i, gene_set in enumerate(gene_set_names)}
    gene_sets_to_keep = ['all_genes']
    gene_set_indices = [gene_set_to_index[gene_set] for gene_set in gene_sets_to_keep]
    
    props = []
    stds = []
    pvals = []
    confidence_intervals = []
    fold_changes = []
    for i in gene_set_indices:
        class0_lof = [v[i] for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 0]
        class0_mis = [v[i] for k, v in pros_to_num_missense.items() 
                      if spid_to_class[k] == 0]
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)]
        class1_lof = [v[i] for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 1]
        class1_mis = [v[i] for k, v in pros_to_num_missense.items() 
                      if spid_to_class[k] == 1]
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)]
        class2_lof = [v[i] for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 2]
        class2_mis = [v[i] for k, v in pros_to_num_missense.items() 
                      if spid_to_class[k] == 2]
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)]
        class3_lof = [v[i] for k, v in pros_to_num_ptvs.items() 
                      if spid_to_class[k] == 3]
        class3_mis = [v[i] for k, v in pros_to_num_missense.items() 
                      if spid_to_class[k] == 3]
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)]
        sibs_lof = [v[i] for k, v in sibs_to_num_ptvs.items()]
        sibs_mis = [v[i] for k, v in sibs_to_num_missense.items()]
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)]

        # compute means
        groups = {
            'sibs': (sibs, num_sibs),
            'class0': (class0, num_class0),
            'class1': (class1, num_class1),
            'class2': (class2, num_class2),
            'class3': (class3, num_class3),
        }
        
        for data, n in groups.values():
            props.append(np.sum(data) / n)
            stds.append(np.std(data) / np.sqrt(n))
            confidence_intervals.append(
                st.t.interval(confidence=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
            )
    
        # hypothesis testing
        classes = [class0, class1, class2, class3]
        sizes = [num_class0, num_class1, num_class2, num_class3]
        
        # class vs sibs comparisons
        for cls_data, cls_size in zip(classes, sizes):
            pvals.append(ttest_ind(cls_data, sibs, equal_var=False, alternative='greater').pvalue)
            fold_changes.append((np.sum(cls_data) / cls_size) / (np.sum(sibs) / num_sibs))
        
        # case-group comparisons
        comparison_pairs = [(3, 0), (3, 2), (0, 2), (3, 1), (1, 2), (1, 0)]
        for i, j in comparison_pairs:
            pvals.append(ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue)
            fold_changes.append((np.sum(classes[i]) / sizes[i]) / (np.sum(classes[j]) / sizes[j]))
        
        uncorrected_pvals = pvals
        pvals = multipletests(pvals, method='fdr_bh')[1]
        group1_names = [class_names[0], class_names[1], class_names[2], class_names[3], class_names[3], 
                        class_names[3], class_names[0], class_names[3], class_names[1], class_names[1]]
        vs_names = ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[2], 
                    class_names[2], class_names[1], class_names[2], class_names[0]]
        for i in range(10):
            supp_table = supp_table.append({
                'variant type': 'inhLoF + inhMis',
                'group1': group1_names[i],
                'vs.': vs_names[i],
                'p': uncorrected_pvals[i],
                'fdr': pvals[i],
                'fold change': fold_changes[i]
            }, ignore_index=True)
        pvals = {i: pval for i, pval in enumerate(pvals)}
        
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', '#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
    
    for i in range(len(x_values)):
        lower_err = y_values[i] - confidence_intervals[i][0]
        upper_err = confidence_intervals[i][1] - y_values[i]
        yerr = np.array([[lower_err], [upper_err]])
        ax[1].errorbar(x_values[i], y_values[i], yerr=yerr, 
                       fmt='o', color=colors[i], markersize=20)
    ax[1].set_xlabel('')
    ax[1].set_xticks(x_values)
    ax[1].tick_params(labelsize=16, axis='y')
    ax[1].set_title('High-impact rare inherited variants', fontsize=17)
    ax[1].set_axisbelow(True)
    
    for axis in ['top','bottom','left','right']:
        ax[1].spines[axis].set_linewidth(1.5)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].grid(color='gray', linestyle='-', linewidth=0.5)
    ax[1].set_xticklabels(['']*len(x_values))

    # add significance to plot
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        ypos = confidence_intervals[grpidx+1][1] - 0.11
        if p_value < 0.01:
            ax[1].annotate('***', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')
        elif p_value < 0.05:
            ax[1].annotate('**', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')
        elif p_value < 0.1:
            ax[1].annotate('*', xy=(x_position, ypos), 
                           ha='center', size=20, fontweight='bold')

    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }

    custom_pvalues = list(pvals.values())[4:7] 
    custom_pvalues.append(list(pvals.values())[8])
    star_labels = get_star_labels(custom_pvalues, custom_thresholds)
    pairs = [(1, 4), (3, 4), (1,3), (2,4)] 
    y_positions = [52, 52.6, 53.2, 53.8] 
    draw_lines_and_stars(ax[1], pairs, y_positions, star_labels)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(
        'figures/WES_LoF_combined_Mis_props_scatter.png', 
        bbox_inches='tight', 
        dpi=900
        )
    plt.close()

    supp_table.to_csv('../supp_tables/Supp_Table_intergroup_variant_set_enrichments.csv', index=False)


if __name__ == "__main__":
    compute_variant_set_proportions()
