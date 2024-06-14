from collections import defaultdict

import matplotlib.pyplot as plt
from utils import load_dnvs, get_gene_sets
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def compute_odds_ratios():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [2.3, 1]})

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_benign = ['synonymous_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['lof'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()    
    gene_set_of_interest = 'fmrp_genes'
    for i in range(len(gene_sets)):
        if gene_set_names[i] == gene_set_of_interest:
            selected_gene_set = gene_sets[i]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    dnvs_pro['lof_final_consequence'] = dnvs_pro['lof'] * dnvs_pro['LoF']
    dnvs_sibs['lof_final_consequence'] = dnvs_sibs['lof'] * dnvs_sibs['LoF']

    odds_ratios = defaultdict(list)
    class_to_gene_set = {}
    class_to_gene_set_log_fold_change = defaultdict(dict)
    num_mutations = []
    props = []
    pvals_lof = []
    pvals_syn = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(selected_gene_set)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'].isin(selected_gene_set)]

        lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['lof_final_consequence'] == 1]
        lof_case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
        lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['lof_final_consequence'] == 1]
        lof_sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique()
        lof_case_variant_absent_count = class_count - lof_case_variant_present_count
        lof_sibs_case_variant_absent_count = num_sibs - lof_sibs_case_variant_present_count

        table_lof = [[lof_case_variant_present_count, lof_case_variant_absent_count],
                    [lof_sibs_case_variant_present_count, lof_sibs_case_variant_absent_count]]
        odds_ratio_lof, pval_lof = fisher_exact(table_lof)
        pvals_lof.append(pval_lof)

        odds_ratios['lof'].append(odds_ratio_lof)

    corrected_lof = multipletests(pvals_lof, method='fdr_bh', alpha=0.05)[1]
    print(corrected_lof)

    odds_ratios_for_plotting = [odds_ratios['lof']]
    bar_width = 0.15
    x_positions = range(len(odds_ratios_for_plotting))

    colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        y_values = [item[i] for item in odds_ratios_for_plotting]
        ax2.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    ax2.set_xticks([x + 2 * bar_width for x in x_positions])
    ax2.set_xticklabels(['dnLoF'], fontsize=18)
    ax2.set_xlabel('')
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylabel('')
    ax2.set_title('FMRP target genes', fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(which='both', axis='y', color='gray', linestyle='-')
    ax2.grid(which='both', axis='x', linestyle='')

    # SECOND SUBPLOT
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_benign = ['synonymous_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    dnvs_pro['synonymous'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
    dnvs_sibs['synonymous'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
    dnvs_pro['lof'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()    
    gene_set_of_interest = 'asd_risk_genes'
    for i in range(len(gene_sets)):
        if gene_set_names[i] == gene_set_of_interest:
            selected_gene_set = gene_sets[i]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    dnvs_pro['syn_final_consequence'] = dnvs_pro['synonymous']
    dnvs_sibs['syn_final_consequence'] = dnvs_sibs['synonymous']
    dnvs_pro['lof_final_consequence'] = dnvs_pro['lof'] * dnvs_pro['LoF'] 
    dnvs_sibs['lof_final_consequence'] = dnvs_sibs['lof'] * dnvs_sibs['LoF']

    odds_ratios = defaultdict(list)
    class_to_gene_set = {}
    class_to_gene_set_log_fold_change = defaultdict(dict)
    num_mutations = []
    props = []
    pvals_lof = []
    pvals_syn = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(selected_gene_set)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'].isin(selected_gene_set)]

        lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['lof_final_consequence'] == 1]
        lof_case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
        lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['lof_final_consequence'] == 1]
        lof_sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique()
        lof_case_variant_absent_count = class_count - lof_case_variant_present_count
        lof_sibs_case_variant_absent_count = num_sibs - lof_sibs_case_variant_present_count

        # syn
        syn_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['syn_final_consequence'] == 1]
        syn_case_variant_present_count = syn_gene_vars_for_class['spid'].nunique()
        syn_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['syn_final_consequence'] == 1]
        syn_sibs_case_variant_present_count = syn_gene_vars_sibs['spid'].nunique()
        syn_case_variant_absent_count = class_count - syn_case_variant_present_count
        syn_sibs_case_variant_absent_count = num_sibs - syn_sibs_case_variant_present_count
        
        table_lof = [[lof_case_variant_present_count, lof_case_variant_absent_count],
                    [lof_sibs_case_variant_present_count, lof_sibs_case_variant_absent_count]]
        odds_ratio_lof, pval_lof = fisher_exact(table_lof)
        pvals_lof.append(pval_lof)

        table_syn = [[syn_case_variant_present_count, syn_case_variant_absent_count],
                    [syn_sibs_case_variant_present_count, syn_sibs_case_variant_absent_count]]
        odds_ratio_syn, pval_syn = fisher_exact(table_syn)
        pvals_syn.append(pval_syn)

        odds_ratios['lof'].append(odds_ratio_lof)
        odds_ratios['syn'].append(odds_ratio_syn)

    # FDR CORRECTION on pvals_lof and pvals_syn
    corrected_lof = multipletests(pvals_lof, method='fdr_bh', alpha=0.05)[1]
    corrected_syn = multipletests(pvals_syn, method='fdr_bh', alpha=0.05)[1]
    print(corrected_lof)
    print(corrected_syn)

    odds_ratios_for_plotting = [odds_ratios['lof'], odds_ratios['syn']] # odds_ratios['mis'],
    bar_width = 0.15
    x_positions = range(len(odds_ratios_for_plotting))

    colors = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        y_values = [item[i] for item in odds_ratios_for_plotting]
        ax1.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    ax1.set_xticks([x + 2 * bar_width for x in x_positions])
    ax1.set_xticklabels(['dnLoF', 'dnSyn'], fontsize=18)
    ax1.set_xlabel('')
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel('Odds ratio', fontsize=18)
    ax1.set_title('ASD risk genes', fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(which='both', axis='y', color='gray', linestyle='-')
    ax1.grid(which='both', axis='x', linestyle='')
    plt.tight_layout()
    plt.savefig('figures/WES_odds_ratios_figure.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    compute_odds_ratios()
