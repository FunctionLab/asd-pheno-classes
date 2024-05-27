from collections import defaultdict

import matplotlib.pyplot as plt
from utils import load_dnvs, get_gene_sets
from scipy.stats import fisher_exact, multipletests


def compute_odds_ratios():
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    consequences_syn = ['synonymous_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['synonymous'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_syn else 0)
    dnvs_sibs['synonymous'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_syn else 0)
    dnvs_pro['lof'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    gene_list = []
    gene_set_of_interest = 'asd_risk_genes' # fmrp_genes
    for i in range(len(gene_sets)):
        if gene_set_names[i] == gene_set_of_interest:
            gene_list += gene_sets[i]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    class_to_odds_ratios = defaultdict(list)

    dnvs_pro['syn_final_consequence'] = dnvs_pro['synonymous']
    dnvs_sibs['syn_final_consequence'] = dnvs_sibs['synonymous']
    dnvs_pro['lof_final_consequence'] = dnvs_pro['lof'] * dnvs_pro['LoF']
    dnvs_sibs['lof_final_consequence'] = dnvs_sibs['lof'] * dnvs_sibs['LoF']

    odds_ratios = defaultdict(list)
    class_to_gene_set = {}
    class_to_gene_set_log_fold_change = defaultdict(dict)
    pvals_lof = []
    pvals_syn = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(gene_list)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]

        # lof
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
        
        # compute odds ratios lof
        table_lof = [[lof_case_variant_present_count, lof_case_variant_absent_count],
                    [lof_sibs_case_variant_present_count, lof_sibs_case_variant_absent_count]]
        odds_ratio_lof, pval_lof = fisher_exact(table_lof)
        pvals_lof.append(pval_lof)

        # compute odds ratios syn
        table = [[syn_case_variant_present_count, syn_case_variant_absent_count],
                    [syn_sibs_case_variant_present_count, syn_sibs_case_variant_absent_count]]
        odds_ratio_syn, pval_syn = fisher_exact(table)
        pvals_syn.append(pval_syn)

        odds_ratios['lof'].append(odds_ratio_lof)
        odds_ratios['syn'].append(odds_ratio_syn)

    corrected_lof = multipletests(pvals_lof, method='fdr_bh', alpha=0.05)[1]
    corrected_syn = multipletests(pvals_syn, method='fdr_bh', alpha=0.05)[1]
    print(corrected_lof)
    print(corrected_syn)

    # plot odds ratios - only LoF
    odds_ratios_for_plotting = [odds_ratios['lof']]
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(3, 6))
    x_positions = range(len(odds_ratios_for_plotting))
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        y_values = [item[i] for item in odds_ratios_for_plotting]
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels(['dnLoF'], fontsize=22)
    ax.set_xlabel('')
    plt.yticks(fontsize=13)
    ax.set_ylabel('Odds ratio', fontsize=18)
    ax.set_title(f'{gene_set_of_interest}', fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-')
    plt.tight_layout()
    plt.savefig(f'figures/WES_odds_ratios_{gene_set_of_interest}_only_LoF.png', bbox_inches='tight')
    plt.close()

    # plot odds ratios - LoF and Syn
    odds_ratios_for_plotting = [odds_ratios['lof'], odds_ratios['syn']]
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5.5, 6))
    x_positions = range(len(odds_ratios_for_plotting))
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        y_values = [item[i] for item in odds_ratios_for_plotting]
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels(['dnLoF', 'dnSyn'], fontsize=22)
    plt.yticks(fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Odds ratio', fontsize=18)
    ax.set_title(f'{gene_set_of_interest}', fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-')
    plt.tight_layout()
    plt.savefig(f'figures/WES_odds_ratios_{gene_set_of_interest}_LoF_Syn.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    compute_odds_ratios()
