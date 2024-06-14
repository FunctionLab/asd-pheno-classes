import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from utils import load_dnvs


def gene_constraint_analysis():
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # get gene sets
    pli = pd.read_csv('gene_sets/pLI_table.txt', sep='\t')
    pli = pli[['gene', 'pLI']]
    pli_higher = pli[pli['pLI'] >= 0.995]['gene'].tolist()
    pli_lower = pli[(pli['pLI'] >= 0.5) & (pli['pLI'] < 0.995)]['gene'].tolist()
    
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)

    gene_sets = [pli_higher, pli_lower]
    gene_set_names = ['pli_higher', 'pli_lower']
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,4.5))
    for gene_set, ax in zip(['pli_higher', 'pli_lower'], (ax1, ax2)):
        props = []
        stds = []
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

        class0_avg = np.mean(class0)
        class1_avg = np.mean(class1)
        class2_avg = np.mean(class2)
        class3_avg = np.mean(class3)
        sibs_avg = np.mean(sibs)

        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
        pvals = {k: pval for k, pval in enumerate(corrected)}

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

        x_values = list(np.arange(len(props)))
        y_values = props
        colors = ['dimgray', '#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
        for i in range(len(x_values)):
            ax.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
        ax.set_xlabel('')
        if ax == ax1:
            ax.set_ylabel('dnLoF per offspring', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.set_xticks(x_values)
        if gene_set == 'pli_highest':
            ax.set_title('pLI ≥ 0.995', fontsize=19)
        elif gene_set == 'pli_medium':
            ax.set_title('0.5 ≤ pLI < 0.995', fontsize=19)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 ≤ pLI < 0.9', fontsize=19)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        ax.grid(color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for grpidx in [0,1,2,3]:
            p_value = pvals[grpidx]
            x_position = grpidx+1
            y_position = y_values[grpidx+1]
            se_value = stds[grpidx+1]
            ypos = y_position + se_value 
            if p_value < 0.01:
                ax.annotate('***', xy=(x_position, ypos), ha='center', size=24, fontweight='bold')
            elif p_value < 0.05:
                ax.annotate('**', xy=(x_position, ypos), ha='center', size=24, fontweight='bold')
            elif p_value < 0.1:
                ax.annotate('*', xy=(x_position, ypos), ha='center', size=24, fontweight='bold')
    fig.tight_layout()
    fig.savefig('figures/WES_gene_constraint_avg_lof_per_offspring.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    gene_constraint_analysis()
