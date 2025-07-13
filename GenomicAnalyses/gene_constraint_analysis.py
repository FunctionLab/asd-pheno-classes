import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import scipy.stats as st

from utils import load_dnvs
from variant_set_enrichments import \
    get_star_labels, \
    draw_lines_and_stars


def gene_constraint_analysis():
    # load DNVs
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs()
    
    # get gene sets
    pli = pd.read_csv('gene_sets/pLI_table.txt', sep='\t')
    pli = pli[['gene', 'pLI']]
    pli_higher = pli[pli['pLI'] >= 0.995]['gene'].tolist()
    pli_lower = pli[(pli['pLI'] > 0.5) 
                    & (pli['pLI'] < 0.995)]['gene'].tolist()
    
    # define LoF consequences
    consequences_lof = ['stop_gained', 'frameshift_variant', 
                        'splice_acceptor_variant', 'splice_donor_variant', 
                        'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)

    # define pLI gene sets
    gene_sets = [pli_higher, pli_lower]
    gene_set_names = ['pli_higher', 'pli_lower']
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

    class_names = ['Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD/DD']
    supp_table = pd.DataFrame()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,4.5))
    for gene_set, ax in zip(['pli_higher', 'pli_lower'], (ax1, ax2)):
        props = []
        stds = []
        pvals = []
        confidence_intervals = []
        fold_changes = []
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
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        classes = [class0, class1, class2, class3]
        nums = [num_class0, num_class1, num_class2, num_class3]

        for cls_data, cls_size in zip(classes, nums):
            pvals.append(ttest_ind(cls_data, sibs, equal_var=False, alternative='greater')[1])
            fold_changes.append((np.sum(cls_data) / cls_size) / (np.sum(sibs) / num_sibs))
        
        comparison_map = {
            'pli_higher': [(1, 0), (1, 2), (1, 3), (0, 2), (3, 2), (0, 3)],
            'pli_lower':  [(1, 0), (0, 2), (0, 3), (1, 2), (1, 3), (3, 2)]
        }

        # select pairs based on gene set
        comparison_pairs = comparison_map.get(gene_set, [])

        # class data and sizes
        class_data = [class0, class1, class2, class3]
        class_sizes = [num_class0, num_class1, num_class2, num_class3]

        # compute p-values and fold changes
        for i, j in comparison_pairs:
            pvals.append(ttest_ind(class_data[i], class_data[j], equal_var=False, alternative='greater')[1])
            fc = (np.sum(class_data[i]) / class_sizes[i]) / (np.sum(class_data[j]) / class_sizes[j])
            fold_changes.append(fc)
        
        uncorrected_pvals = pvals
        corrected = multipletests(pvals, method='fdr_bh')[1]

        if gene_set == 'pli_higher':
            group1_names = [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], 
                            class_names[1], class_names[1], class_names[0], class_names[3], class_names[0]]
            vs_names = ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[2], 
                        class_names[3], class_names[2], class_names[2], class_names[3]]
            for i in range(10):
                supp_table = supp_table.append({
                    'variant type': 'dnLoF pLI ≥ 0.995',
                    'group1': group1_names[i],
                    'vs.': vs_names[i],
                    'p': uncorrected_pvals[i],
                    'fdr': corrected[i],
                    'fold change': fold_changes[i]
                }, ignore_index=True)
        elif gene_set == 'pli_lower':
            group1_names = [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], 
                            class_names[0], class_names[0], class_names[1], class_names[1], class_names[2]]
            vs_names = ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[2], 
                        class_names[3], class_names[2], class_names[3], class_names[3]]
            for i in range(10):
                supp_table = supp_table.append({
                    'variant type': 'dnLoF 0.5 ≤ pLI < 0.995',
                    'group1': group1_names[i],
                    'vs.': vs_names[i],
                    'p': uncorrected_pvals[i],
                    'fdr': corrected[i],
                    'fold change': fold_changes[i]
                }, ignore_index=True)
        
        pvals = {k: pval for k, pval in enumerate(corrected)}
        print(gene_set)
        print(pvals)

        # compute average dnLoF per offspring for each group
        groups = {
            'sibs': (sibs, num_sibs),
            'class0': (class0, num_class0),
            'class1': (class1, num_class1),
            'class2': (class2, num_class2),
            'class3': (class3, num_class3),
        }

        for data, n in groups.values():
            # mean proportion per group
            props.append(np.sum(data) / n)

            # standard error
            stds.append(np.std(data) / np.sqrt(n))

            # 95% confidence interval
            confidence_intervals.append(
                st.t.interval(confidence=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
            )
            
        x_values = list(np.arange(len(props)))
        y_values = props
        colors = ['dimgray', '#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']
        
        for i in range(len(x_values)):
            lower_err = y_values[i] - confidence_intervals[i][0] 
            upper_err = confidence_intervals[i][1] - y_values[i] 
            yerr = np.array([[lower_err], [upper_err]])

            ax.errorbar(
                x_values[i], y_values[i], yerr=yerr, 
                fmt='o', color=colors[i], markersize=20)

        ax.set_xlabel('')
        if ax == ax1:
            ax.set_ylabel('dnLoF per offspring', fontsize=18)
        else:
            ax.set_ylabel('')
        ax.set_xticks(x_values)
        if gene_set == 'pli_higher':
            ax.set_title('pLI ≥ 0.995', fontsize=19)
        elif gene_set == 'pli_lower':
            ax.set_title('0.5 ≤ pLI < 0.995', fontsize=19)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 ≤ pLI < 0.9', fontsize=18)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=17)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        ax.grid(color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if gene_set == 'pli_higher':
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([ymin, ymax * 1.25])

        for grpidx in [0,1,2,3]:
            p_value = pvals[grpidx]
            x_position = grpidx+1
            y_position = y_values[grpidx+1]
            se_value = stds[grpidx+1]
            #ypos = y_position + se_value 
            if gene_set == 'pli_higher':
                ypos = confidence_intervals[grpidx+1][1] -0.003
            elif gene_set == 'pli_lower':
                ypos = confidence_intervals[grpidx+1][1] -0.001
            if p_value < 0.01:
                ax.annotate('***', xy=(x_position, ypos), 
                            ha='center', size=24, fontweight='bold')
            elif p_value < 0.05:
                ax.annotate('**', xy=(x_position, ypos), 
                            ha='center', size=24, fontweight='bold')
            elif p_value < 0.1:
                ax.annotate('*', xy=(x_position, ypos), 
                            ha='center', size=24, fontweight='bold')

        custom_thresholds = {
            0.01: '***',
            0.05: '**',
            0.1: '*',
            1: 'ns'
        }

        custom_pvalues = list(pvals.values())[5:7]
        star_labels = get_star_labels(custom_pvalues, custom_thresholds)
        if gene_set == 'pli_higher':
            pairs = [(2, 3), (2, 4)]
            y_positions = [0.145, 0.155]
        
            # Call the function to draw lines and stars
            draw_lines_and_stars(ax, pairs, y_positions, star_labels, star_size=24, scaling=0.98)

    supp_table.to_csv('../supp_tables/supp_table_gene_constraint.csv', index=False)

    fig.tight_layout()
    fig.savefig(
        'figures/WES_gene_constraint_avg_lof_per_offspring.png', 
        bbox_inches='tight', 
        dpi=900
        )
    plt.close()


if __name__ == '__main__':
    gene_constraint_analysis()
