import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_dnvs, get_gene_sets, pad_lists


def get_impacted_genes_per_class():
    """
    For each class, get the set of genes impacted by high-impact 
    dnLoF or dnMissense variants.
    """
    dnvs_pro, _, _, _ = load_dnvs()
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 
                             'inframe_insertion', 'protein_altering_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 
                        'splice_acceptor_variant', 'splice_donor_variant', 
                        'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    dnvs_pro['consequence_lof'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    all_genes_idx = 0 # retrieve index for all protein coding genes
    dnvs_pro[gene_set_names[all_genes_idx]] = dnvs_pro['name'].apply(
        lambda x: 1 if x in gene_sets[all_genes_idx] else 0)

    # further subset variants based on LOFTEE and AlphaMissense
    dnvs_pro['final_consequence'] = dnvs_pro['consequence'] * dnvs_pro['am_class']
    dnvs_pro['final_consequence_lof'] = dnvs_pro['consequence_lof'] * dnvs_pro['LoF']
        
    class_to_gene_set = {}
    for class_id in [0,1,2,3]:
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(gene_sets[all_genes_idx])]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        
        mis_gene_vars_for_class = gene_vars_for_class[
            gene_vars_for_class['final_consequence'] == 1]
        lof_gene_vars_for_class = gene_vars_for_class[
            gene_vars_for_class['final_consequence_lof'] == 1]

        genes = [] # initialize list of genes for the class
        genes.extend(mis_gene_vars_for_class['name'].unique())
        genes.extend(lof_gene_vars_for_class['name'].unique())
       
        class_to_gene_set[class_id] = list(set(genes))
        print(class_id, len(class_to_gene_set[class_id]))
        print(class_to_gene_set[class_id])

    # build a dataframe with the gene sets
    padded_class_to_gene_set = pad_lists(class_to_gene_set)
    class_to_gene_set_df = pd.DataFrame(padded_class_to_gene_set)
    class_to_gene_set_df.to_csv('data/impacted_genes_per_class.csv', index=False)

    # print the impacted genes for each class
    for class_id in [0,1,2,3]:
        print(f"Impacted genes for class {class_id}: \
                {len(class_to_gene_set[class_id])} genes in total.")
        print(', '.join([f'{gene}' for gene in class_to_gene_set[class_id]]))


def GO_term_analysis(num_top_terms=3):
    # go term files - WES v3
    class0_biol_processes = pd.read_csv('data/class0_bio_processes.csv')
    class0_mol_functions = pd.read_csv('data/class0_mol_functions.csv')
    class1_biol_processes = pd.read_csv('data/class1_bio_processes.csv')
    class1_mol_functions = pd.read_csv('data/class1_mol_functions.csv')
    class2_biol_processes = pd.read_csv('data/class2_bio_processes.csv')
    class2_mol_functions = pd.read_csv('data/class2_mol_functions.csv')
    class3_biol_processes = pd.read_csv('data/class3_bio_processes.csv')
    class3_mol_functions = pd.read_csv('data/class3_mol_functions.csv')

    class0_biol_processes = class0_biol_processes.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class0_biol_processes = class0_biol_processes.iloc[:num_top_terms, :]
    class0_mol_functions = class0_mol_functions.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class0_mol_functions = class0_mol_functions.iloc[:num_top_terms, :]
    class1_biol_processes = class1_biol_processes.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class1_biol_processes = class1_biol_processes.iloc[:num_top_terms, :]
    class1_mol_functions = class1_mol_functions.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class1_mol_functions = class1_mol_functions.iloc[:num_top_terms, :]
    class2_biol_processes = class2_biol_processes.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class2_biol_processes = class2_biol_processes.iloc[:num_top_terms, :]
    class2_mol_functions = class2_mol_functions.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class2_mol_functions = class2_mol_functions.iloc[:num_top_terms, :]
    class3_biol_processes = class3_biol_processes.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class3_biol_processes = class3_biol_processes.iloc[:num_top_terms, :]
    class3_mol_functions = class3_mol_functions.sort_values(
        by=['Fold Enrichment'], ascending=False)
    class3_mol_functions = class3_mol_functions.iloc[:num_top_terms, :]
    
    class2_enrich = pd.concat(
        [class2_biol_processes, class2_mol_functions], axis=0)
    class3_enrich = pd.concat(
        [class3_biol_processes, class3_mol_functions], axis=0)
    class0_enrich = pd.concat(
        [class0_biol_processes, class0_mol_functions], axis=0)
    class1_enrich = pd.concat(
        [class1_biol_processes, class1_mol_functions], axis=0)

    # process
    class0_enrich['Enrichment FDR'] = class0_enrich['Enrichment FDR'].apply(
        lambda x: -np.log10(x))
    class1_enrich['Enrichment FDR'] = class1_enrich['Enrichment FDR'].apply(
        lambda x: -np.log10(x))
    class2_enrich['Enrichment FDR'] = class2_enrich['Enrichment FDR'].apply(
        lambda x: -np.log10(x))
    class3_enrich['Enrichment FDR'] = class3_enrich['Enrichment FDR'].apply(
        lambda x: -np.log10(x))
    
    class0_enrich['Pathway'] = class0_enrich['Pathway'].apply(
        lambda x: ' '.join(x.split()[1:]))
    class1_enrich['Pathway'] = class1_enrich['Pathway'].apply(
        lambda x: ' '.join(x.split()[1:]))
    class2_enrich['Pathway'] = class2_enrich['Pathway'].apply(
        lambda x: ' '.join(x.split()[1:]))
    class3_enrich['Pathway'] = class3_enrich['Pathway'].apply(
        lambda x: ' '.join(x.split()[1:]))
    class3_enrich = class3_enrich.replace(
        'double-strand break repair via alternative nonhomologous end joining', 
        'double-strand break repair')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(14,16))

    # Calculate scaling factor for legend
    max_bubble_size = class0_enrich['Enrichment FDR'].max() * 32

    ax0.hlines(y=class0_enrich['Pathway'], xmin=0, 
               xmax=class0_enrich['Fold Enrichment']-1.4, 
               color='black',linewidth=2.5, alpha=0.8)
    sns.scatterplot(x='Fold Enrichment', y='Pathway', palette='black', 
                    color='black', 
                    s=class0_enrich['Enrichment FDR'] * max_bubble_size, 
                    data=class0_enrich, ax=ax0)
    for i in range(3):
        ax0.scatter([], [], s=(i + 1) * max_bubble_size, 
                    c='black', label=str(i + 1))
    ax0.legend(scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', 
               title_fontsize=23, fontsize=18, loc='upper left', 
               bbox_to_anchor=(1, 1))
    ax0.set_xlim([0,47])
    ax0.set_title('Moderate Challenges', fontsize=22)
    ax0.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax0.spines[axis].set_linewidth(1.5)
        ax0.spines[axis].set_color('black')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.set_xlabel('')
    ax0.set_ylabel('')
    
    ax1.hlines(y=class1_enrich['Pathway'], xmin=0, 
               xmax=class1_enrich['Fold Enrichment']-2.6, color='black', 
               linewidth=2.5, alpha=0.8)
    sns.scatterplot(x='Fold Enrichment', y='Pathway', palette='black', 
                    color='black', 
                    s=class1_enrich['Enrichment FDR'] * max_bubble_size, 
                    data=class1_enrich, ax=ax1)
    ax1.set_xlim([0,110])
    ax1.set_title('Broadly Impacted', fontsize=22)
    ax1.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color('black')
    ax1.spines['top'].set_visible(False)        
    ax1.spines['right'].set_visible(False)    
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    ax2.hlines(y=class2_enrich['Pathway'], xmin=0, 
               xmax=class2_enrich['Fold Enrichment']-3, color='black', 
               linewidth=2.5, alpha=0.8)
    sns.scatterplot(x='Fold Enrichment', y='Pathway', palette='black', 
                    color='black', 
                    s=class2_enrich['Enrichment FDR'] * max_bubble_size, 
                    data=class2_enrich, ax=ax2)
    ax2.set_title('Social/Behavioral', fontsize=22)
    ax2.tick_params(labelsize=18)
    ax2.set_xlim([0,105])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    ax3.hlines(y=class3_enrich['Pathway'], xmin=0, 
               xmax=class3_enrich['Fold Enrichment']-1, color='black', 
               linewidth=2.5, alpha=0.8)
    sns.scatterplot(x='Fold Enrichment', y='Pathway', palette='black', 
                    color='black', 
                    s=class3_enrich['Enrichment FDR'] * max_bubble_size, 
                    data=class3_enrich, ax=ax3)
    ax3.set_title('Mixed ASD with DD', fontsize=22)
    ax3.set_xlim([0,34])
    ax3.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(1.5)
        ax3.spines[axis].set_color('black')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlabel('Fold Enrichment', fontsize=24)
    ax3.set_ylabel('')

    fig.suptitle('Enrichment of pathways and processes', fontsize=26)
    fig.tight_layout()
    plt.savefig('figures/GO_enrichment_figure.png', 
                bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == '__main__':
    get_impacted_genes_per_class()
    GO_term_analysis()
