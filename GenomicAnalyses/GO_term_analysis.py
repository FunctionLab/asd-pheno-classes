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
    dnvs_pro['consequence_missense'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_missense else 0)
    dnvs_pro['consequence_lof'] = dnvs_pro['Consequence'].apply(
        lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    all_genes_idx = 0 # retrieve index for all protein coding genes
    dnvs_pro[gene_set_names[all_genes_idx]] = dnvs_pro['name'].apply(
        lambda x: 1 if x in gene_sets[all_genes_idx] else 0)

    # further subset variants based on LOFTEE and AlphaMissense
    dnvs_pro['final_consequence_mis'] = dnvs_pro['consequence_missense'] * dnvs_pro['am_class']
    dnvs_pro['final_consequence_lof'] = dnvs_pro['consequence_lof'] * dnvs_pro['LoF']
        
    class_to_gene_set = {}
    for class_id in [0,1,2,3]:
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(gene_sets[all_genes_idx])]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        
        mis_gene_vars_for_class = gene_vars_for_class[
            gene_vars_for_class['final_consequence_mis'] == 1]
        lof_gene_vars_for_class = gene_vars_for_class[
            gene_vars_for_class['final_consequence_lof'] == 1]

        genes = [] # initialize list of genes for the class
        genes.extend(mis_gene_vars_for_class['name'].unique())
        genes.extend(lof_gene_vars_for_class['name'].unique())
       
        class_to_gene_set[class_id] = list(set(genes))

    # build a dataframe with the gene sets
    padded_class_to_gene_set = pad_lists(class_to_gene_set)
    class_to_gene_set_df = pd.DataFrame(padded_class_to_gene_set)
    class_to_gene_set_df.to_csv('data/impacted_genes_per_class.csv', index=False)

    # print the impacted genes for each class
    for class_id in [0,1,2,3]:
        print(f"Impacted genes for class {class_id}: \
                {len(class_to_gene_set[class_id])} genes in total.")
        print(', '.join([f'{gene}' for gene in class_to_gene_set[class_id]]))


def plot_GO_term_analysis(num_top_terms=3):
    # load and process GO term files for each class
    class_enrich = {}
    for i in range(4):
        biol = pd.read_csv(f'data/class{i}_bio_processes.csv')
        mol = pd.read_csv(f'data/class{i}_mol_functions.csv')

        # sort and keep top terms
        biol = biol.sort_values('Fold Enrichment', ascending=False).iloc[:num_top_terms]
        mol = mol.sort_values('Fold Enrichment', ascending=False).iloc[:num_top_terms]

        # combine and process
        df = pd.concat([biol, mol], axis=0)
        df['Enrichment FDR'] = -np.log10(df['Enrichment FDR'])
        df['Pathway'] = df['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))

        if i == 3:
            df.replace('double-strand break repair via alternative nonhomologous end joining',
                       'double-strand break repair', inplace=True)

        class_enrich[i] = df

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    titles = ['Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD with DD']
    xlims = [47, 110, 105, 34]
    x_shifts = [1.4, 2.6, 3.0, 1.0]

    max_bubble_size = class_enrich[0]['Enrichment FDR'].max() * 32

    for i, ax in enumerate(axes):
        df = class_enrich[i]

        ax.hlines(
            y=df['Pathway'],
            xmin=0,
            xmax=df['Fold Enrichment'] - x_shifts[i],
            color='black',
            linewidth=2.5,
            alpha=0.8
        )

        sns.scatterplot(
            x='Fold Enrichment',
            y='Pathway',
            data=df,
            ax=ax,
            color='black',
            s=df['Enrichment FDR'] * max_bubble_size
        )

        if i == 0:
            for j in range(3):
                ax.scatter([], [], s=(j + 1) * max_bubble_size, c='black', label=str(j + 1))
            ax.legend(
                scatterpoints=1,
                labelspacing=1.1,
                title='-log10(FDR)',
                title_fontsize=23,
                fontsize=18,
                loc='upper left',
                bbox_to_anchor=(1, 1)
            )

        ax.set_title(titles[i], fontsize=22)
        ax.set_xlim([0, xlims[i]])
        ax.tick_params(labelsize=18)
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(1.5)
            ax.spines[side].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Fold Enrichment' if i == 3 else '', fontsize=24 if i == 3 else 0)
        ax.set_ylabel('')

    fig.suptitle('Enrichment of pathways and processes', fontsize=26)
    fig.tight_layout()
    plt.savefig('figures/GO_enrichment_figure.png', bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == '__main__':
    get_impacted_genes_per_class()
    plot_GO_term_analysis()
