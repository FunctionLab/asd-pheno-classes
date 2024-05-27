import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import sort_and_select_top


def GO_term_analysis(num_top_terms=3):
    class0_biol_processes = pd.read_csv('data/class0_top_pathways.csv')
    class0_mol_functions = pd.read_csv('data/class0_top_mol_functions.csv')
    class1_biol_processes = pd.read_csv('data/class1_top_pathways.csv')
    class1_mol_functions = pd.read_csv('data/class1_top_mol_functions.csv')
    class2_biol_processes = pd.read_csv('data/class2_top_pathways.csv')
    class2_mol_functions = pd.read_csv('data/class2_top_mol_functions.csv')
    class3_biol_processes = pd.read_csv('data/class3_top_pathways.csv')
    class3_mol_functions = pd.read_csv('data/class3_top_mol_functions.csv')
    
    dataframes = {
    'class0_biol_processes': class0_biol_processes,
    'class0_mol_functions': class0_mol_functions,
    'class1_biol_processes': class1_biol_processes,
    'class1_mol_functions': class1_mol_functions,
    'class2_biol_processes': class2_biol_processes,
    'class2_mol_functions': class2_mol_functions,
    'class3_biol_processes': class3_biol_processes,
    'class3_mol_functions': class3_mol_functions
    }

    for name, df in dataframes.items():
        dataframes[name] = sort_and_select_top(df, num_top_terms)

    # Unpack the modified dataframes back into individual variables
    class0_biol_processes = dataframes['class0_biol_processes']
    class0_mol_functions = dataframes['class0_mol_functions']
    class1_biol_processes = dataframes['class1_biol_processes']
    class1_mol_functions = dataframes['class1_mol_functions']
    class2_biol_processes = dataframes['class2_biol_processes']
    class2_mol_functions = dataframes['class2_mol_functions']
    class3_biol_processes = dataframes['class3_biol_processes']
    class3_mol_functions = dataframes['class3_mol_functions']

    # concat biological processes and molecular functions
    class0_enrich = pd.concat([class0_biol_processes, class0_mol_functions], axis=0)
    class1_enrich = pd.concat([class1_biol_processes, class1_mol_functions], axis=0)
    class2_enrich = pd.concat([class2_biol_processes, class2_mol_functions], axis=0)
    class3_enrich = pd.concat([class3_biol_processes, class3_mol_functions], axis=0)
    
    class0_enrich['Enrichment FDR'] = class0_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class1_enrich['Enrichment FDR'] = class1_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class2_enrich['Enrichment FDR'] = class2_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class3_enrich['Enrichment FDR'] = class3_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    # remove GO term ID from pathway name
    class0_enrich['Pathway'] = class0_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class1_enrich['Pathway'] = class1_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class2_enrich['Pathway'] = class2_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class3_enrich['Pathway'] = class3_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class3_enrich = class3_enrich.replace('double-strand break repair via alternative nonhomologous end joining', 'double-strand break repair') # simplify term
    
    # bubble plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(13.5,16))
    
    # class 0
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("violet", as_cmap=True), s=450, data=class0_enrich, edgecolor='black', linewidth=1, ax=ax0)
    handles, labels = ax0.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax0.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax0.set_title('ASD-Low Support Needs', fontsize=28)
    ax0.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax0.spines[axis].set_linewidth(1.5)
        ax0.spines[axis].set_color('black')
    ax0.set_xlabel('')
    ax0.set_ylabel('')
    for i in range(num_top_terms): # color y tick labels
        ax0.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax0.get_yticklabels()[i].set_color('purple')
    
    # class 1
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("red", as_cmap=True), s=450, data=class1_enrich, edgecolor='black', linewidth=1, ax=ax1)
    handles, labels = ax1.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax1.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_title('ASD-High Support Needs', fontsize=28)
    ax1.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color('black')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    for i in range(num_top_terms):
        ax1.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax1.get_yticklabels()[i].set_color('purple')
    
    # class 2
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("limegreen", as_cmap=True), s=450, data=class2_enrich, edgecolor='black', linewidth=1, ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax2.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax2.set_title('ASD-Social/RRB', fontsize=28)
    ax2.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_color('black')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    for i in range(num_top_terms):
        ax2.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax2.get_yticklabels()[i].set_color('purple')

    # class 3
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("blue", as_cmap=True), s=450, data=class3_enrich, edgecolor='black', linewidth=1, ax=ax3)
    handles, labels = ax3.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax3.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax3.set_title('ASD-Developmentally Delayed', fontsize=28)
    ax3.tick_params(labelsize=18)
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(1.5)
        ax3.spines[axis].set_color('black')
    ax3.set_xlabel('Fold Enrichment', fontsize=24)
    ax3.set_ylabel('')
    for i in range(num_top_terms):
        ax3.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax3.get_yticklabels()[i].set_color('purple')

    # add legend: dark orange = Biological Processes, purple = Molecular functions
    #ax2.text(1.05, 0.2, 'Biological Process', fontsize=19, color='darkorange', transform=ax0.transAxes)
    #ax2.text(1.05, 0.25, 'Molecular Function', fontsize=19, color='purple', transform=ax0.transAxes)

    fig.tight_layout()
    plt.savefig('figures/WES_GO_term_enrichment.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    GO_term_analysis()
