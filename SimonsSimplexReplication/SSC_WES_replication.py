import os
from math import sqrt
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle as rick
import statistics
from scipy.stats import hypergeom, ttest_ind, fisher_exact
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

from WES_Variant_Set_Enrichments import get_gene_sets
from WES_Variant_Set_Enrichments import load_dnvs as load_dnvs_spark


BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
GENES = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v29.comprehensive_gene_annotation.bed', sep='\t', header=0, index_col=None)
LOG_FOLDER = '/mnt/home/alitman/SPARK_genomics/logfiles/slurm-%j.out'
with open('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gene_ensembl_ID_to_name.pkl', 'rb') as f:
        ENSEMBL_TO_GENE_NAME = rick.load(f)


def load_dnvs_ssc():
    file = '/mnt/home/alitman/ceph/SSC_replication/SSC_DNVs_VEP_output_LOFTEE_try2.vcf'
    #file = '/mnt/home/alitman/ceph/SSC_replication/SSC_DNVs_VEP_output_LOFTEE_hg19.vcf'
    #file = '/mnt/home/alitman/ceph/SSC_replication/SSC_DNVs_VEP_output_LOFTEE_with_repeat.vcf' # with repeat regions
    dnvs = pd.read_csv(file, sep='\t', comment='#', header=0, index_col=None)
    dnvs = dnvs[['Uploaded_variation', 'Consequence', 'Gene', 'Extra']]
    dnvs['Consequence'] = dnvs['Consequence'].str.split(',').str[0]
    dnvs = dnvs.rename({'Uploaded_variation': 'id'}, axis='columns')
    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    dnvs['name'] = dnvs['Gene'].map(ensembl_to_gene)
    dnvs = dnvs.dropna(subset=['name'])

    dnvs['id'] = dnvs['id'].replace('/', '_', regex=True)

    # parse Extra column to get the following features:
    # am_class feature
    # LoF feature
    # LoF_flags feature
    dnvs['am_class'] = dnvs['Extra'].str.extract(r'am_class=(.*?);')
    dnvs['am_class'] = dnvs['am_class'].apply(lambda x: 1 if x in ['likely_pathogenic'] else 0)
    dnvs['am_pathogenicity'] = dnvs['Extra'].str.extract(r'am_pathogenicity=([\d.]+)').astype(float)
    dnvs['am_pathogenicity'] = dnvs['am_pathogenicity'].apply(lambda x: 1 if x>=0.9 else 0)
    
    dnvs['LoF'] = dnvs['Extra'].str.extract(r'LoF=(.*?);')
    dnvs['LoF'] = dnvs['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
    dnvs['LoF_flags'] = dnvs['Extra'].str.extract(r'LoF_flags=(.*?);')
    # reformat lof_flags to 0/1 (whether passed flags or not)
    dnvs['LoF_flags'] = dnvs['LoF_flags'].fillna(1)
    dnvs['LoF_flags'] = dnvs['LoF_flags'].apply(lambda x: 1 if x in ['SINGLE_EXON',1] else 0)
    dnvs = dnvs.drop('Extra', axis=1)
    
    var_to_spid, var_to_asd = build_dictionaries()
    
    # annotate each variant with the SPID
    dnvs['spid'] = dnvs['id'].map(var_to_spid)
    dnvs = dnvs.dropna(subset=['spid'])
    print(dnvs['spid'].nunique())
    dnvs['asd'] = dnvs['id'].map(var_to_asd)
    dnvs = dnvs.dropna(subset=['asd'])
    
    dnvs_sibs = dnvs[dnvs['asd'] == 0]
    dnvs_pro = dnvs[dnvs['asd'] == 1]
    print(dnvs_pro['spid'].nunique())
    print(dnvs_sibs['spid'].nunique())

    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final_noimpute.csv', index_col=False, header=0) # no imputation, no bootstrap
    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final_10.csv', index_col=False, header=0) # imputed, bootstrap
    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final.csv', index_col=False, header=0) # imputed, no bootstrap
    gfmm_labels = gfmm_labels.rename(columns={'individual': 'spid'})
    gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
    gfmm_labels['spid'] = gfmm_labels['spid'].str.replace('.', '-')
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))
    dnvs_pro['class'] = dnvs_pro['spid'].map(spid_to_class)
    dnvs_pro = dnvs_pro.dropna(subset=['class'])
    print(dnvs_pro.groupby('class')['spid'].nunique())
    counts = dnvs_pro.groupby('class')['spid'].count() / dnvs_pro.groupby('class')['spid'].nunique()
    print(counts)
    
    # GET PEOPLE WITH NO DNVs
    # get spids in gfmm_labels['spid'] that are not in dnvs_pro - they probably have no dnvs?
    #spids_zero = gfmm_labels[~gfmm_labels['spid'].isin(dnvs_pro['spid'])][['spid', 'mixed_pred']]
    # get sibs with zero dnvs
    #zero_sibs = zero.merge(sibling_list, on='spid').drop('asd', axis=1)

    return dnvs_pro, dnvs_sibs #, zero_pros, zero_sibs


def build_dictionaries():
    variant_file = '/mnt/home/alitman/ceph/SSC_replication/SSC_DNVs_clean.tsv'
    variants = pd.read_csv(variant_file, sep='\t', header=0)
    variants['Pos'] = variants['Pos'].astype(int)
    variants['id'] = variants['Chr'].astype(str) + '_' + variants['Pos'].astype(str) + '_' + variants['Ref'].astype(str) + '_' + variants['Alt'].astype(str)
    # get dictionary mapping variant id to individual id
    var_to_spid = dict(zip(variants['id'], variants['Individual id']))
    # get dictionary mapping variant id to asd
    var_to_asd = dict(zip(variants['id'], variants['Proband']))

    return var_to_spid, var_to_asd


def compute_odds_ratios():
    dnvs_pro, dnvs_sibs = load_dnvs_ssc()
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_benign = ['synonymous_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['synonymous'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
    dnvs_sibs['synonymous'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
    dnvs_pro['missense'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['missense'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_pro['lof'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)

    gene_sets, gene_set_names = get_gene_sets()
    print(gene_set_names)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    satterstrom_gene_set = []
    gene_set_of_interest = ['pli_genes_highest']
    for i in range(len(gene_sets)):
        if gene_set_names[i] in gene_set_of_interest:
            #print(gene_sets[i][:-3])
            satterstrom_gene_set += gene_sets[i][:-3]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    print(len(satterstrom_gene_set))

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() #+ zero_sibs['spid'].nunique()
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['syn_final_consequence'] = dnvs_pro['synonymous']
    dnvs_sibs['syn_final_consequence'] = dnvs_sibs['synonymous']
    dnvs_pro['mis_final_consequence'] = dnvs_pro['missense'] * dnvs_pro['am_class']
    dnvs_sibs['mis_final_consequence'] = dnvs_sibs['missense'] * dnvs_pro['am_class']
    dnvs_pro['lof_final_consequence'] = dnvs_pro['lof'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
    dnvs_sibs['lof_final_consequence'] = dnvs_sibs['lof'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']

    odds_ratios = defaultdict(list)
    class_to_gene_set = {}
    props = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        print(class_id)
        
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(satterstrom_gene_set)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        mis_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['mis_final_consequence'] == 1]
        mis_case_variant_present_count = mis_gene_vars_for_class['spid'].nunique()
        mis_case_variant_absent_count = class_count - mis_case_variant_present_count

        # now siblings
        gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'].isin(satterstrom_gene_set)]
        mis_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['mis_final_consequence'] == 1]
        mis_sibs_case_variant_present_count = mis_gene_vars_sibs['spid'].nunique()
        mis_sibs_case_variant_absent_count = num_sibs - mis_sibs_case_variant_present_count

        # get the counts for lof and sum
        lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['lof_final_consequence'] == 1]
        lof_case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
        lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['lof_final_consequence'] == 1]
        lof_sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique()
        lof_case_variant_absent_count = class_count - lof_case_variant_present_count
        lof_sibs_case_variant_absent_count = num_sibs - lof_sibs_case_variant_present_count

        props.append(mis_case_variant_present_count/class_count)

        # syn
        syn_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['syn_final_consequence'] == 1]
        syn_case_variant_present_count = syn_gene_vars_for_class['spid'].nunique()
        syn_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['syn_final_consequence'] == 1]
        syn_sibs_case_variant_present_count = syn_gene_vars_sibs['spid'].nunique()
        syn_case_variant_absent_count = class_count - syn_case_variant_present_count
        syn_sibs_case_variant_absent_count = num_sibs - syn_sibs_case_variant_present_count
        
        total_case_variant_present_count = mis_case_variant_present_count + lof_case_variant_present_count
        total_case_variant_absent_count = class_count - total_case_variant_present_count
        total_sibs_case_variant_present_count = mis_sibs_case_variant_present_count + lof_sibs_case_variant_present_count
        total_sibs_case_variant_absent_count = num_sibs - total_sibs_case_variant_present_count

        # compute missense odds ratios
        table_mis = [[mis_case_variant_present_count, mis_case_variant_absent_count],
                    [mis_sibs_case_variant_present_count, mis_sibs_case_variant_absent_count]]
        odds_ratio_mis, _ = fisher_exact(table_mis)

        # compute lof, total, and syn odds ratios
        table_lof = [[lof_case_variant_present_count, lof_case_variant_absent_count],
                    [lof_sibs_case_variant_present_count, lof_sibs_case_variant_absent_count]]
        odds_ratio_lof, _ = fisher_exact(table_lof)

        # get fold change for class over sibs for each gene 
        table = [[total_case_variant_present_count, total_case_variant_absent_count],
                    [total_sibs_case_variant_present_count, total_sibs_case_variant_absent_count]]

        # compute odds ratio
        odds_ratio_total, _ = fisher_exact(table)

        table = [[syn_case_variant_present_count, syn_case_variant_absent_count],
                    [syn_sibs_case_variant_present_count, syn_sibs_case_variant_absent_count]]
        odds_ratio_syn, _ = fisher_exact(table)

        odds_ratios['lof'].append(odds_ratio_lof)
        odds_ratios['total'].append(odds_ratio_total)
        odds_ratios['syn'].append(odds_ratio_syn)
        odds_ratios['mis'].append(odds_ratio_mis)

    # plot
    odds_ratios = [odds_ratios['lof'], odds_ratios['total'], odds_ratios['syn']] # odds_ratios['mis'],
    
    # Width of each bar
    bar_width = 0.15

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x positions for each group of bars
    x_positions = range(len(odds_ratios))

    # Plot each group of bars
    colors = ['limegreen', 'violet', 'red', 'blue']
    for i in range(len(odds_ratios[0])):
        # Compute x positions for bars in this group
        x_values = [x + i * bar_width for x in x_positions]
        
        # Extract data for this group
        y_values = [item[i] for item in odds_ratios]
        
        # Plot bars
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    # Set xticks and labels
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels(['dnLoF', 'dnMis or dnLoF', 'dnSyn'], fontsize=22)

    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Odds ratio', fontsize=18)
    ax.set_title(f'{gene_set_of_interest}', fontsize=18)

    # make border bold
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    # add grid behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-')

    # Show plot
    plt.tight_layout()
    # Show the plot
    plt.savefig(f'SSC_WES_figures/WES_SSC_replication_odds_ratios_{gene_set_of_interest}.png', bbox_inches='tight')
    plt.close()

    '''
    # analyze class_to_gene_set
    # print number of genes in each class
    for class_id in [0,1,2]:
        print(class_id)
        print(len(class_to_gene_set[class_id]))
        print(' '.join(class_to_gene_set[class_id]))
    # print intersection between classes 2 and 3
    print(set(class_to_gene_set[0]).intersection(set(class_to_gene_set[2])))

    intersection = set(class_to_gene_set[0]).intersection(set(class_to_gene_set[2]))
    class2_genes = set(class_to_gene_set[0]) - intersection
    class3_genes = set(class_to_gene_set[2]) - intersection
    '''

def make_gene_trend_figure(consequence='lof', fdr=0.1):
    #gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets()
    # read the pickles
    with open('pickles/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('pickles/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('pickles/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('pickles/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)

    dnvs_pro, dnvs_sibs = load_dnvs()
    
    # subset to target Consequences 
    if consequence == 'lof':
        consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    elif consequence == 'missense':
        consequences = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    #num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() #+ zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 #+ num_class3

    celltype_to_enrichment = {}
    class_to_go_enrichment = {}
    validation_subset = pd.DataFrame()
    prop_table = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        if consequence == 'lof':
            dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
            dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        elif consequence == 'missense':
            dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_pathogenicity']
            dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_pathogenicity']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        #class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        all_pros_data = class0 + class1 + class2 #+ class3

        '''
        genes = list(set(dnvs_pro[dnvs_pro[gene_set] == 1]['name'].tolist()))
        # get the number of unique genes in each class with PTVs in gene
        class0_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 0) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        class1_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 1) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        class2_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 2) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        #class3_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 3) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        sibs_genes = len(set(dnvs_sibs[(dnvs_sibs['name'].isin(genes)) & (dnvs_sibs['gene_set&consequence'] > 0)]['name'].tolist()))
        prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': class0_genes/num_class0, 'class1': class1_genes/num_class1, 'class2': class2_genes/num_class2, 'class3': class3_genes/num_class3, 'sibs': sibs_genes/num_sibs}, index=[0])], axis=0)
        '''

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + sibs
        class1_rest_of_sample = class0 + class2 + sibs
        class2_rest_of_sample = class0 + class1 + sibs
        class3_rest_of_sample = class0 + class1 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        #class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        #corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh')[1]
        corrected = [class0_pval, class1_pval, class2_pval, sibs_pval, all_pros_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        #class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[3])
        all_pros_pval = -np.log10(corrected[4])
        # UNCOMMENT TO SKIP INSIGNIFICANT FEATURES:
        #if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]) < -np.log10(fdr):
        #    continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        #class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 2, 'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 3, 'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 1, 'trend': trend}, index=[0])
        #class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 0, 'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': -1, 'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -2, 'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, class0_df, class1_df, class2_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, sibs_fe]

    order_of_variables = ['Principal_excitatory_neuron_down',
                            'Inhibitory_interneuron_MGE_down',
                            'Inhibitory_interneuron_CGE_down',
                            'Glia_down',
                            'Principal_excitatory_neuron_trans_down',
                            'Inhibitory_interneuron_MGE_trans_down',
                            'Inhibitory_interneuron_CGE_trans_down',
                            'Glia_trans_down',
                            'Principal_excitatory_neuron_trans_up',
                            'Inhibitory_interneuron_MGE_trans_up',
                            'Inhibitory_interneuron_CGE_trans_up',
                            'Glia_trans_up',
                            'Principal_excitatory_neuron_up',
                            'Inhibitory_interneuron_MGE_up',
                            'Inhibitory_interneuron_CGE_up',
                            'Glia_up'
                            ]
    # order the dataframe by the order of variables
    validation_subset['variable'] = pd.Categorical(validation_subset['variable'], categories=order_of_variables, ordered=True)
    validation_subset = validation_subset.sort_values(['variable', 'cluster'])
    
    # BUBBLE PLOT
    colors = ['black', 'purple', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o'})
    validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'blue', 1: 'limegreen', 2: 'red'})
    validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'High-ASD/High-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(fdr), 'gray', validation_subset['color'])
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    #sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', hue='color', sizes=(50, 300))
    for _, row in validation_subset.iterrows():
        plt.scatter('trend', row['variable'], c=row['trend_color'], s=300)
        plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*200, c=row['color'])

    # get legend for bubble sizes
    for i in range(5):
        plt.scatter([], [], s=(i+1)*200, c='gray', label=str(i+1))
    plt.legend(scatterpoints=1, labelspacing=1.1, title='Fold Enrichment', title_fontsize=22, fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=35, ha='right')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    if consequence == 'lof':
        plt.savefig(f'SSC_WES_figures/WES_SSC_gene_trends_ptvs_analysis.png', bbox_inches='tight')
    elif consequence == 'missense':
        plt.savefig(f'SSC_WES_figures/WES_SSC_gene_trends_missense_analysis.png', bbox_inches='tight')
    plt.close()


def make_gene_trend_spark_and_ssc(impute=False, fdr=0.05):
    # read the pickles
    with open('pickles/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('pickles/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('pickles/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('pickles/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt' # 1588 sibs paired
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    dnvs_pro, dnvs_sibs = load_dnvs() # SSC variants

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() #+ zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    # LOAD DNVs
    spark_dnvs_pro, spark_dnvs_sibs, spark_zero_pro, spark_zero_sibs = load_dnvs_spark(imputed=impute)
    consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    num_class0_spark = spark_dnvs_pro[spark_dnvs_pro['class'] == 0]['spid'].nunique() + spark_zero_pro[spark_zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1_spark = spark_dnvs_pro[spark_dnvs_pro['class'] == 1]['spid'].nunique() + spark_zero_pro[spark_zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2_spark = spark_dnvs_pro[spark_dnvs_pro['class'] == 2]['spid'].nunique() + spark_zero_pro[spark_zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3_spark = spark_dnvs_pro[spark_dnvs_pro['class'] == 3]['spid'].nunique() + spark_zero_pro[spark_zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs_spark = spark_dnvs_sibs['spid'].nunique() + spark_zero_sibs['spid'].nunique()
    all_spids_spark = num_class0_spark + num_class1_spark + num_class2_spark + num_class3_spark

    # sum up SSC and SPARK
    num_class0 += num_class0_spark
    num_class1 += num_class1_spark
    num_class2 += num_class2_spark
    num_class3 += num_class3_spark
    num_sibs += num_sibs_spark
    all_spids += all_spids_spark
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    spark_dnvs_pro['consequence'] = spark_dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    spark_dnvs_sibs['consequence'] = spark_dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        spark_dnvs_pro[gene_set_names[i]] = spark_dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        spark_dnvs_sibs[gene_set_names[i]] = spark_dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    validation_subset = pd.DataFrame()
    prop_table = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        print(gene_set)
        
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        spark_dnvs_pro['gene_set&consequence'] = spark_dnvs_pro[gene_set] * spark_dnvs_pro['consequence'] * spark_dnvs_pro['LoF'] * spark_dnvs_pro['LoF_flags']
        spark_dnvs_sibs['gene_set&consequence'] = spark_dnvs_sibs[gene_set] * spark_dnvs_sibs['consequence'] * spark_dnvs_sibs['LoF'] * spark_dnvs_sibs['LoF_flags']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0_dnv = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_class0_dnv = spark_dnvs_pro[spark_dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_zero_class0 = spark_zero_pro[spark_zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0_dnv = class0_dnv + spark_class0_dnv + spark_zero_class0
        class1_dnv = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_class1_dnv = spark_dnvs_pro[spark_dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_zero_class1 = spark_zero_pro[spark_zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1_dnv = class1_dnv + spark_class1_dnv + spark_zero_class1
        class2_dnv = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_class2_dnv = spark_dnvs_pro[spark_dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_zero_class2 = spark_zero_pro[spark_zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2_dnv = class2_dnv + spark_class2_dnv + spark_zero_class2
        class3_dnv = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_class3_dnv = spark_dnvs_pro[spark_dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_zero_class3 = spark_zero_pro[spark_zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3_dnv = class3_dnv + spark_class3_dnv + spark_zero_class3
        sibs_dnv = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        spark_sibs_dnv = spark_dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        sibs_dnv = sibs_dnv + spark_sibs_dnv + spark_zero_sibs['count'].astype(int).tolist()

        all_pros_data_dnv = class0_dnv + class1_dnv + class2_dnv + class3_dnv

        # keep track of proportion of individuals with at least 1 PTV in gene set
        #prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': len([i for i in class0 if i > 0])/num_class0, 'class1': len([i for i in class1 if i > 0])/num_class1, 'class2': len([i for i in class2 if i > 0])/num_class2, 'class3': len([i for i in class3 if i > 0])/num_class3, 'sibs': len([i for i in sibs if i > 0])/num_sibs}, index=[0])], axis=0)

        # get pvalue comparing each class to the rest
        class0_rest_of_sample = class1_dnv + class2_dnv + class3_dnv + sibs_dnv
        class1_rest_of_sample = class0_dnv + class2_dnv + class3_dnv + sibs_dnv
        class2_rest_of_sample = class0_dnv + class1_dnv + class3_dnv + sibs_dnv
        class3_rest_of_sample = class0_dnv + class1_dnv + class2_dnv + sibs_dnv
        sibs_rest_of_sample = class0_dnv + class1_dnv + class2_dnv + class3_dnv

        # get pvalue comparing each class to sibs using a t-test
        class0_pval = ttest_ind(class0_dnv, sibs_dnv, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1_dnv, sibs_dnv, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2_dnv, sibs_dnv, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3_dnv, sibs_dnv, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data_dnv, sibs_dnv, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs_dnv, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]

        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        all_pros_pval = -np.log10(corrected[4])
        sibs_pval = -np.log10(corrected[5])
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        #if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]) < -np.log10(fdr):
        #    continue
        
        # get fold enrichment of PTVs for each class
        background_all = (np.sum(class0_dnv) + np.sum(class1_dnv) + np.sum(class2_dnv) + np.sum(class3_dnv) + np.sum(sibs_dnv))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs_dnv)/num_sibs 
        class0_fe = np.log2((np.sum(class0_dnv)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1_dnv)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2_dnv)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3_dnv)/num_class3)/background)
        all_pros_fe = np.log2((np.sum(all_pros_data_dnv)/(num_class0+num_class1+num_class2+num_class3))/background)
        sibs_fe = np.log2((np.sum(sibs_dnv)/num_sibs)/background_all)

        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 2, 'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 3, 'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 1, 'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 0, 'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': -1, 'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -2, 'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, class0_df, class1_df, class2_df, class3_df], axis=0)

    order_of_variables = ['Principal_excitatory_neuron_down',
                            'Inhibitory_interneuron_MGE_down',
                            'Inhibitory_interneuron_CGE_down',
                            'Glia_down',
                            'Principal_excitatory_neuron_trans_down',
                            'Inhibitory_interneuron_MGE_trans_down',
                            'Inhibitory_interneuron_CGE_trans_down',
                            'Glia_trans_down',
                            'Principal_excitatory_neuron_trans_up',
                            'Inhibitory_interneuron_MGE_trans_up',
                            'Inhibitory_interneuron_CGE_trans_up',
                            'Glia_trans_up',
                            'Principal_excitatory_neuron_up',
                            'Inhibitory_interneuron_MGE_up',
                            'Inhibitory_interneuron_CGE_up',
                            'Glia_up'
                            ]
    # order the dataframe by the order of variables
    validation_subset['variable'] = pd.Categorical(validation_subset['variable'], categories=order_of_variables[::-1], ordered=True)
    validation_subset = validation_subset.sort_values(['variable', 'cluster'])

    if impute:
        colors = ['gray', 'purple', 'violet', 'limegreen', 'red', 'blue']
    else:
        colors = ['gray', 'purple', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'violet', 1: 'red', 2: 'green', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'Low-ASD/Low-Delays', 3: 'High-ASD/High-Delays'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'red', 1: 'violet', 2: 'green', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(fdr), 'gray', validation_subset['color'])
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
    
    # find nans in the dataframe
    print(validation_subset)
    exit()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7,8))
    #for _, row in validation_subset.iterrows():
    #    plt.scatter('Gene Trend', row['variable'], s=300, c=row['trend_color'])
    sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', sizes=(250, 500), hue='color', palette=colors, legend=False)
    
    # make legend outside plot
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=35, ha='right')
    plt.ylabel('')
    plt.xlabel('')
    plt.title('dnPTVs SSC + SPARK', fontsize=22)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_SSC_SPARK_ptvs_analysis.png')
    plt.close()


def plot_proportions():
    dnvs_pro, dnvs_sibs = load_dnvs_ssc()
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['lof_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_pro['mis_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['mis_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() #+ zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() #+ zero_sibs['spid'].nunique()
    
    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes', 'satterstrom', 'brain_expressed_genes', 'dd_genes', 'asd_coexpression_networks', 'psd_genes', 'ddg2p']
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['id'].count().tolist()
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['id'].count().tolist()
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['id'].count().tolist()
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['id'].count().tolist()
        sibs = dnvs_sibs.groupby('spid')['id'].count().tolist()
        
        props.append(np.sum(sibs)/num_sibs)
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)
        #props.append(np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))
        
        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))
        #stds.append(np.std(all_pros_data)/np.sqrt(num_class0 + num_class1 + num_class2 + num_class3))
        
        break 
        
    fig, ax = plt.subplots(figsize=(7,5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'red', 'violet', 'limegreen', 'blue']

    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=15)
    plt.xlabel('')
    plt.ylabel('DNMs per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'HighASD/HighDelays', 'LowASD/LowDelays', 'HighASD/LowDelays', 'LowASD/HighDelays'], fontsize=16, rotation=90)
    plt.title('All DNMs - SSC', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('SSC_WES_figures/WES_DNMs_props_scatter.png', bbox_inches='tight')
    plt.close()

    # NOW LOF VARIANTS
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['lof_consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist()
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist()
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist()
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist()
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        
        props.append(np.sum(sibs)/num_sibs)
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)
        #props.append(np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))
        
        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))
        #stds.append(np.std(all_pros_data)/np.sqrt(num_class0 + num_class1 + num_class2 + num_class3))
        break 
        
    fig, ax = plt.subplots(figsize=(7,5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'red', 'violet', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=15)
    plt.xlabel('')
    plt.ylabel('dnLoF per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'HighASD/LowDelays', 'LowASD/LowDelays', 'LowASD/HighDelays', 'red'], fontsize=16, rotation=90)
    plt.title('dnLoF - SSC', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('SSC_WES_figures/WES_LOF_props_scatter.png', bbox_inches='tight')
    plt.close()
    
    # NOW MIS VARIANTS
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['mis_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['mis_consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist()
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist()
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist()
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist()
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist()
        
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
        break 
        
    fig, ax = plt.subplots(figsize=(7,5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'red', 'violet', 'limegreen', 'blue']

    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=15)
    plt.xlabel('')
    plt.ylabel('dnMis per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'HighASD/LowDelays', 'LowASD/LowDelays', 'LowASD/HighDelays', 'red'], fontsize=16, rotation=90)
    plt.title('dnMis - SSC', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('SSC_WES_figures/WES_MIS_props_scatter.png', bbox_inches='tight')
    plt.close()

    # NOW DN LOF+MISSENSE COMBINED PLOT
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['lof_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['lof_consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags'] 
        dnvs_sibs['lof_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        dnvs_pro['mis_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['mis_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['mis_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['mis_consequence'] * dnvs_sibs['am_class']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0_lof = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class0_mis = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)]
        class1_lof = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class1_mis = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)]
        class2_lof = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class2_mis = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)]
        class3_lof = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class3_mis = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)]
        sibs_lof = dnvs_sibs.groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        sibs_mis = dnvs_sibs.groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)]
        all_pros_data = class0 + class1 + class2 + class3
        
        # plot all_pros_data mean and sibs mean
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(4, 5))
        plt.scatter(0.3, np.mean(all_pros_data), color='purple', s=200)
        plt.scatter(0.7, np.mean(sibs), color='black', s=200)
        plt.xticks([0.3, 0.7], ['All probands', 'Siblings'], fontsize=16)
        plt.gca().set_axisbelow(True)
        plt.xlim([0.1,0.9])
        # make borders black
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)
            plt.gca().spines[axis].set_color('black')
        plt.savefig('GFMM_WGS_Analysis_Plots/WES_LOF_all_pros_sibs.png', bbox_inches='tight')
        plt.close()

        print(np.sum(class0))
        print(np.sum(class1))
        print(np.sum(class2))
        print(np.sum(class3))
        print(np.sum(sibs))
        
        props.append(np.sum(sibs)/num_sibs)
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)
        print(props)
        #props.append(np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))
        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))
        #stds.append(np.std(all_pros_data)/np.sqrt(num_class0 + num_class1 + num_class2 + num_class3))

        print('LOF+MIS DNMs')
        print(f"class 0: {stats.ttest_ind(class0, sibs, equal_var=False, alternative='greater')}")
        print(f"class 1: {stats.ttest_ind(class1, sibs, equal_var=False, alternative='greater')}")
        print(f"class 2: {stats.ttest_ind(class2, sibs, equal_var=False, alternative='greater')}")
        print(f"class 3: {stats.ttest_ind(class3, sibs, equal_var=False, alternative='greater')}")
        pvals = []
        pvals.append(stats.ttest_ind(class0, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class1, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class2, sibs, equal_var=False, alternative='greater').pvalue)
        pvals.append(stats.ttest_ind(class3, sibs, equal_var=False, alternative='greater').pvalue)
        # FDR
        pvals = multipletests(pvals, method='fdr_bh')[1]
        pvals = {i: pval for i, pval in enumerate(pvals)}
        break 
        
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'limegreen', 'violet', 'red', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('Deleterious DNMs per offspring', fontsize=16)
    plt.xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('SSC - All deleterious DNMs', fontsize=20)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('SSC_WES_figures/WES_LOF_MIS_props_scatter.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    #plot_proportions(); exit()
    compute_odds_ratios(); exit()
    make_gene_trend_figure(consequence='lof', fdr=0.05)
