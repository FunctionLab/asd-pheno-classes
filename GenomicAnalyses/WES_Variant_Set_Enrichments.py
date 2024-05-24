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
from scipy.stats import hypergeom, ttest_ind, fisher_exact, ranksums, mannwhitneyu
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests


BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
GENES = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v29.comprehensive_gene_annotation.bed', sep='\t', header=0, index_col=None)
LOG_FOLDER = '/mnt/home/alitman/SPARK_genomics/logfiles/slurm-%j.out'
with open('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gene_ensembl_ID_to_name.pkl', 'rb') as f:
        ENSEMBL_TO_GENE_NAME = rick.load(f)


def load_GFMM_labels():
    '''get lists of SPIDs for each GFMM model class.'''
    #model_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv') #6406
    model_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_imputed_genetic_diagnosis_labeled.csv') #6515 (109 more with genetic diagnoses)
    #model_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_genetic_diagnosis_labeled.csv') # 5837, high-accuracy imputation on only CBCL scores
    class0_spids = model_labels[model_labels['mixed_pred'] == 0]['subject_sp_id'].tolist()
    class1_spids = model_labels[model_labels['mixed_pred'] == 1]['subject_sp_id'].tolist()
    class2_spids = model_labels[model_labels['mixed_pred'] == 2]['subject_sp_id'].tolist()
    class3_spids = model_labels[model_labels['mixed_pred'] == 3]['subject_sp_id'].tolist()
    spid_to_class = dict(zip(model_labels['subject_sp_id'], model_labels['mixed_pred']))

    return spid_to_class, class0_spids, class1_spids, class2_spids, class3_spids


def load_AF():
    # load AF for all WES variants
    af = pd.read_csv('/mnt/ceph/SFARI/SPARK/pub/iWES_v2/variants/deepvariant/iWES_v2.deepvariant.pvcf_variants.tsv', sep='\t', header=0, index_col=None)
    af['id'] = af['chrom'].astype(str) + '_' + af['pos'].astype(str) + '_' + af['ref'].astype(str) + '_' + af['alt'].astype(str)
    af = af.drop(['chrom', 'pos', 'ref', 'alt'], axis=1)
    af['af'] = pd.to_numeric(af['af'], errors='coerce')
    return af


def get_genetic_diagnoses():
    clinical_lab_results = pd.read_csv(f'{BASE_PHENO_DIR}/clinical_lab_results-2022-06-03.csv')
    # invert 'snv' column to 'cnv' column
    #clinical_lab_results['cnv'] = clinical_lab_results['snv'].apply(lambda x: True if x is False else False)
    spids = clinical_lab_results['subject_sp_id'].unique().tolist()
    print('number of genetic diagnoses:')
    print(len(spids))

    _, class0, class1, class2, class3 = load_GFMM_labels()

    # get overlapping spids between clinical lab results and LCA data for each class
    class0_spids = list(set(class0).intersection(spids))
    class1_spids = list(set(class1).intersection(spids))
    class2_spids = list(set(class2).intersection(spids))
    class3_spids = list(set(class3).intersection(spids))
    print(f'Number of SPIDs in class 0: {len(class0_spids)}')
    print(f'Number of SPIDs in class 1: {len(class1_spids)}')
    print(f'Number of SPIDs in class 2: {len(class2_spids)}')
    print(f'Number of SPIDs in class 3: {len(class3_spids)}')
    print(f'Number of SPIDs in class 0: {len(class0_spids)/len(class0)}')
    print(f'Number of SPIDs in class 1: {len(class1_spids)/len(class1)}')
    print(f'Number of SPIDs in class 2: {len(class2_spids)/len(class2)}')
    print(f'Number of SPIDs in class 3: {len(class3_spids)/len(class3)}')
    print(f'Total number of SPIDs across all classes: {len(class0_spids)+len(class1_spids)+len(class2_spids)+len(class3_spids)}')
    
    # plot proportions for each class 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([0, 1, 2, 3], [len(class0_spids)/len(class0)*100, len(class1_spids)/len(class1)*100, len(class2_spids)/len(class2)*100, len(class3_spids)/len(class3)*100], color=['blue', 'violet', 'limegreen', 'red'], width=0.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['ID', 'High Function', 'Social+Anxiety', 'Severe'], fontsize=14)
    ax.set_ylabel('Proportion with genetic diagnoses', fontsize=14)
    ax.set_xlabel('Class', fontsize=14)
    ax.set_title('Proportion of probands with genetic diagnoses per class', fontsize=16)
    fig.savefig('GFMM_WGS_Analysis_Plots/genetic_diagnoses_per_class_light.png', bbox_inches='tight')
    
    # plot fold enrichment of genetic diagnoses for each class
    fig, ax = plt.subplots(figsize=(8, 5))
    background = (len(class0_spids)+len(class1_spids)+len(class2_spids)+len(class3_spids))/(len(class0)+len(class1)+len(class2)+len(class3))
    print(background)
    print(len(class0_spids)/len(class0))
    fold_class0 = (len(class0_spids)/len(class0))/background
    fold_class1 = (len(class1_spids)/len(class1))/background
    fold_class2 = (len(class2_spids)/len(class2))/background
    fold_class3 = (len(class3_spids)/len(class3))/background
    print(f'Fold enrichment of genetic diagnoses in class 0: {fold_class0}')
    print(f'Fold enrichment of genetic diagnoses in class 1: {fold_class1}')
    print(f'Fold enrichment of genetic diagnoses in class 2: {fold_class2}')
    print(f'Fold enrichment of genetic diagnoses in class 3: {fold_class3}')
    ax.bar([0, 1, 2, 3], [fold_class0, fold_class1, fold_class2, fold_class3], color=['blue', 'violet', 'limegreen', 'red'])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Low-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'High-ASD/High-Delays'], fontsize=14)
    ax.set_ylabel('Fold enrichment', fontsize=14)
    ax.set_xlabel('')
    ax.set_title('Fold enrichment of genetic diagnoses per class', fontsize=16)
    # line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    fig.savefig('GFMM_WGS_Analysis_Plots/fold_enrichment_genetic_diagnoses_per_class.png', bbox_inches='tight')

    # get number of SPIDs with 'snv' genetic diagnoses per class
    class0_snv = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class0)]['snv'].value_counts()
    class1_snv = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class1)]['snv'].value_counts()
    class2_snv = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class2)]['snv'].value_counts()
    class3_snv = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class3)]['snv'].value_counts()
    print(f'Number of SPIDs with snv genetic diagnoses in class 0: {class0_snv/len(class0)*100}')
    print(f'Number of SPIDs with snv genetic diagnoses in class 1: {class1_snv/len(class1)*100}')
    print(f'Number of SPIDs with snv genetic diagnoses in class 2: {class2_snv/len(class2)*100}')
    print(f'Number of SPIDs with snv genetic diagnoses in class 3: {class3_snv/len(class3)*100}')

    # Calculate fold enrichment for snv and cnv genetic diagnoses
    # class 0
    background = (class0_snv+class1_snv+class2_snv+class3_snv)/(len(class0)+len(class1)+len(class2)+len(class3))
    print(background)
    print(len(class3))
    print(class3_snv/len(class3))
    class0_fe = (class0_snv/len(class0))/background
    class1_fe = (class1_snv/len(class1))/background
    class2_fe = (class2_snv/len(class2))/background
    class3_fe = (class3_snv/len(class3))/background
    print(f'Fold enrichment of snv genetic diagnoses in class 0: {class0_fe}')
    print(f'Fold enrichment of snv genetic diagnoses in class 1: {class1_fe}')
    print(f'Fold enrichment of snv genetic diagnoses in class 2: {class2_fe}')
    print(f'Fold enrichment of snv genetic diagnoses in class 3: {class3_fe}')

    # plot fold enrichment of snv genetic diagnoses for each class
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))
    # plot each class as two bars: snv and non-snv
    ax.bar([0, 1, 3, 4, 6, 7, 9, 10], [class0_fe[True], class0_fe[False], class1_fe[True], class1_fe[False], class2_fe[True], class2_fe[False], class3_fe[True], class3_fe[False]], color=['blue', 'blue', 'violet', 'violet', 'limegreen', 'limegreen', 'red', 'red'], alpha=0.8, width=0.6)
    ax.set_xticks([0, 1, 3, 4, 6, 7, 9, 10])
    ax.set_xticklabels(['SNV', 'non-SNV', 'SNV', 'non-SNV', 'SNV', 'non-SNV', 'SNV', 'non-SNV'], fontsize=11.5)
    ax.set_ylabel('Fold enrichment', fontsize=14)
    ax.set_xlabel('Class', fontsize=14)
    ax.set_title('Fold enrichment of snv genetic diagnoses per class', fontsize=16)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    fig.savefig('GFMM_WGS_Analysis_Plots/fold_enrichment_snv_genetic_diagnoses_per_class.png', bbox_inches='tight')

    # get inheritence_confirmed counts per class
    class0_snv_confirmed = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class0)]['snv_inheritance_confirmed'].value_counts()
    class1_snv_confirmed = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class1)]['snv_inheritance_confirmed'].value_counts()
    class2_snv_confirmed = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class2)]['snv_inheritance_confirmed'].value_counts()
    class3_snv_confirmed = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class3)]['snv_inheritance_confirmed'].value_counts()
    print(f'Number of SPIDs with inheritence_confirmed genetic diagnoses in class 0: {class0_snv_confirmed/len(class0)}')
    print(f'Number of SPIDs with inheritence_confirmed genetic diagnoses in class 1: {class1_snv_confirmed/len(class1)}')
    print(f'Number of SPIDs with inheritence_confirmed genetic diagnoses in class 2: {class2_snv_confirmed/len(class2)}')
    print(f'Number of SPIDs with inheritence_confirmed genetic diagnoses in class 3: {class3_snv_confirmed/len(class3)}')

    background_denovo = (class0_snv_confirmed['De Novo']+class1_snv_confirmed['De Novo']+class2_snv_confirmed['De Novo']+class3_snv_confirmed['De Novo'])/(len(class0)+len(class1)+len(class2)+len(class3))
    class0_denovo = class0_snv_confirmed['De Novo']/len(class0)
    class0_inherited = (class0_snv_confirmed['Not Maternal']+class0_snv_confirmed['Not Paternal']+class0_snv_confirmed['Maternal']+class0_snv_confirmed['Paternal'])/len(class0)
    class0_maternal = class0_snv_confirmed['Maternal']/len(class0)
    class0_paternal = class0_snv_confirmed['Paternal']/len(class0)
    class1_denovo = class1_snv_confirmed['De Novo']/len(class1)
    class1_inherited = (class1_snv_confirmed['Not Maternal'])/len(class1)
    class1_paternal = 0
    class1_maternal = 0
    class2_denovo = class2_snv_confirmed['De Novo']/len(class2)
    class2_inherited = (class2_snv_confirmed['Not Maternal']+class2_snv_confirmed['Paternal'])/len(class2)
    class2_paternal = class2_snv_confirmed['Paternal']/len(class2)
    class2_maternal = 0
    class3_denovo = class3_snv_confirmed['De Novo']/len(class3)
    class3_inherited = (class3_snv_confirmed['Not Maternal']+class3_snv_confirmed['Not Paternal'])/len(class3)
    class3_maternal = class3_snv_confirmed['Not Paternal']/len(class3)
    class3_paternal = 0
    background_inherited = (class0_snv_confirmed['Not Maternal']+class0_snv_confirmed['Not Paternal']+class0_snv_confirmed['Maternal']+class0_snv_confirmed['Paternal']+class1_snv_confirmed['Not Maternal']+class2_snv_confirmed['Not Maternal']+class2_snv_confirmed['Paternal']+class3_snv_confirmed['Not Maternal']+class3_snv_confirmed['Not Paternal'])/(len(class0)+len(class1)+len(class2)+len(class3))
    background_maternal = (class0_snv_confirmed['Maternal']+class3_snv_confirmed['Not Paternal'])/(len(class0)+len(class3))
    background_paternal = (class0_snv_confirmed['Paternal']+class1_snv_confirmed['Not Maternal']+class2_snv_confirmed['Paternal']+class3_snv_confirmed['Not Maternal'])/(len(class0)+len(class1)+len(class2)+len(class3))
    print(background_denovo)
    print(background_inherited)
    # divide by background
    class0_denovo = class0_denovo/background_denovo
    class0_inherited = class0_inherited/background_inherited
    class1_denovo = class1_denovo/background_denovo
    class1_inherited = class1_inherited/background_inherited
    class2_denovo = class2_denovo/background_denovo
    class2_inherited = class2_inherited/background_inherited
    class3_denovo = class3_denovo/background_denovo
    class3_inherited = class3_inherited/background_inherited

    class0_maternal = class0_maternal/background_maternal
    class1_maternal = class1_maternal/background_maternal
    class2_maternal = class2_maternal/background_maternal
    class3_maternal = class3_maternal/background_maternal
    class0_paternal = class0_paternal/background_paternal
    class1_paternal = class1_paternal/background_paternal
    class2_paternal = class2_paternal/background_paternal
    class3_paternal = class3_paternal/background_paternal

    print(f'Fold enrichment of de novo genetic diagnoses in class 0: {class0_denovo}')
    print(f'Fold enrichment of de novo genetic diagnoses in class 1: {class1_denovo}')
    print(f'Fold enrichment of de novo genetic diagnoses in class 2: {class2_denovo}')
    print(f'Fold enrichment of de novo genetic diagnoses in class 3: {class3_denovo}')
    print(f'Fold enrichment of inherited genetic diagnoses in class 0: {class0_inherited}')
    print(f'Fold enrichment of inherited genetic diagnoses in class 1: {class1_inherited}')
    print(f'Fold enrichment of inherited genetic diagnoses in class 2: {class2_inherited}')
    print(f'Fold enrichment of inherited genetic diagnoses in class 3: {class3_inherited}')
    print('MATERNAL/PATERNAL:')
    print(f'Fold enrichment of maternal genetic diagnoses in class 0: {class0_maternal}')
    print(f'Fold enrichment of maternal genetic diagnoses in class 1: {class1_maternal}')
    print(f'Fold enrichment of maternal genetic diagnoses in class 2: {class2_maternal}')
    print(f'Fold enrichment of maternal genetic diagnoses in class 3: {class3_maternal}')
    print(f'Fold enrichment of paternal genetic diagnoses in class 0: {class0_paternal}')
    print(f'Fold enrichment of paternal genetic diagnoses in class 1: {class1_paternal}')
    print(f'Fold enrichment of paternal genetic diagnoses in class 2: {class2_paternal}')
    print(f'Fold enrichment of paternal genetic diagnoses in class 3: {class3_paternal}')
    
    # plot fold enrichment of de novo vs. inherited diagnoses for each class
    fig, ax = plt.subplots(figsize=(8, 5))
    # plot each class as two bars: de novo and inherited
    ax.bar([0, 1, 3, 4, 6, 7, 9, 10], [class0_denovo, class0_inherited, class1_denovo, class1_inherited, class2_denovo, class2_inherited, class3_denovo, class3_inherited], color=['blue', 'blue', 'violet', 'violet', 'limegreen', 'limegreen', 'red', 'red'])
    ax.set_xticks([0, 1, 3, 4, 6, 7, 9, 10])
    ax.set_xticklabels(['de novo', 'inherited', 'de novo', 'inherited', 'de novo', 'inherited', 'de novo', 'inherited'], fontsize=11, rotation=45, ha='right')
    ax.set_ylabel('Fold enrichment', fontsize=14)
    ax.set_xlabel('Class', fontsize=14)
    ax.set_title('Fold enrichment of de novo vs. inherited genetic diagnoses per class', fontsize=16)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    fig.savefig('GFMM_WGS_Analysis_Plots/fold_enrichment_denovo_inherited_genetic_diagnoses_per_class_light.png', bbox_inches='tight')
    plt.close()

    # get top genes in each class
    class0_snv_genes = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class0)]['snv_genetic_status'].value_counts()
    class1_snv_genes = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class1)]['snv_genetic_status'].value_counts()
    class2_snv_genes = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class2)]['snv_genetic_status'].value_counts()
    class3_snv_genes = clinical_lab_results[clinical_lab_results['subject_sp_id'].isin(class3)]['snv_genetic_status'].value_counts()
    print(f'Number of SPIDs with top genes in class 0: {class0_snv_genes/len(class0)}')
    print(f'Number of SPIDs with top genes in class 1: {class1_snv_genes/len(class1)}')
    print(f'Number of SPIDs with top genes in class 2: {class2_snv_genes/len(class2)}')
    print(f'Number of SPIDs with top genes in class 3: {class3_snv_genes/len(class3)}')
    print(class0_snv_genes)
    print(class1_snv_genes)
    print(class2_snv_genes)
    print(class3_snv_genes)


def filter_for_DNVs():
    wes_spids = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes_spids = pd.read_csv(wes_spids, sep='\t')
    wes_spids = wes_spids[['father', 'mother', 'spid']]
    # remove where FID=0 or MID=0
    wes_spids = wes_spids[(wes_spids['father'] != '0') & (wes_spids['mother'] != '0')]
    # save to file
    wes_spids.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/thorough_spark_trios_WES2_cleaned_tab.txt', sep='\t', header=False, index=False)
    wes_spids.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/thorough_spark_trios_WES2_cleaned_comma.txt', sep=',', header=False, index=False)

    deepvar_dir = '/mnt/ceph/SFARI/SPARK/pub/iWES_v2/variants/deepvariant/gvcf/'
    gatk_dir = '/mnt/ceph/SFARI/SPARK/pub/iWES_v2/variants/gatk/gvcf/'
    ids = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/thorough_spark_trios_WES2_cleaned_tab.txt'
    ids = pd.read_csv(ids, sep='\t')
    ids.columns = ['FID', 'MID', 'SPID']
    # check if all SPIDs are in the deepvar dir in the form of {SPID}.gvcf.gz
    for fid, mid, spid in zip(ids['FID'], ids['MID'], ids['SPID']):
        # if spid is not in any of the deepvar dirs, remove it from ids
        # add i to the directory: {deepvar_dir}{i}/{spid}.gvcf.gz
        # and check if it exists in any of those for both deepvar and gatk
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{spid}.gvcf.gz'):
                break
            elif i == 10:
                # remove spid from ids
                ids = ids[ids['SPID'] != spid]
        # do the same for the gatk dir - must exist in both to be considered
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{spid}.gvcf.gz'):
                break
            elif i == 10:
                # remove from ids
                ids = ids[ids['SPID'] != spid]
        # check mid and fid too
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{mid}.gvcf.gz'):
                break
            elif i == 10:
                # remove from ids
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{mid}.gvcf.gz'):
                break
            elif i == 10:
                # remove from ids
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{fid}.gvcf.gz'):
                break
            elif i == 10:
                # remove from ids
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{fid}.gvcf.gz'):
                break
            elif i == 10:
                # remove from ids
                ids = ids[ids['SPID'] != spid]
        
    print(ids.shape)
    # print to file
    ids.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/thorough_spark_trios_WES2_cleaned_tab.txt', sep='\t', header=False, index=False)
    ids.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/thorough_spark_trios_WES2_cleaned_comma.txt', sep=',', header=False, index=False)


def get_paired_sibs():
    file = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes = pd.read_csv(file, sep='\t')
    # subset wes to only include siblings
    sibs = wes[wes['asd'] == 1]
    spids_for_model = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0) # 5280 PROBANDS
    probands = spids_for_model.index.tolist()
    # get SPIDs for siblings who have the same FID/MID as the proband
    sibling_spids = []
    for i, row in wes.iterrows():
        if row['spid'] in probands:
            # get FID and MID
            fid = row['father']
            mid = row['mother']
            # get all siblings with that FID/MID
            if fid == '0' and mid == '0':
                continue
            if fid == '0':
                siblings = sibs[sibs['mother'] == mid]['spid'].tolist()
            elif mid == '0':
                siblings = sibs[sibs['father'] == fid]['spid'].tolist()
            else:
                siblings = sibs[(sibs['father'] == fid) & (sibs['mother'] == mid)]['spid'].tolist()
            # add to list of siblings
            sibling_spids.extend(siblings)
    print(len(sibling_spids))
    # remove duplicates
    sibling_spids = list(set(sibling_spids))
    print(len(sibling_spids))
    # save to file - either 4700 (not imputed) or 6400 (imputed)
    with open('/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt', 'w') as f:
        for item in sibling_spids:
            f.write("%s\n" % item)


def process_DNVs():
    dir = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/output/'
    # for every subdir in dir, check if the following file exists: {subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf
    # if it does, then get the number of DNVs in that file
    # if it doesn't, then skip and print("missing!")
    # get all subdirs
    subdirs = os.listdir(dir)
    # get all SPIDs
    var_to_spid = defaultdict(list) # dictionary with variant ID as key and list of SPIDs as value
    SPID_to_vars = defaultdict(list) # dictionary with SPID as key and list of variant IDs as value
    spid_to_count = defaultdict(int) # dictionary with SPID as key and number of DNVs as value
    spids = []
    missing = 0
    for subdir in subdirs:
        # check if the file exists
        if os.path.exists(f'{dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf'):
            # get number of DNVs
            try:
                dnv = pd.read_csv(f'{dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf', sep='\t', comment='#', header=None)
                # populate var_to_spid and SPID_to_vars
                for i, row in dnv.iterrows():
                    # get variant ID
                    var_id = row[2]
                    # get SPID
                    spid = str(subdir)
                    # add to var_to_spid
                    var_to_spid[var_id].append(spid)
                    # add to SPID_to_vars
                    SPID_to_vars[spid].append(var_id)
                    # add to spid_to_count
                    spid_to_count[spid] += 1
            except pd.errors.EmptyDataError:
                spid_to_count[subdir] = 0                
        else:
            print(f'{subdir} missing!')
            missing += 1
    print(f'Number of missing SPIDs: {missing}')

    # get mean+3SD of counts
    counts = []
    for spid, count in spid_to_count.items():
        counts.append(count)
    mean = np.mean(counts)
    sd = np.std(counts)
    print(f'Mean: {mean}')
    print(f'SD: {sd}')
    # get threshold
    threshold = mean + 3*sd
    # FILTER: remove SPIDs with more than 3SD DNVs
    spid_to_count = {k: v for k, v in spid_to_count.items() if v <= threshold}
    SPID_to_vars = {k: v for k, v in SPID_to_vars.items() if k in spid_to_count.keys()}

    # iterate through SPID_to_vars and remove variants that are not unique to that SPID (non-singletons)
    for spid, vars in SPID_to_vars.items():
        # get all SPIDs for each variant
        for var in vars:
            spids = var_to_spid[var]
            # if there is more than one SPID, remove the variant from SPID_to_vars
            if len(spids) > 1:
                SPID_to_vars[spid].remove(var)
    # update counts in spid_to_count
    for spid, vars in SPID_to_vars.items():
        spid_to_count[spid] = len(vars)

    # update var_to_spid to only include singletons
    var_to_spid = {}
    for spid, vars in SPID_to_vars.items():
        for var in vars:
            var_to_spid[var] = spid
    
    # convert spid_to_count to a dataframe
    spid_to_count = pd.DataFrame.from_dict(spid_to_count, orient='index')
    spid_to_count.columns = ['count']
    spid_to_count.index.name = 'SPID'
    spid_to_count = spid_to_count.reset_index()
    # save filtered df to file
    spid_to_count.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_DNV_count.txt', sep='\t', index=False)

    # save dictionary of filtered var to spid to pickle
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/var_to_spid.pkl', 'wb') as handle:
        rick.dump(var_to_spid, handle, protocol=rick.HIGHEST_PROTOCOL)
    
    # save SPID_to_vars to file
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_vars.pkl', 'wb') as f:
        rick.dump(SPID_to_vars, f, rick.HIGHEST_PROTOCOL)


def load_dnvs(imputed=False):
    #file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/VEP_most_severe_consequence_DNV_calls_WES_v2.vcf'
    #file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/VEP_most_severe_consequence_DNV_calls_filtered_WES_v2.vcf' # filtered out centromeres and repeats
    #file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/VEP_most_severe_consequence_LOFTEE_DNV_calls_WES_v2.vcf' # LOFTEE + ALPHAMISSENSE PREDICTIONS
    file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/VEP_most_severe_consequence_LOFTEE_DNV_calls_filtered_WES_v2.vcf' # filtered out centromeres and repeats
    dnvs = pd.read_csv(file, sep='\t', comment='#', header=0, index_col=None)
    dnvs = dnvs[['Uploaded_variation', 'Consequence', 'Gene', 'Extra']]
    dnvs['Consequence'] = dnvs['Consequence'].str.split(',').str[0]
    dnvs = dnvs.rename({'Uploaded_variation': 'id'}, axis='columns')
    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    dnvs['name'] = dnvs['Gene'].map(ensembl_to_gene)
    dnvs = dnvs.dropna(subset=['name'])

    # parse Extra column to get the following features:
    # am_class feature
    # LoF feature
    # LoF_flags feature
    dnvs['am_class'] = dnvs['Extra'].str.extract(r'am_class=(.*?);')
    dnvs['am_class'] = dnvs['am_class'].apply(lambda x: 1 if x in ['likely_pathogenic'] else 0)
    dnvs['am_pathogenicity'] = dnvs['Extra'].str.extract(r'am_pathogenicity=([\d.]+)').astype(float)
    dnvs['am_pathogenicity'] = dnvs['am_pathogenicity'].apply(lambda x: 1 if x>=0.9 else 0)
    #print(dnvs['am_pathogenicity'].value_counts()); exit()
    dnvs['LoF'] = dnvs['Extra'].str.extract(r'LoF=(.*?);')
    dnvs['LoF'] = dnvs['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
    dnvs['LoF_flags'] = dnvs['Extra'].str.extract(r'LoF_flags=(.*?);')
    # reformat lof_flags to 0/1 (whether passed flags or not)
    dnvs['LoF_flags'] = dnvs['LoF_flags'].fillna(1)
    dnvs['LoF_flags'] = dnvs['LoF_flags'].apply(lambda x: 1 if x in ['SINGLE_EXON',1] else 0)
    dnvs = dnvs.drop('Extra', axis=1)

    # load in SPID_to_vars
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_vars.pkl', 'rb') as f:
        SPID_to_vars = rick.load(f)
    # load var_to_spid
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/var_to_spid.pkl', 'rb') as handle:
        var_to_spid = rick.load(handle)
    
    # annotate each variant with the SPID
    dnvs['spid'] = dnvs['id'].map(var_to_spid)
    dnvs = dnvs.dropna(subset=['spid'])

    # read in master table
    master = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    master = pd.read_csv(master, sep='\t')
    master = master[['spid', 'asd']]
    # get dictionary mapping SPID to ASD status
    spid_to_asd = dict(zip(master['spid'], master['asd']))
    # annotate spids with ASD status
    dnvs['asd'] = dnvs['spid'].map(spid_to_asd)
    dnvs = dnvs.dropna(subset=['asd'])

    '''
    dnvs = dnvs[dnvs['name'] == 'PTEN']
    dnvs = dnvs[dnvs['LoF'] == 1].reset_index()
    dnvs = dnvs.drop('index', axis=1)
    print(dnvs)
    bms = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12/basic_medical_screening_2022-12-12.csv', index_col=False)
    # rename subject_sp_id to spid
    bms = bms.rename(columns={'subject_sp_id': 'spid'})
    # intersect dnvs with bms
    #dnvs = dnvs.merge(bms[['spid', 'growth_macroceph']], how='inner', on='spid')
    spid_to_macroceph = dict(zip(bms['spid'], bms['growth_macroceph']))
    # fillna growth_macroceph
    dnvs['growth_macroceph'] = dnvs['spid'].map(spid_to_macroceph)
    print(dnvs['growth_macroceph'])
    print(dnvs); exit()
    '''
    
    dnvs_sibs = dnvs[dnvs['asd'] == 1]
    dnvs_pro = dnvs[dnvs['asd'] == 2]

    # label pros with GFMM labels
    if imputed:
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_6406_imputed_cohort_GFMM_labeled.csv', index_col=False, header=0) # 6406 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_imputed_genetic_diagnosis_labeled.csv', index_col=False, header=0) # 6550 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_genetic_diagnosis_labeled.csv', index_col=False, header=0) # 5837 probands, cbcl scores only
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_no_bms_5721_imputed_labeled.csv', index_col=False, header=0) # 5721, impute all cbcl
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_no_bms_5727_imputed_labeled.csv', index_col=False, header=0) # 5727, impute all cbcl
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_no_bms_5714_labeled.csv', index_col=False, header=0) # 5714, cbcl scores only
        # rename 'subject_sp_id' to 'spid'
        gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
        gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
        # create a dictionary mapping spid to class
        spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))
        # annotate dnvs_pro with class
        dnvs_pro['class'] = dnvs_pro['spid'].map(spid_to_class)
        dnvs_pro = dnvs_pro.dropna(subset=['class'])
        # print counts of spids in each class
        #print(dnvs_pro.groupby('class')['spid'].nunique())
    else:
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_recode_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5282 probands
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        # rename 'subject_sp_id' to 'spid'
        gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
        gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
        # create a dictionary mapping spid to class
        spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))
        # annotate dnvs_pro with class
        dnvs_pro['class'] = dnvs_pro['spid'].map(spid_to_class)
        # dropna
        dnvs_pro = dnvs_pro.dropna(subset=['class'])

    # subset dnv_sibs to paired sibs
    if imputed:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt' # 2027 sibs
        sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
        sibling_list.columns = ['spid']
        dnvs_sibs = pd.merge(dnvs_sibs, sibling_list, how='inner', on='spid') 
        # print number of sibs
        #print(dnvs_sibs['spid'].nunique())
    else:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
        #sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5282_siblings_spids.txt'
        #sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt' # 1588 sibs
        sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
        sibling_list.columns = ['spid']
        dnvs_sibs = pd.merge(dnvs_sibs, sibling_list, how='inner', on='spid')
    
    #counts = dnvs_pro.groupby('spid')['class'].count()
    #print(counts.mean())

    # print mean number of dnvs per spid in dnvs_pro and dnvs_sibs
    print(dnvs_pro['spid'].nunique())
    print(dnvs_pro.groupby('spid')['id'].count().mean())
    print(dnvs_pro.groupby('spid')['id'].count().std())
    print(dnvs_sibs['spid'].nunique())
    print(dnvs_sibs.groupby('spid')['id'].count().mean())
    print(dnvs_sibs.groupby('spid')['id'].count().std())

    # MERGE WITH PEOPLE WITH NO DNVs
    count_file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_DNV_count.txt'
    counts = pd.read_csv(count_file, sep='\t', index_col=False)
    # rename 'SPID' to 'spid'
    counts = counts.rename(columns={'SPID': 'spid'})
    # get people with 0 DNVs
    zero = counts[counts['count'] == 0]
    # annotate zero with asd status
    zero = zero.merge(master[['spid', 'asd']], on='spid')
    # annotate with class
    zero_pros = zero.merge(gfmm_labels[['spid', 'mixed_pred']], on='spid').drop('asd', axis=1)
    zero_sibs = zero.merge(sibling_list, on='spid').drop('asd', axis=1)
    # print nuumber of unique spids in dnvs_pro, zero_pros, dnvs_sibs, zero_sibs
    print(dnvs_pro['spid'].nunique())
    print(zero_pros['spid'].nunique())
    print(dnvs_sibs['spid'].nunique())
    print(zero_sibs['spid'].nunique())

    return dnvs_pro, dnvs_sibs, zero_pros, zero_sibs


def get_gene_sets():

    sfari_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/SFARI_genes.csv', header=0, index_col=False).rename({'gene-symbol': 'name'}, axis='columns')
    lof_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/Constrained_PLIScoreOver0.9.bed', sep='\t', index_col=None)
    chd8_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/CHD8_targets_Cotney2015_Sugathan2014.bed', sep='\t', index_col=None)
    fmrp_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/FMRP_targets_Darnell2011.bed', sep='\t', index_col=None)
    dd = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/Developmental_delay_DDD.bed', sep='\t', index_col=None)
    asd_risk_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/ASD_risk_genes_TADA_FDR0.3.bed', sep='\t', index_col=None)
    haplo_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/haploinsufficiency_hesc_2022_ST.csv', header=0, index_col=False).rename({'Symbol': 'name'}, axis='columns')
    brain_expressed_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/BrainExpressed_Kang2011.bed', sep='\t', index_col=None)
    antisense_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/Antisense_GencodeV19.bed', sep='\t', index_col=None)
    asd_coexpression_networks = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/ASD_coexpression_networks_Willsey2013.bed', sep='\t', index_col=None)
    linc_rna_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/lincRNA_GencodeV19.bed', sep='\t', index_col=None)
    psd_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/PSD_Genes2Cognition.bed', sep='\t', index_col=None)
    satterstrom = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/satterstrom_2020_102_ASD_genes.csv', header=0, index_col=False).rename({'gene': 'name'}, axis='columns')
    ddg2p = pd.read_csv('/mnt/home/alitman/ceph/Marker_Genes/DDG2P.csv', header=0, index_col=False).rename({'gene symbol': 'name'}, axis='columns')
    sfari_genes1 = list(sfari_genes[sfari_genes['gene-score'] == 1]['name'])
    sfari_genes2 = list(sfari_genes[sfari_genes['gene-score'] == 2]['name'])
    sfari_syndromic = list(sfari_genes[sfari_genes['syndromic'] == 1]['name'])
    sfari_genes = list(sfari_genes['name'])
    lof_genes = list(lof_genes['name'])
    chd8_genes = list(chd8_genes['name'])
    fmrp_genes = list(fmrp_genes['name'])
    dd_genes = list(dd['name'])
    asd_risk_genes = list(asd_risk_genes['name'])
    haplo_genes = list(haplo_genes['name'])
    brain_expressed_genes = list(brain_expressed_genes['name'])
    asd_coexpression_networks = list(asd_coexpression_networks['name'])
    psd_genes = list(psd_genes['name'])
    satterstrom = list(satterstrom['name'])
    ddg2p = list(ddg2p['name'])
    #all_genes = list(set(pro_vars['name'].unique().tolist() + sib_vars['name'].unique().tolist()))
    all_genes = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v29.annotation.protein_coding_genes.hg38.bed', sep='\t', index_col=None, header=None)
    all_genes.columns = ['chr', 'start', 'end', 'gene', 'name', 'strand']
    all_genes = list(all_genes['name'])

    # liver-expression genes
    liver_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/liver_genes.txt', header=None)
    liver_genes = liver_genes[0].tolist()

    # get pLI scores for genes
    pli_table = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/pLI_table.txt', sep='\t', index_col=False)
    # get genes with pLI >= 0.995
    pli_genes_highest = pli_table[pli_table['pLI'] >= 0.995]['gene'].tolist()
    # get genes with pli 0.5-0.995
    pli_genes_high = pli_table[(pli_table['pLI'] < 0.995) & (pli_table['pLI'] >= 0.5)]['gene'].tolist()

    gene_list = [all_genes, lof_genes, chd8_genes, fmrp_genes, dd_genes, asd_risk_genes, haplo_genes, brain_expressed_genes, asd_coexpression_networks, psd_genes, satterstrom, sfari_genes, sfari_genes1, sfari_genes2, sfari_syndromic, pli_genes_highest, pli_genes_high, ddg2p, liver_genes]
    gene_list_names = ['all_genes', 'lof_genes', 'chd8_genes', 'fmrp_genes', 'dd_genes', 'asd_risk_genes', 'haplo_genes', 'brain_expressed_genes', 'asd_coexpression_networks', 'psd_genes', 'satterstrom', 'sfari_genes', 'sfari_genes1', 'sfari_genes2', 'sfari_syndromic', 'pli_genes_highest', 'pli_genes_medium', 'ddg2p', 'liver_genes']

    return gene_list, gene_list_names


def volcano_lof(impute=False, all_pros=False):
    '''Produce volcano plot for LOF variants (PTVs)'''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # FAST EXTRACT GENES
    # subset to sfari_genes1
    #dnvs_pro = dnvs_pro[dnvs_pro['name'].isin(gene_sets[11])]
    #num_class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('name')['consequence'].sum()
    # print list of top 20 gene names with most PTVs
    #print(num_class3.sort_values(ascending=False).head(20)[0:20].index.tolist())

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['violet', 'red', 'limegreen', 'blue']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes']
    shapes = []
    potential_shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    potential_shapes = potential_shapes[0:len(gene_sets_to_keep)]

    props = []
    # for each class, sum the number of PTVs per individual and create list of counts
    for gene_set in gene_sets_to_keep:
        shape = potential_shapes.pop(0)
        if all_pros:
            shapes += [shape, shape, shape, shape, shape]
        else:
            shapes += [shape, shape, shape, shape]
        dnvs_pro[f'{gene_set}&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags'] # only keep high confidence LoF variants
        dnvs_sibs[f'{gene_set}&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        # add zero_pro to class0 to account for probands in class 0 with no DNVs
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist() # these probands have no DNVs -> therefore no dnPTVs either
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')[f'{gene_set}&consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        #print(np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))
        props.append(np.sum(class0)/num_class0)
        props.append(np.sum(class1)/num_class1)
        props.append(np.sum(class2)/num_class2)
        props.append(np.sum(class3)/num_class3)
        props.append(np.sum(sibs)/num_sibs)

        # get pvalue comparing each class to the rest of the sample using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3))/(num_class0 + num_class1 + num_class2 + num_class3)
        background = np.sum(sibs)/(num_sibs)

        # get two pvalues - one with 'greater' alternative and one with 'less' alternative and take max of -log10(pvalue)
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])
        # FDR CORRECTION
        # multiple testing correction
        if all_pros:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
            all_pros_pval = -np.log10(corrected[4])
            #sibs_pval = -np.log10(corrected[5])
        else:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
            print(corrected)
            #sibs_pval = -np.log10(corrected[4])
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        all_pros_fe = np.log2((np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

        print([class0_fe, class1_fe, class2_fe, class3_fe])
        
        # append fold enrichment and pvalue to lists
        if all_pros:
            FE += [class0_fe, class1_fe, class2_fe, class3_fe, all_pros_fe] 
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval]
        else:
            FE += [class0_fe, class1_fe, class2_fe, class3_fe] # sibs_fe
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval] # sibs_pval
        # append colors to list
        # append gray if p_value < 0.05, else append class color (orange for sibs)
        thresh = -np.log10(0.05)
        if class0_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[0])
            #colors.append('pink')
        else:
            #if class0_pval > 3.5:
            #    tick_labels.append(gene_set)
            #else:
            tick_labels.append('')
            print('class0')
            print(class0_pval)
            print(gene_set)
            colors.append(ref_colors[0])
        if class1_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[1])
            #colors.append('violet')
        else:
            if class1_fe > 3.8:
                tick_labels.append('')
            else:
                tick_labels.append('')
            print('class1')
            print(class1_pval)
            print(gene_set)
            colors.append(ref_colors[1])
        if class2_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[2])
            #colors.append('lightlimegreen')
        else:
            # if FE > 4, add tick
            if class2_fe > 4:
                tick_labels.append('')
            else:
                tick_labels.append('')
            print('class2')
            print(class2_pval)
            print(gene_set)
            colors.append(ref_colors[2])
        if class3_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[3])
            #colors.append('lightblue')
        else:
            if class3_fe > 4:
                tick_labels.append('')
            else:
                tick_labels.append('')
            print('class3')
            print(class3_pval)
            print(gene_set)
            colors.append(ref_colors[3])
        if all_pros:
            if all_pros_pval < thresh:
                tick_labels.append('')
                colors.append('purple')
            else:
                print('all_pros')
                print(all_pros_pval)
                print(gene_set)
                colors.append('purple')
        #if sibs_pval < thresh:
        #    tick_labels.append('')
        #    colors.append('black')
        #else:
        #    tick_labels.append(gene_set)
        #    colors.append(ref_colors[4])

    # plot volcano plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All genes', markerfacecolor='gray', markersize=10),
                          Line2D([0], [0], marker='v', color='w', label='LoF-Intolerant', markerfacecolor='gray', markersize=10),
                            Line2D([0], [0], marker='p', color='w', label='FMRP targets', markerfacecolor='gray', markersize=10),
                                Line2D([0], [0], marker='^', color='w', label='ASD risk', markerfacecolor='gray', markersize=10),
                                    Line2D([0], [0], marker='d', color='w', label='SFARI', markerfacecolor='gray', markersize=10),
                                        Line2D([0], [0], marker='P', color='w', label='Satterstrom', markerfacecolor='gray', markersize=10),
                                            Line2D([0], [0], marker='s', color='w', label='Brain-expressed', markerfacecolor='gray', markersize=10),
                                                Line2D([0], [0], marker='>', color='w', label='DDG2P', markerfacecolor='gray', markersize=10),
                                                    #Line2D([0], [0], marker='*', color='w', label='ASD coexpression', markerfacecolor='gray', markersize=10),
                                                        #Line2D([0], [0], marker='X', color='w', label='PSD', markerfacecolor='gray', markersize=10),
                                                            #Line2D([0], [0], marker='D', color='w', label='DDG2P', markerfacecolor='gray', markersize=10),
                                                            Line2D([0], [0], marker='o', color='w', label='High-ASD/High-Delays', markerfacecolor='red', markersize=10),
                                                                Line2D([0], [0], marker='o', color='w', label='Low-ASD/Low-Delays', markerfacecolor='violet', markersize=10),
                                                                    Line2D([0], [0], marker='o', color='w', label='High-ASD/Low-Delays', markerfacecolor='limegreen', markersize=10),
                                                                        Line2D([0], [0], marker='o', color='w', label='Low-ASD/High-Delays', markerfacecolor='blue', markersize=10)]
                                                                            #Line2D([0], [0], marker='o', color='w', label='All probands', markerfacecolor='purple', markersize=10),
                                                                                #Line2D([0], [0], marker='o', color='w', label='Siblings', markerfacecolor='black', markersize=10)]
    #ax.legend(handles=legend_elements, loc='upper left', fontsize=16, bbox_to_anchor=(1, 1))
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('dnLoF', fontsize=18)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=1, linewidth=1)
    #for i, txt in enumerate(tick_labels):
    #    ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_6400_volcano_plot_DNV_PTVs.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_volcano_plot_DNV_PTVs.png', bbox_inches='tight')
    plt.close()


def plot_proportions(impute=False):
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
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

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes', 'satterstrom', 'brain_expressed_genes', 'dd_genes']
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        #dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_pathogenicity'] # only keep likely pathogenic missense variants
        #dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_pathogenicity']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['id'].count().tolist()
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['id'].count().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['id'].count().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['id'].count().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['id'].count().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist() #+ [1] # add pseudocount to avoid division by zero
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
        plt.savefig('GFMM_WGS_Analysis_Plots/WES_DNMs_all_pros_sibs.png', bbox_inches='tight')
        plt.close()
        
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
        
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('\x1B[3mDe novo\x1B[0m variants per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('All \x1B[de novo\x1B[0m variants', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_DNMs_props_scatter.png', bbox_inches='tight')
    plt.close()

    # NOW LOF VARIANTS
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['lof_consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags'] 
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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
        sibs = sibs + zero_sibs['count'].astype(int).tolist() + [1] # add pseudocount to avoid division by zero
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
        
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('dnLoF per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('dnLoF', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_LOF_props_scatter.png', bbox_inches='tight')
    plt.close()
    
    # NOW MIS VARIANTS
    props = []
    stds = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['mis_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['mis_consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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
        sibs = sibs + zero_sibs['count'].astype(int).tolist() + [1] # add pseudocount to avoid division by zero
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
        plt.savefig('GFMM_WGS_Analysis_Plots/WES_MIS_all_pros_sibs.png', bbox_inches='tight')
        plt.close()
        
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
        
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('dnMis per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('dnMis', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_MIS_props_scatter.png', bbox_inches='tight')
    plt.close()

    # now inherited LOF
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited_noaf.pkl', 'rb') as f:
    #    spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited.pkl', 'rb') as f: # _90patho
        spid_to_num_missense = rick.load(f)

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' 
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in spid_to_class}
    print(len(pros_to_num_missense))
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in sibling_list}
    print(len(sibs_to_num_missense))

    gene_sets, gene_set_names = get_gene_sets()

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    gene_set_to_index = {gene_set: i for i, gene_set in enumerate(gene_set_names)}
    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes', 'satterstrom', 'brain_expressed_genes', 'dd_genes', 'asd_coexpression_networks', 'psd_genes']
    indices = [gene_set_to_index[gene_set] for gene_set in gene_sets_to_keep]
    
    props = []
    stds = []
    for i in indices:
        gene_set = gene_set_names[i] # get name of gene set
        # get number of PTVs for each spid in gene set
        class0 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        all_pros_data = class0 + class1 + class2 + class3
        sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]
        
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
        
        print("INH LOF")
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
        print(f'CORRECTED: {pvals}')
        break
    
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('inhLoF per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('inhLoF', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_LOF_INH_props_scatter.png', bbox_inches='tight')
    plt.close()

    # NOW INHERITED MIS
    props = []
    stds = []
    for i in indices:
        gene_set = gene_set_names[i] # get name of gene set
        # get number of PTVs for each spid in gene set
        class0 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 3]
        all_pros_data = class0 + class1 + class2 + class3
        sibs = [v[i] for k, v in sibs_to_num_missense.items()]
        
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
        
        print("INH MIS")
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
        print(f'CORRECTED: {pvals}')
        break
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
    for i in range(len(x_values)):
        plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    plt.xlabel('')
    plt.ylabel('inhMis per offspring', fontsize=16)
    ax.set_xticks(x_values)
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    plt.title('inhMis', fontsize=18)
    ax.set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_MIS_INH_props_scatter.png', bbox_inches='tight')
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
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)] + zero_class0
        class1_lof = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class1_mis = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)] + zero_class1
        class2_lof = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class2_mis = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)] + zero_class2
        class3_lof = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class3_mis = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)] + zero_class3
        sibs_lof = dnvs_sibs.groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        sibs_mis = dnvs_sibs.groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)] + zero_sibs['count'].astype(int).tolist()
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
        print(pvals)
        break 
        
    fig, ax = plt.subplots(1,2,figsize=(11,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    for i in range(len(x_values)):
        ax[0].errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Count per offspring', fontsize=16)
    ax[0].set_xticks(x_values)
    ax[0].tick_params(labelsize=16, axis='y')
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    ax[0].set_title('High-impact de novo variants', fontsize=21)
    ax[0].set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax[0].spines[axis].set_linewidth(1.5)
        ax[0].spines[axis].set_color('black')
    ax[0].grid(color='gray', linestyle='-', linewidth=0.5)

    # ADD SIGNIFICANCE TO PLOT
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx+1]
        se_value = stds[grpidx+1]
        ypos = y_position + se_value - 0.001
        if p_value < 0.01:
            ax[0].annotate('***', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.05:
            ax[0].annotate('**', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.1:
            ax[0].annotate('*', xy=(x_position, ypos), ha='center', size=20)
    
    # COMBINED INH LOF + MIS
    props = []
    stds = []
    for i in indices:
        gene_set = gene_set_names[i] # get name of gene set
        # get number of PTVs for each spid in gene set
        class0_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class0_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 0]
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)]
        class1_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class1_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 1]
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)]
        class2_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class2_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 2]
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)]
        class3_lof = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        class3_mis = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 3]
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)]
        all_pros_data = class0 + class1 + class2 + class3
        sibs_lof = [v[i] for k, v in sibs_to_num_ptvs.items()]
        sibs_mis = [v[i] for k, v in sibs_to_num_missense.items()]
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)]

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

        # HYPOTHESIS TESTING of class against sibs
        print("LOF+MIS INH")
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
        # make pvals a dictionary mapping index to pval
        pvals = {i: pval for i, pval in enumerate(pvals)}
        print(pvals)
        break
    
    #fig, ax = plt.subplots(figsize=(5.5,4.5))
    # plot props as scatter
    x_values = np.arange(len(props))
    y_values = props
    colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']

    #plt.scatter(x_values, y_values, color=colors, s=200)
    # plot error bars
    # customize the colors to 'colors' list
    print(y_values)
    for i in range(len(x_values)):
        ax[1].errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Count per offspring', fontsize=16)
    ax[1].set_xticks(x_values)
    ax[1].tick_params(labelsize=16, axis='y')
    #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
    ax[1].set_title('High-impact rare inherited variants', fontsize=21)
    ax[1].set_axisbelow(True)
    # make border dark
    for axis in ['top','bottom','left','right']:
        ax[1].spines[axis].set_linewidth(1.5)
        ax[1].spines[axis].set_color('black')
    ax[1].grid(color='gray', linestyle='-', linewidth=0.5)

    # ADD SIGNIFICANCE TO PLOT
    for grpidx in [0,1,2,3]:
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx+1]
        se_value = stds[grpidx+1]
        ypos = y_position + se_value-0.05
        print(p_value, x_position, ypos)
        if p_value < 0.01:
            ax[1].annotate('***', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.05:
            ax[1].annotate('**', xy=(x_position, ypos), ha='center', size=20)
        elif p_value < 0.1:
            ax[1].annotate('*', xy=(x_position, ypos), ha='center', size=20)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_LOF_COMBINED_MIS_props_scatter.png', bbox_inches='tight')
    plt.close()


def volcano_missense(impute=False, all_pros=False):
    '''Produce volcano plot for dnMis variants'''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    gene_sets, gene_set_names = get_gene_sets()

    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # FAST EXTRACT GENES
    # subset to asd_risk_genes
    #dnvs_pro = dnvs_pro[dnvs_pro['name'].isin(gene_sets[4])]
    #num_class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('name')['consequence'].sum()
    # print list of top 20 gene names with most missense variants
    #print(num_class0.sort_values(ascending=False).head(20))
    #print(num_class0.sort_values(ascending=False).head(20)[0:20].index.tolist()); exit()

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes', 'dd_genes']
    shapes = []
    potential_shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    potential_shapes = potential_shapes[0:len(gene_sets_to_keep)]
    props = []
    for gene_set in gene_sets_to_keep:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class'] # only keep likely pathogenic missense variants
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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
        sibs = sibs + zero_sibs['count'].astype(int).tolist() + [1] # add pseudocount to avoid division by zero
        all_pros_data = class0 + class1 + class2 + class3
        
        # get pvalue comparing each class to the rest of the sample using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3
        total = class0 + class1 + class2 + class3 + sibs
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3))/(num_class0 + num_class1 + num_class2 + num_class3)
        background = np.sum(sibs)/(num_sibs)

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]

        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])

        # FDR CORRECTION
        if all_pros:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
            all_pros_pval = -np.log10(corrected[4])
            #sibs_pval = -np.log10(corrected[5])
        else:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
            print(corrected)
            #sibs_pval = -np.log10(corrected[4])
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        
        # get fold enrichment of PTVs for each class
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)
        all_pros_fe = np.log2((np.sum(all_pros_data)/(num_class0 + num_class1 + num_class2 + num_class3))/background)

        print([class0_fe, class1_fe, class2_fe, class3_fe])
        
        shape = potential_shapes.pop(0)
        if all_pros:
            shapes += [shape, shape, shape, shape, shape]
            FE += [class0_fe, class1_fe, class2_fe, class3_fe, all_pros_fe]
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval]    
        else:
            shapes += [shape, shape, shape, shape]
            FE += [class0_fe, class1_fe, class2_fe, class3_fe]
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        
        thresh = -np.log10(0.05)
        if class0_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[0])
            #colors.append('pink')
        else:
            tick_labels.append('')
            print('class0')
            print(class0_pval)
            print(gene_set)
            colors.append(ref_colors[0])
        if class1_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[1])
            #colors.append('violet')
        else:
            tick_labels.append('')
            print('class 1')
            print(class1_pval)
            print(gene_set)
            colors.append(ref_colors[1])
        if class2_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[2])
            #colors.append('lightlimegreen')
        else:
            tick_labels.append('')
            print('class 2')
            print(class2_pval)
            print(gene_set)
            colors.append(ref_colors[2])
        if class3_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[3])
            #colors.append('lightblue')
        else:
            tick_labels.append('')
            print('class 3')
            print(class3_pval)
            print(gene_set)
            colors.append(ref_colors[3])
        if all_pros:
            if all_pros_pval < thresh:
                tick_labels.append('')
                colors.append('purple')
            else:
                print('all_pros')
                print(all_pros_pval)
                print(gene_set)
                colors.append('purple')
        #if sibs_pval < thresh:
        #    tick_labels.append('')
        #    colors.append('black')
        #else:
        #    tick_labels.append(gene_set)
        #    colors.append(ref_colors[4])

    # volcano plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('dnMis', fontsize=18)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='High-ASD/High-Delays', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/Low-Delays', markerfacecolor='violet', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='High-ASD/Low-Delays', markerfacecolor='limegreen', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/High-Delays', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Siblings', markerfacecolor='black', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='All probands', markerfacecolor='purple', markersize=10)]
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    for i, txt in enumerate(tick_labels):
        ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_6400_volcano_plot_DNV_missense_variants.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_volcano_plot_DNV_missense_variants.png', bbox_inches='tight')
    plt.close()


def volcano_noncoding(impute=False):
    '''Produce volcano plot for LOF variants (PTVs)'''
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    consequences_noncoding = ['intron_variant', 'upstream_gene_variant', 'downstream_gene_variant', '3_prime_UTR_variant', '5_prime_UTR_variant']
    #consequences_noncoding = ['synonymous_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_noncoding else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_noncoding else 0)
    
    gene_sets, gene_set_names = get_gene_sets()

    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    ref_colors = ['violet', 'red', 'limegreen', 'blue', 'black']
    colors = []

    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes', 'satterstrom', 'brain_expressed_genes', 'dd_genes']
    shapes = []
    potential_shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    potential_shapes = potential_shapes[0:len(gene_sets_to_keep)]

    for gene_set in gene_sets_to_keep:
        gene_subset = dnvs_pro[dnvs_pro[gene_set] == 1] # get subset of dnvs_pro which fall within genes in the gene set
        class0 = gene_subset[gene_subset['class'] == 0].groupby('spid')['consequence'].sum().tolist() # sum PTVs for each spid in class
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = gene_subset[gene_subset['class'] == 1].groupby('spid')['consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = gene_subset[gene_subset['class'] == 2].groupby('spid')['consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = gene_subset[gene_subset['class'] == 3].groupby('spid')['consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs[dnvs_sibs[gene_set] == 1].groupby('spid')['consequence'].sum().tolist()
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3
        
        # get pvalue comparing each class to the rest of the sample using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3
        total = class0 + class1 + class2 + class3 + sibs
        background = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3))/(num_class0 + num_class1 + num_class2 + num_class3)
        sib_background = np.sum(sibs)/num_sibs

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, all_pros_data, equal_var=False, alternative='greater')[1]

        # FDR CORRECTION
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        class0_fe = np.log2((np.sum(class0)/num_class0)/sib_background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/sib_background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/sib_background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/sib_background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background)

        shape = potential_shapes.pop(0)
        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]
        shapes += [shape, shape, shape, shape]
        thresh = -np.log10(0.05)
        if class0_pval < thresh:
            colors.append('gray')
            #colors.append('pink')
        else:
            print('class0')
            print(class0_pval)
            print(gene_set)
            colors.append(ref_colors[0])
        if class1_pval < thresh:
            colors.append('gray')
            #colors.append('violet')
        else:
            print('class 1')
            print(class1_pval)
            print(gene_set)
            colors.append(ref_colors[1])
        if class2_pval < thresh:
            colors.append('gray')
            #colors.append('lightlimegreen')
        else:
            print('class 2')
            print(class2_pval)
            print(gene_set)
            colors.append(ref_colors[2])
        if class3_pval < thresh:
            colors.append('gray')
            #colors.append('lightblue')
        else:
            print('class 3')
            print(class3_pval)
            print(gene_set)
            colors.append(ref_colors[3])
        #if sibs_pval < thresh:
        #    colors.append('black')
        #else:
        #    colors.append(ref_colors[4])
        
    # plot volcano plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
    ax.set_xlabel('log2 fold change', fontsize=14)
    ax.set_ylabel('-log10(q-value)', fontsize=14)
    ax.set_title('dnNoncoding', fontsize=16)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='High-ASD/High-Delays', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/Low-Delays', markerfacecolor='violet', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='High-ASD/Low-Delays', markerfacecolor='limegreen', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/High-Delays', markerfacecolor='blue', markersize=10)]
                       #Line2D([0], [0], marker='o', color='w', label='Siblings', markerfacecolor='black', markersize=10)]
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    #ax.set_xlim(-1.5, 1.5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    fig.savefig('GFMM_WGS_Analysis_Plots/WES_volcano_plot_DNV_synonymous_variants.png', bbox_inches='tight')
    plt.close()


def compare_constrained_gene_sets(impute=False, all_pros=False):
    # load dnvs
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    # get gene sets
    pli = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/pLI_table.txt', sep='\t')
    pli = pli[['gene', 'pLI']]
    pli_highest = pli[pli['pLI'] >= 0.995]['gene'].tolist()
    #pli_highest = pli[pli['pLI'] >= 0.9]['gene'].tolist()
    pli_medium = pli[(pli['pLI'] >= 0.5) & (pli['pLI'] < 0.995)]['gene'].tolist()
    pli_low = pli[(pli['pLI'] >= 0.5) & (pli['pLI'] < 0.9)]['gene'].tolist()
    #pli_low = pli[pli['pLI'] <= 0.1]['gene'].tolist()
    
    # get lof consequences
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)

    gene_sets = [pli_highest, pli_medium, pli_low]
    gene_set_names = ['pli_highest', 'pli_medium', 'pli_low']
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
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
    for gene_set, ax in zip(['pli_highest', 'pli_medium'], (ax1, ax2)):
        props = []
        stds = []
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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

        # get average number of PTVs per spid in each class
        class0_avg = np.mean(class0)
        class1_avg = np.mean(class1)
        class2_avg = np.mean(class2)
        class3_avg = np.mean(class3)
        sibs_avg = np.mean(sibs)

        # get pvalues comparing each class to siblings using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])
        if all_pros:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
        else:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
        pvals = {k: pval for k, pval in enumerate(corrected)}
        print(pvals)

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

        # PLOT AS MEAN+S.E.
        # plot props as scatter
        x_values = list(np.arange(len(props)))
        print(f'x_values: {x_values}')
        y_values = props
        print(f'y values: {y_values}')
        print(f'std: {stds}')
        colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
        for i in range(len(x_values)):
            ax.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
        ax.set_xlabel('')
        ax.set_ylabel('dnLoF per offspring', fontsize=16)
        ax.set_xticks(x_values)
        #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
        if gene_set == 'pli_highest':
            ax.set_title('pLI ≥ 0.995', fontsize=21)
        elif gene_set == 'pli_medium':
            ax.set_title('0.5 ≤ pLI < 0.995', fontsize=21)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 ≤ pLI < 0.9', fontsize=21)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=14)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        ax.grid(color='gray', linestyle='-', linewidth=0.5)

        # ADD SIGNIFICANCE TO PLOT
        for grpidx in [0,1,2,3]:
            p_value = pvals[grpidx]
            x_position = grpidx+1
            y_position = y_values[grpidx+1]
            se_value = stds[grpidx+1]
            ypos = y_position + se_value 
            if p_value < 0.01:
                ax.annotate('***', xy=(x_position, ypos), ha='center', size=20)
            elif p_value < 0.05:
                ax.annotate('**', xy=(x_position, ypos), ha='center', size=20)
            elif p_value < 0.1:
                ax.annotate('*', xy=(x_position, ypos), ha='center', size=20)
    fig.tight_layout()
    fig.savefig('GFMM_WGS_Analysis_Plots/WES_CONSTRAINT_avg_ptvs_per_spid.png', bbox_inches='tight')
    plt.close()

    # PLOT FOR MISSENSE VARIANTS
    # get lof consequences
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)

    gene_sets = [pli_highest, pli_medium, pli_low]
    gene_set_names = ['pli_highest', 'pli_medium', 'pli_low']
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    for gene_set in ['pli_highest', 'pli_medium', 'pli_low']:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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

        # ALL PROS VS. ALL SIBS ANALYSIS
        all_pros_data = dnvs_pro.groupby('spid')['gene_set&consequence'].sum().tolist()
        all_pros_avg = np.mean(all_pros_data)
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        print(all_pros_pval)
        # plot bar of average number of PTVs per proband and siblings
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(['Siblings', 'Probands'], [np.mean(sibs), all_pros_avg], color=['black', 'purple'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
        ax.set_ylabel('dnLoF per sample', fontsize=14)
        if gene_set == 'pli_highest':
            ax.set_title('pLI >= 0.995', fontsize=16)
        elif gene_set == 'pli_medium':
            ax.set_title('0.9 <= pLI < 0.995', fontsize=16)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 <= pLI < 0.9', fontsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black')
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_avg_missense_constrained_all_pros_vs_sibs_' + gene_set + '.png', bbox_inches='tight')
        plt.close()

        # get average number of missense variants per spid in each class
        class0_avg = np.mean(class0)
        class1_avg = np.mean(class1)
        class2_avg = np.mean(class2)
        class3_avg = np.mean(class3)
        sibs_avg = np.mean(sibs)

        # get pvalues comparing each class to siblings using a t-test
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
        print(corrected)

        # plot average number of missense variants per spid in each class
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if impute:
            ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
        else:
            ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
        ax.set_ylabel('dnMissense per sample', fontsize=14)
        if gene_set == 'pli_highest':
            ax.set_title('pLI >= 0.995', fontsize=16)
        elif gene_set == 'pli_medium':
            ax.set_title('0.5 <= pLI < 0.995', fontsize=16)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 <= pLI < 0.9', fontsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black')
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_avg_missense_per_spid_' + gene_set + '.png', bbox_inches='tight')
        plt.close()

    # rare INH PTV variants - crPTVs
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited.pkl', 'rb') as f: # likely pathogenic
    #    spid_to_num_missense = rick.load(f)

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt' # 1588 sibs paired
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}

    gene_sets, gene_set_names = get_gene_sets()

    for gene_set in gene_set_names:
        if gene_set == 'pli_genes_highest':
            i = gene_set_names.index(gene_set)
            ptv_class0_high = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
            ptv_class1_high = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
            ptv_class2_high = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
            ptv_class3_high = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
            all_pros_high = [v[i] for k, v in pros_to_num_ptvs.items()]
            ptv_sibs_high = [v[i] for k, v in sibs_to_num_ptvs.items()]
        elif gene_set == 'pli_genes_medium':
            i = gene_set_names.index(gene_set)
            ptv_class0_medium = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
            ptv_class1_medium = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
            ptv_class2_medium = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
            ptv_class3_medium = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
            all_pros_medium = [v[i] for k, v in pros_to_num_ptvs.items()]
            ptv_sibs_medium = [v[i] for k, v in sibs_to_num_ptvs.items()]

    # get pvalues
    print('pli_highest')
    pval_class0_high = ttest_ind(all_pros_high, ptv_sibs_high, equal_var=False, alternative='greater')[1]
    pval_class1_high = ttest_ind(ptv_class0_high, ptv_sibs_high, equal_var=False, alternative='greater')[1]
    pval_class2_high = ttest_ind(ptv_class1_high, ptv_sibs_high, equal_var=False, alternative='greater')[1]
    pval_class3_high = ttest_ind(ptv_class2_high, ptv_sibs_high, equal_var=False, alternative='greater')[1]
    all_pros_high_pval = ttest_ind(all_pros_high, ptv_sibs_high, equal_var=False, alternative='greater')[1]
    print(all_pros_high_pval)
    corrected = multipletests([pval_class0_high, pval_class1_high, pval_class2_high, pval_class3_high, all_pros_high_pval], method='fdr_bh', alpha=0.05)[1]
    print(corrected)
    print('pli_medium')
    pval_class0_medium = ttest_ind(all_pros_medium, ptv_sibs_medium, equal_var=False, alternative='greater')[1]
    pval_class1_medium = ttest_ind(ptv_class0_medium, ptv_sibs_medium, equal_var=False, alternative='greater')[1]
    pval_class2_medium = ttest_ind(ptv_class1_medium, ptv_sibs_medium, equal_var=False, alternative='greater')[1]
    pval_class3_medium = ttest_ind(ptv_class2_medium, ptv_sibs_medium, equal_var=False, alternative='greater')[1]
    all_pros_medium_pval = ttest_ind(all_pros_medium, ptv_sibs_medium, equal_var=False, alternative='greater')[1]
    print(all_pros_medium_pval)
    corrected = multipletests([pval_class0_medium, pval_class1_medium, pval_class2_medium, pval_class3_medium, all_pros_medium_pval], method='fdr_bh', alpha=0.05)[1]
    print(corrected)

    # plot average number of PTVs per spid in each class
    colors = ['red', 'violet', 'limegreen', 'blue', 'black']
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if impute:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [np.mean(ptv_class1_high), np.mean(ptv_class0_high), np.mean(ptv_class2_high), np.mean(ptv_class3_high), np.mean(ptv_sibs_high)], color=colors, alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [np.mean(ptv_class0_high), np.mean(ptv_class1_high), np.mean(ptv_class2_high), np.mean(ptv_class3_high), np.mean(ptv_sibs_high)], color=colors, alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('crPTVs per sample', fontsize=14)
    plt.xlabel('')
    ax.set_title('pLI >= 0.995', fontsize=16)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    fig.savefig('GFMM_WGS_Analysis_Plots/WES_avg_high_inherited_ptvs_per_spid.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if impute:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [np.mean(ptv_class0_medium), np.mean(ptv_class1_medium), np.mean(ptv_class2_medium), np.mean(ptv_class3_medium), np.mean(ptv_sibs_medium)], color=colors, alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [np.mean(ptv_class1_medium), np.mean(ptv_class0_medium), np.mean(ptv_class2_medium), np.mean(ptv_class3_medium), np.mean(ptv_sibs_medium)], color=colors, alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('crPTVs per sample', fontsize=14)
    ax.set_title('0.5 <= pLI < 0.995', fontsize=16)
    plt.xlabel('')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    fig.savefig('GFMM_WGS_Analysis_Plots/WES_avg_medium_inherited_ptvs_per_spid.png', bbox_inches='tight')
    plt.close()
        

def GO_term_analysis(impute=False):
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute) # DENOVO VARIANTS
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_synonymous = ['synonymous_variant']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    go_terms = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/GO_genes.csv', header=0, index_col=False)
    go_terms = go_terms[['gene', 'GO-0005634_nucleus',
       'GO-0000122_neg_reg_pol2', 'GO-0000785_chromatin',
       'GO-0000981_pol2_tf_activity', 'GO-0000988_Protein_tf_activity',
       'GO-0003682_chromatin_binding', 'GO-0003700_DNA_tf_binding',
       'GO-0006325_chromatin_organization', 'GO-0010468_reg_gene_exp',
       'GO-0045944_pos_reg_pol2', 'GO-0005737_cytoplasm',
       'GO-0042391_reg_mem_potential', 'GO-0045202_synapse',
       'GO-0043005_neuron_projection', 'GO-0044325_ion_channel_binding', 'GO-0007010_cytoskeleton_organization']]
    # get list of genes in each GO term
    go_terms = go_terms.set_index('gene')
    go_terms = go_terms.transpose()
    go_terms = go_terms.reset_index()
    go_terms = go_terms.rename({'index': 'GO_term'}, axis='columns')
    go_terms = go_terms.melt(id_vars='GO_term', var_name='gene', value_name='in_GO_term')
    go_terms = go_terms[go_terms['in_GO_term'] == 1]
    go_terms = go_terms.drop('in_GO_term', axis=1)
    go_terms = go_terms.groupby('GO_term')['gene'].apply(list).reset_index()
    go_terms['num_genes'] = go_terms['gene'].apply(lambda x: len(x))
    #go_terms = go_terms[go_terms['num_genes'] > 10]
    go_terms = go_terms.reset_index()

    # QUICK GENE EXTRACTION
    # subset dnvs_pro to 'GO-0006325_chromatin_organization' genes 
    #chromatin_genes = go_terms[go_terms['GO_term'] == 'GO-0006325_chromatin_organization']['gene'].tolist()[0]
    #dnvs_pro['chromatin'] = dnvs_pro['name'].apply(lambda x: 1 if x in chromatin_genes else 0)
    #dnvs_pro_chromatin = dnvs_pro[dnvs_pro['class'] == 1]
    #dnvs_pro_chromatin = dnvs_pro_chromatin[dnvs_pro_chromatin['chromatin'] == 1]
    #dnvs_pro_chromatin = dnvs_pro_chromatin.groupby('name')['consequence'].sum().reset_index()
    #dnvs_pro_chromatin = dnvs_pro_chromatin.sort_values('consequence', ascending=False)
    #top_10_chromatin = dnvs_pro_chromatin.head(10)
    #print(top_10_chromatin['name'].tolist())

    # annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(go_terms)):
        dnvs_pro[go_terms['GO_term'][i]] = dnvs_pro['name'].apply(lambda x: 1 if x in go_terms.iloc[i]['gene'] else 0)
        dnvs_sibs[go_terms['GO_term'][i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in go_terms.iloc[i]['gene'] else 0)

    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []
    
    # for each GO_term, count number of PTVs for each SPID which fall within genes in that GO_term
    go_term_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for go_term in go_terms['GO_term']:
        print(go_term)
        dnvs_pro['gene_set&consequence'] = dnvs_pro[go_term] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[go_term] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

        class0_df = pd.DataFrame({'variable': go_term, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': go_term, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': go_term, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': go_term, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': go_term, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        go_term_to_enrichment[go_term] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='value', y='variable', size='Fold Enrichment', hue='Cluster', palette=colors, markers=markers, sizes=(50, 500))
    for i, row in validation_subset.iterrows():
        if row['cluster'] == -1:
            ax.scatter(row['value'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('-log10(q-value)', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnPTVs', fontsize=24)
    plt.axvline(x=-np.log10(0.05), color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_plot_ptvs_GO_term_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_plot_ptvs_GO_term_analysis.png', bbox_inches='tight')
    plt.close()


    # MISSENSE VARIANTS
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    # annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(go_terms)):
        dnvs_pro[go_terms['GO_term'][i]] = dnvs_pro['name'].apply(lambda x: 1 if x in go_terms.iloc[i]['gene'] else 0)
        dnvs_sibs[go_terms['GO_term'][i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in go_terms.iloc[i]['gene'] else 0)

    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []
    
    go_term_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for go_term in go_terms['GO_term']:
        print(go_term)
        dnvs_pro['gene_set&consequence'] = dnvs_pro[go_term] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[go_term] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

        class0_df = pd.DataFrame({'variable': go_term, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': go_term, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': go_term, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': go_term, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': go_term, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        go_term_to_enrichment[go_term] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='value', y='variable', size='Fold Enrichment', hue='Cluster', palette=colors, markers=markers, sizes=(50, 500))
    for i, row in validation_subset.iterrows():
        if row['cluster'] == -1:
            ax.scatter(row['value'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('-log10(q-value)', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnMissense', fontsize=24)
    plt.axvline(x=-np.log10(0.05), color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_plot_missense_GO_term_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_plot_missense_GO_term_analysis.png', bbox_inches='tight')
    plt.close()
    

def combine_inherited_vep_files():

    #dir = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/inherited_vep_predictions_nofilter/'
    dir = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/inherited_vep_predictions_plugins_filtered/' # filtered repeats + centromeres
    # get files that end with '.vcf'
    files = [f for f in os.listdir(dir) if f.endswith('.vcf')]
    spids = [f.split('.')[0] for f in files]

    # dictionary of spid to number of PTVs
    spid_to_num_ptvs = {}
    # dictionary of spid to number of missense variants
    spid_to_num_missense = {}
    # these will map spid to num_ptvs and num_missense for each gene set (list for each spid)

    #gene_sets, gene_set_names, _, _ = get_trend_celltype_gene_sets()
    gene_sets, gene_set_names = get_gene_sets()
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0)
    gfmm_ids = gfmm_labels['subject_sp_id'].tolist()

    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' # 2027 sibs paired
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()

    # COMBINE gfmm_ids and sibling_list
    #gfmm_ids = gfmm_ids + sibling_list

    #af = load_AF()
    #var_to_af = dict(zip(af['id'], af['af']))

    counts_pros = []
    counts_sibs = []

    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    gene_to_spid_counts = defaultdict(dict)
    for i in range(len(files)):
        if (spids[i] not in gfmm_ids) and (spids[i] not in sibling_list):
            continue
        cols = ['Uploaded_variation', 'Location', 'Allele', 'Gene', 'Feature', 'Feature_type', 'Consequence', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 'Extra']
        df = pd.read_csv(dir + files[i], sep='\t', comment='#', header=None, names=cols, index_col=False)
        if spids[i] in gfmm_ids:
            counts_pros.append(df.shape[0])
        elif spids[i] in sibling_list:
            counts_sibs.append(df.shape[0])
        continue
        df = df[['Uploaded_variation', 'Gene', 'Consequence', 'Extra']]
        df['af'] = df['Uploaded_variation'].map(var_to_af)
        # filter to rare variants
        df = df[(df['af'] <= 0.01) | (df['af'].isna())]
        # no AF variants
        #df = df[df['af'].isna()]
        
        # filter PTVs and missense variants with plugins LOFTEE, AM
        #df['am_class'] = df['Extra'].str.extract(r'am_class=(.*?);')
        #df['am_class'] = df['am_class'].apply(lambda x: 1 if x in ['likely_pathogenic'] else 0)
        df['am_pathogenicity'] = df['Extra'].str.extract(r'am_pathogenicity=([\d.]+)').astype(float)
        df['am_pathogenicity'] = df['am_pathogenicity'].apply(lambda x: 1 if x>=0.9 else 0)
        df['LoF'] = df['Extra'].str.extract(r'LoF=(.*?);')
        df['LoF'] = df['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
        df['LoF_flags'] = df['Extra'].str.extract(r'LoF_flags=(.*?);')
        # reformat lof_flags to 0/1 (whether passed flags or not)
        df['LoF_flags'] = df['LoF_flags'].fillna(1) # no flag is good
        df['LoF_flags'] = df['LoF_flags'].apply(lambda x: 1 if x in ['SINGLE_EXON',1] else 0)
        df = df.drop('Extra', axis=1)

        df['name'] = df['Gene'].map(ensembl_to_gene)
        '''
        # for each gene set, get number of PTVs and missense variants for each spid in each gene
        for gene_set in gene_sets:
            for gene in gene_set:
                # get number of PTVs for spid in gene set, filter with LoF and LoF_flags
                num_ptvs = df[(df['name'] == gene) & df['Consequence'].isin(consequences_lof) & (df['LoF'] == 1) & (df['LoF_flags'] == 1)].shape[0]
                num_missense = df[(df['name'] == gene) & df['Consequence'].isin(consequences_missense) & (df['am_pathogenicity'] == 1)].shape[0]
                gene_to_spid_counts[gene][spids[i]] = [num_ptvs, num_missense]
                
        '''
        ptv_counts = []
        missense_counts = []
        for gene_set in gene_sets:
            # get number of PTVs for spid in gene set, filter with LoF and LoF_flags
            num_ptvs = df[df['name'].isin(gene_set) & df['Consequence'].isin(consequences_lof) & (df['LoF'] == 1) & (df['LoF_flags'] == 1)].shape[0]
            # get number of missense variants for spid in gene set
            num_missense = df[df['name'].isin(gene_set) & df['Consequence'].isin(consequences_missense) & (df['am_pathogenicity'] == 1)].shape[0]
            ptv_counts.append(num_ptvs)
            missense_counts.append(num_missense)
        spid_to_num_ptvs[spids[i]] = ptv_counts
        spid_to_num_missense[spids[i]] = missense_counts
        
    # save dictionaries to pickle
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited_noaf.pkl', 'wb') as f:
    #    rick.dump(spid_to_num_ptvs, f)
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited_noaf_90patho.pkl', 'wb') as f:
    #    rick.dump(spid_to_num_missense, f)

    print(counts_pros)
    print(counts_sibs)
    print(np.mean(counts_pros))
    print(np.std(counts_pros))
    print(np.mean(counts_sibs))
    print(np.std(counts_sibs))
    exit()

    # save gene_to_spid_counts to pickle
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/gene_to_spid_counts_rare_inherited_noaf.pkl', 'wb') as f:
    #    rick.dump(gene_to_spid_counts, f)


def get_inherited_variant_count():

    dir = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/inherited_vep_predictions_nofilter/'
    # get files that end with '.vcf'
    files = [f for f in os.listdir(dir) if f.endswith('.vcf')]
    # spids - get from file name (split by . and take first element)
    spids = [f.split('.')[0] for f in files]

    spid_to_num_inherited_high_pli = {}
    spid_to_num_inherited_medium_pli = {}
    spid_to_num_inherited_low_pli = {}

    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv') #6406
    gfmm_ids = gfmm_labels['subject_sp_id'].tolist()

    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt' # 2027 sibs paired
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()

    af = load_AF()
    # get dictionary mapping variant id to AF
    var_to_af = dict(zip(af['id'], af['af']))

    pli = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/pLI_table.txt', sep='\t')
    pli = pli[['gene', 'pLI']]
    pli_highest = pli[pli['pLI'] > 0.995]['gene'].tolist()
    #pli_highest = pli[pli['pLI'] >= 0.9]['gene'].tolist()
    pli_medium = pli[(pli['pLI'] >= 0.5) & (pli['pLI'] < 0.995)]['gene'].tolist()
    pli_low = pli[pli['pLI'] < 0.5]['gene'].tolist()
    #pli_low = pli[pli['pLI'] <= 0.1]['gene'].tolist()
    pten = []

    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    for i in range(len(files)):
        # continue if spid is not in gfmm_labels
        #if spids[i] not in sibling_list:
        #    continue
        cols = ['Uploaded_variation', 'Location', 'Allele', 'Gene', 'Feature', 'Feature_type', 'Consequence', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 'Extra']
        df = pd.read_csv(dir + files[i], sep='\t', comment='#', header=None, names=cols, index_col=False)
        # only keep 'Uploaded_variation' and 'Consequence' columns
        df['af'] = df['Uploaded_variation'].map(var_to_af)
        # filter to rare variants
        df = df[df['af'] < 0.01]
        # count number of inherited variants for each spid
        #num_inherited = df.shape[0]
        #spid_to_num_inherited[spids[i]] = num_inherited
        # annotate with gene
        df['name'] = df['Gene'].map(ensembl_to_gene)
        # count number of LOF inherited variants in each gene set (pLI_highest, pli_medium)
        num_inherited_highest = df[df['name'].isin(pli_highest) & df['Consequence'].isin(consequences_lof)].shape[0]
        num_inherited_medium = df[df['name'].isin(pli_medium) & df['Consequence'].isin(consequences_lof)].shape[0]
        num_inherited_low = df[df['name'].isin(pli_low) & df['Consequence'].isin(consequences_lof)].shape[0]

        spid_to_num_inherited_high_pli[spids[i]] = num_inherited_highest
        spid_to_num_inherited_medium_pli[spids[i]] = num_inherited_medium
        spid_to_num_inherited_low_pli[spids[i]] = num_inherited_low
        
    # save dictionaries to pickle
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_rare_inherited_sibs.pkl', 'wb') as f:
    #    rick.dump(spid_to_num_inherited, f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_inherited_high_pli_sibs.pkl', 'wb') as f:
        rick.dump(spid_to_num_inherited_high_pli, f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_inherited_medium_pli_sibs.pkl', 'wb') as f:
        rick.dump(spid_to_num_inherited_medium_pli, f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_inherited_low_pli_sibs.pkl', 'wb') as f:
        rick.dump(spid_to_num_inherited_low_pli, f)


def get_count_burden_figure(impute=False):
    # 1. AVERAGE NUMBER OF TOTAL DNVS PER SPID
    count_file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_DNV_count.txt'
    counts = pd.read_csv(count_file, sep='\t', index_col=False)
    counts = counts.rename(columns={'SPID': 'spid'})

    # first annotate as proband or sibling using WES2 mastertable
    wes_spids = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes_spids = pd.read_csv(wes_spids, sep='\t')
    wes_spids = wes_spids[['spid', 'asd']]
    counts = pd.merge(counts, wes_spids, how='inner', on='spid')
    pro_counts = counts[counts['asd'] == 2]
    sib_counts = counts[counts['asd'] == 1]

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0)
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0)
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
    pro_counts = pd.merge(pro_counts, gfmm_labels, how='inner', on='spid')
    
    if impute:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt'
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sib_counts = pd.merge(sib_counts, sibling_list, how='inner', on='spid')

    # HYPOTHESIS TESTING - compare each class to siblings
    class0_pval = ttest_ind(pro_counts[pro_counts['mixed_pred'] == 0]['count'], sib_counts['count'], equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind(pro_counts[pro_counts['mixed_pred'] == 1]['count'], sib_counts['count'], equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind(pro_counts[pro_counts['mixed_pred'] == 2]['count'], sib_counts['count'], equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind(pro_counts[pro_counts['mixed_pred'] == 3]['count'], sib_counts['count'], equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of DNVs per SPID for each class and sibs
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 5))
    if impute:
        colors = ['violet', 'red', 'limegreen', 'blue', 'black']
    else:
        colors = ['red', 'violet', 'limegreen', 'blue', 'black']
    ax.bar([0, 1, 2, 3, 4], [pro_counts[pro_counts['mixed_pred'] == 0]['count'].mean(), pro_counts[pro_counts['mixed_pred'] == 1]['count'].mean(), pro_counts[pro_counts['mixed_pred'] == 2]['count'].mean(), pro_counts[pro_counts['mixed_pred'] == 3]['count'].mean(), sib_counts['count'].mean()], color=colors, width=0.7, alpha=0.9, edgecolor='black', linewidth=1)
    ax.set_xticks([0, 1, 2, 3, 4])
    if impute:
        labels = ['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings']
    else:
        labels = ['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings']
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel('DNVs per sample', fontsize=18)
    ax.set_title('Total DNVs', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_DNV_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_DNV_burden.png', bbox_inches='tight')
    plt.close()

    # load dnvs
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # 2. AVERAGE NUMBER OF PTVS PER SPID
    # get number of PTVs for each spid
    # annotate variants as PTV or not
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['ptv'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['ptv'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_pro['final_consequence_lof'] = dnvs_pro['ptv'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
    dnvs_sibs['final_consequence_lof'] = dnvs_sibs['ptv'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
    # get average number of PTVs per class
    class0_avg = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['final_consequence_lof'].sum().mean()
    class1_avg = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['final_consequence_lof'].sum().mean()
    class2_avg = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['final_consequence_lof'].sum().mean()
    class3_avg = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['final_consequence_lof'].sum().mean()
    sibs_avg = dnvs_sibs.groupby('spid')['final_consequence_lof'].sum().mean()

    # hypothesis testing - compare each class to sibs
    class0_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['ptv'].sum().tolist(), dnvs_sibs.groupby('spid')['ptv'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['ptv'].sum().tolist(), dnvs_sibs.groupby('spid')['ptv'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['ptv'].sum().tolist(), dnvs_sibs.groupby('spid')['ptv'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['ptv'].sum().tolist(), dnvs_sibs.groupby('spid')['ptv'].sum().tolist(), equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of PTVs per spid in each class
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if impute:
        ax.bar(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('dnPTVs per sample', fontsize=18)
    ax.set_title('dnPTVs', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_DNV_PTV_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_DNV_PTV_burden.png', bbox_inches='tight')
    plt.close()

    # 3. AVERAGE NUMBER OF DNV MISSENSE VARIANTS PER SPID
    # get number of missense variants for each spid
    # annotate variants as missense or not
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    dnvs_pro['missense'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['missense'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_pro['final_consequence_missense'] = dnvs_pro['missense'] * dnvs_pro['am_pathogenicity']
    dnvs_sibs['final_consequence_missense'] = dnvs_sibs['missense'] * dnvs_sibs['am_pathogenicity']
    # get average number of missense variants per class
    class0_avg = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['final_consequence_missense'].sum().mean()
    class1_avg = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['final_consequence_missense'].sum().mean()
    class2_avg = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['final_consequence_missense'].sum().mean()
    class3_avg = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['final_consequence_missense'].sum().mean()
    sibs_avg = dnvs_sibs.groupby('spid')['final_consequence_missense'].sum().mean()

    # hypothesis testing - compare each class to sibs
    class0_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['missense'].sum().tolist(), dnvs_sibs.groupby('spid')['missense'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['missense'].sum().tolist(), dnvs_sibs.groupby('spid')['missense'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['missense'].sum().tolist(), dnvs_sibs.groupby('spid')['missense'].sum().tolist(), equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind(dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['missense'].sum().tolist(), dnvs_sibs.groupby('spid')['missense'].sum().tolist(), equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of missense variants per spid in each class
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if impute:
        ax.bar(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('dnMissense per sample', fontsize=18)
    ax.set_title('dn Missense', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_DNV_missense_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_DNV_missense_burden.png', bbox_inches='tight')
    plt.close()

    # get spid_to_class dictionary
    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0)
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0)
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    # 4. AVERAGE NUMBER OF INHERITED SNPS PER SPID
    # get average number of inherited PTVs per class
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_inherited_probands.pkl', 'rb') as f:
        spid_to_num_inherited = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_inherited = rick.load(f)
    
    # if not impute, subset to 4700
    if not impute:
        spid_to_num_inherited = {k: v for k, v in spid_to_num_inherited.items() if k in gfmm_labels['spid'].tolist()}
        sibs_to_num_inherited = {k: v for k, v in sibs_to_num_inherited.items() if k in sibling_list['spid'].tolist()}
    
    class0_avg = np.mean([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 0])
    class1_avg = np.mean([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 1])
    class2_avg = np.mean([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 2])
    class3_avg = np.mean([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 3])
    sibs_avg = np.mean([v for k, v in sibs_to_num_inherited.items()])

    # hypothesis testing - compare each class to sibs
    class0_pval = ttest_ind([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 0], [v for k, v in sibs_to_num_inherited.items()], equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 1], [v for k, v in sibs_to_num_inherited.items()], equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 2], [v for k, v in sibs_to_num_inherited.items()], equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind([v for k, v in spid_to_num_inherited.items() if spid_to_class[k] == 3], [v for k, v in sibs_to_num_inherited.items()], equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of inherited variants per spid in each class
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if impute:
        ax.bar(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('Inherited SNPs per sample', fontsize=18)
    ax.set_title('Inherited SNPs', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_inherited_SNP_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_inherited_SNP_burden.png', bbox_inches='tight')
    plt.close()

    # LOAD INHERITED VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_rare_inherited_6400.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_rare_inherited_6400.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_rare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_rare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_missense = rick.load(f)
    
    if not impute:
        spid_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in gfmm_labels['spid'].tolist()}
        spid_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in gfmm_labels['spid'].tolist()}
        sibs_to_num_ptvs = {k: v for k, v in sibs_to_num_ptvs.items() if k in sibling_list['spid']}
        sibs_to_num_missense = {k: v for k, v in sibs_to_num_missense.items() if k in sibling_list['spid']}

    # 5. AVERAGE NUMBER OF INHERITED PTVs VARIANTS PER SPID
    # inherited PTVs is the first element in the list for each spid (ptvs across all genes)
    class0_avg = np.mean([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 0])
    class1_avg = np.mean([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 1])
    class2_avg = np.mean([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 2])
    class3_avg = np.mean([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 3])
    sibs_avg = np.mean([v[0] for k, v in sibs_to_num_ptvs.items()])
    print([class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg])

    # hypothesis testing - compare each class to sibs
    class0_pval = ttest_ind([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 0], [v[0] for k, v in sibs_to_num_ptvs.items()], equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 1], [v[0] for k, v in sibs_to_num_ptvs.items()], equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 2], [v[0] for k, v in sibs_to_num_ptvs.items()], equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind([v[0] for k, v in spid_to_num_ptvs.items() if spid_to_class[k] == 3], [v[0] for k, v in sibs_to_num_ptvs.items()], equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of PTVs per spid in each class
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if impute:
        ax.bar(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('Inherited PTVs per sample', fontsize=18)
    ax.set_title('Inherited PTVs', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_inherited_PTV_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_inherited_PTV_burden.png', bbox_inches='tight')
    plt.close()

    # 6. AVERAGE NUMBER OF INHERITED MISSSENSE VARIANTS PER SPID
    # inherited missense is the first element in the list for each spid (missense across all genes)
    class0_avg = np.mean([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 0])
    class1_avg = np.mean([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 1])
    class2_avg = np.mean([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 2])
    class3_avg = np.mean([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 3])
    sibs_avg = np.mean([v[0] for k, v in sibs_to_num_missense.items()])
    print(v[0] for k, v in sibs_to_num_missense.items())
    print([class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg])
    
    # hypothesis testing - compare each class to sibs
    class0_pval = ttest_ind([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 0], [
        v[0] for k, v in sibs_to_num_missense.items()], equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 1], [
        v[0] for k, v in sibs_to_num_missense.items()], equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 2], [
        v[0] for k, v in sibs_to_num_missense.items()], equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind([v[0] for k, v in spid_to_num_missense.items() if spid_to_class[k] == 3], [
        v[0] for k, v in sibs_to_num_missense.items()], equal_var=False, alternative='greater')[1]
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(f'p-values: {corrected}')

    # plot average number of missense variants per spid in each class
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if impute:
        ax.bar(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    else:
        ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
    ax.set_ylabel('Inherited Missense per sample', fontsize=18)
    ax.set_title('Inherited Missense', fontsize=22)
    ax.tick_params(axis='x', labelsize=12)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_6400_average_inherited_missense_burden.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_COUNT_average_inherited_missense_burden.png', bbox_inches='tight')
    plt.close()
    

def volcano_inherited(impute=False, all_pros=False):

    # LOAD FILTERED RARE VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited.pkl', 'rb') as f: # only rare INH vars
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited.pkl', 'rb') as f: # likely pathogenic
        spid_to_num_missense = rick.load(f)
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited_90patho.pkl', 'rb') as f: # 0.9 pathogenicity
    #    spid_to_num_missense = rick.load(f)

    '''
    # LOAD RARE VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_rare_inherited_6400.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_rare_inherited_6400.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_rare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_rare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_missense = rick.load(f)
    '''
    '''
    # LOAD ULTRA-RARE VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_ultrarare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ultrarare_inherited.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_ultrarare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ultrarare_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_missense = rick.load(f)
    '''
    '''
    # LOAD no AF VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_noAF_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_noAF_inherited.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_noAF_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_noAF_inherited_sibs.pkl', 'rb') as f:
        sibs_to_num_missense = rick.load(f)
    '''

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' # 1588 sibs paired
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in spid_to_class}
    # print average number of ptvs and missense per individual
    print(np.mean([v for k, v in pros_to_num_ptvs.items()]))
    print(np.mean([v for k, v in pros_to_num_missense.items()]))
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in sibling_list}
    print(np.mean([v for k, v in sibs_to_num_ptvs.items()]))
    print(np.mean([v for k, v in sibs_to_num_missense.items()]))
    print(len(sibs_to_num_missense))

    gene_sets, gene_set_names = get_gene_sets()

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    gene_set_to_index = {gene_set: i for i, gene_set in enumerate(gene_set_names)}
    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes', 'dd_genes']
    indices = [gene_set_to_index[gene_set] for gene_set in gene_sets_to_keep]
    shapes = []
    potential_shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    potential_shapes = potential_shapes[0:len(gene_sets_to_keep)]
    
    for i in indices:
        gene_set = gene_set_names[i] # get name of gene set
        # get number of PTVs for each spid in gene set
        class0 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        all_pros_data = class0 + class1 + class2 + class3
        sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]
        
        # get pvalue comparing each class to the rest
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        # get pvalue comparing each class to sibs using a t-test
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]

        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])

        # multiple testing correction
        if all_pros:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval], method='fdr_bh')[1]
            all_pros_pval = -np.log10(corrected[4])
            #sibs_pval = -np.log10(corrected[5])
        else:
            corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh')[1]
            print(corrected)
            #sibs_pval = -np.log10(corrected[4])
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])

        # get fold enrichment of PTVs for each class
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3))/(num_class0 + num_class1 + num_class2 + num_class3)
        background = np.sum(sibs)/(num_sibs)
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        all_pros_fe = np.log2((np.sum(all_pros_data)/(num_class0+num_class1+num_class2+num_class3))/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

        print([class0_fe, class1_fe, class2_fe, class3_fe])

        shape = potential_shapes.pop(0)
        if all_pros:
            shapes += [shape, shape, shape, shape, shape]
            FE += [class0_fe, class1_fe, class2_fe, class3_fe, all_pros_fe]
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval, all_pros_pval]
        else:
            shapes += [shape, shape, shape, shape]
            FE += [class0_fe, class1_fe, class2_fe, class3_fe]
            pvals += [class0_pval, class1_pval, class2_pval, class3_pval]

        thresh = -np.log10(0.05)
        if class0_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[0])
            #colors.append('pink')
        else:
            tick_labels.append('')
            print('class0')
            print(class0_pval)
            print(gene_set)
            colors.append(ref_colors[0])
        if class1_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[1])
            #colors.append('violet')
        else:
            tick_labels.append('')
            print('class1')
            print(class1_pval)
            print(gene_set)
            colors.append(ref_colors[1])
        if class2_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[2])
            #colors.append('lightlimegreen')
        else:
            tick_labels.append('')
            print('class2')
            print(class2_pval)
            print(gene_set)
            colors.append(ref_colors[2])
        if class3_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[3])
            #colors.append('lightblue')
        else:
            # if fe > 0.15, append gene set name
            if class3_fe > 0.15:
                tick_labels.append('')
            else:
                tick_labels.append('')
            print('class3')
            print(class3_pval)
            print(gene_set)
            colors.append(ref_colors[3])
        if all_pros:
            tick_labels.append('')
            colors.append('purple')
        #if sibs_pval < thresh:
        #    tick_labels.append('')
        #    colors.append('black')
        #else:
        #    tick_labels.append('')
        #    colors.append(ref_colors[4])
        
    # plot volcano plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhLoF', fontsize=18)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='High-ASD/High-Delays', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/Low-Delays', markerfacecolor='violet', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='High-ASD/Low-Delays', markerfacecolor='limegreen', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/High-Delays', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Siblings', markerfacecolor='black', markersize=10)]
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    #ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    #for i, txt in enumerate(tick_labels):
    #    ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_6400_volcano_plot_INH_PTVs.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_volcano_plot_INH_PTVs.png', bbox_inches='tight')
    plt.close()

    # INHERITED MISSENSE VARIANT ANALYSIS
    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    gene_sets_to_keep = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes', 'sfari_genes1', 'satterstrom', 'brain_expressed_genes', 'dd_genes']
    shapes = []
    potential_shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    potential_shapes = potential_shapes[0:len(gene_sets_to_keep)]

    for i in indices:
        gene_set = gene_set_names[i]
        class0 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_missense.items() if spid_to_class[k] == 3]
        sibs = [v[i] for k, v in sibs_to_num_missense.items()]

        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]

        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        #sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/(num_sibs)
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

        shape = potential_shapes.pop(0)
        shapes += [shape, shape, shape, shape]

        FE += [class0_fe, class1_fe, class2_fe, class3_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval]

        thresh = -np.log10(0.05)
        if class0_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[0])
            #colors.append('pink')
        else:
            tick_labels.append('')
            print('class0')
            print(class0_pval)
            print(gene_set)
            colors.append(ref_colors[0])
        if class1_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[1])
            #colors.append('violet')
        else:
            tick_labels.append('')
            print('class1')
            print(class1_pval)
            print(gene_set)
            colors.append(ref_colors[1])
        if class2_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[2])
            #colors.append('lightlimegreen')
        else:
            if class2_fe > 0.15:
                tick_labels.append('')
            else:
                tick_labels.append('')
            print('class2')
            print(class2_pval)
            print(gene_set)
            colors.append(ref_colors[2])
        if class3_pval < thresh:
            tick_labels.append('')
            colors.append(ref_colors[3])
            #colors.append('lightblue')
        else:
            tick_labels.append('')
            print('class3')
            print(class3_pval)
            print(gene_set)
            colors.append(ref_colors[3])
        #if sibs_pval < thresh:
        #    tick_labels.append('')
        #    colors.append('black')
        #else:
        #    tick_labels.append('')
        #    colors.append(ref_colors[4])
        
    # volcano plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(len(FE)):
        if pvals[i] > -np.log10(0.05):
            ax.scatter(FE[i], pvals[i], c=colors[i], s=90, marker=shapes[i])
        else:
            ax.scatter(FE[i], pvals[i], c='white', s=90, marker=shapes[i], linewidths=1.5, edgecolors=colors[i])
    ax.set_xlabel('log2 fold change', fontsize=15)
    ax.set_ylabel('-log10(q-value)', fontsize=15)
    ax.set_title('inhMis', fontsize=18)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='High-ASD/High-Delays', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/Low-Delays', markerfacecolor='violet', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='High-ASD/Low-Delays', markerfacecolor='limegreen', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Low-ASD/High-Delays', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Siblings', markerfacecolor='black', markersize=10)]
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1)
    #ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    for i, txt in enumerate(tick_labels):
        ax.annotate(txt, (FE[i], pvals[i]), fontsize=11)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    if impute:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_6400_volcano_plot_INH_missense.png', bbox_inches='tight')
    else:
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_volcano_plot_INH_missense.png', bbox_inches='tight')
    plt.close()


def analyze_CNVs():
    file = '/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/CNV_calls_WES1_SPARK.csv'
    cnvs = pd.read_csv(file, header=0, index_col=False)
    # rename 'sample' to 'spid'
    cnvs = cnvs.rename(columns={'sample': 'spid'})
    # load gfmm labels
    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv') #6406
    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_no_bms_5714_labeled.csv') # 5714
    #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_imputed_genetic_diagnosis_labeled.csv') #6515 (109 more)
    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5392 (no bms)
    # subset to WES1 cohort
    wes_mastertable = pd.read_csv('/mnt/ceph/SFARI/SPARK/pub/iWES_v1/mastertable/SPARK.iWES_v1.mastertable.2022_02.tsv', sep='\t', header=0, index_col=None) # VERSION 1
    wes_SPIDS = wes_mastertable['spid'].tolist()
    gfmm_labels = gfmm_labels[gfmm_labels['subject_sp_id'].isin(wes_SPIDS)]
    #print(gfmm_labels.shape); exit()
    # rename 'subject_sp_id' to 'spid'
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]

    # get sibs from master table
    sibs = wes_mastertable[wes_mastertable['asd'] == 1]
    #sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt' # 2027 sibs paired
    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' # 1588 sibs paired
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    #sibs = sibs[sibs['spid'].isin(sibling_list['spid'])]
    
    # group cnvs by spid and get count of CNVs for each spid
    cnvs = cnvs.groupby('spid')['site_name'].count().reset_index()
    # make dictionary mapping spid to cnv count
    spid_to_cnvs = dict(zip(cnvs['spid'], cnvs['site_name']))
    # annotate spids with CNV count
    gfmm_labels['cnv_count'] = gfmm_labels['spid'].map(spid_to_cnvs)
    sibs['cnv_count'] = sibs['spid'].map(spid_to_cnvs)
    # fill in NaN values with 0
    gfmm_labels['cnv_count'] = gfmm_labels['cnv_count'].fillna(0)
    sibs['cnv_count'] = sibs['cnv_count'].fillna(0)
    class0 = gfmm_labels[gfmm_labels['mixed_pred'] == 0]['cnv_count'].tolist()
    class1 = gfmm_labels[gfmm_labels['mixed_pred'] == 1]['cnv_count'].tolist()
    class2 = gfmm_labels[gfmm_labels['mixed_pred'] == 2]['cnv_count'].tolist()
    class3 = gfmm_labels[gfmm_labels['mixed_pred'] == 3]['cnv_count'].tolist()
    sibs = sibs['cnv_count'].tolist()
    
    # print average CNV count for each class
    print(np.mean(class0))
    print(np.mean(class1))
    print(np.mean(class2))
    print(np.mean(class3))
    print(np.mean(sibs))

    # get pvalue comparing each class to sibs using a t-test
    class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
    class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
    class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
    class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
    # FDR CORRECTION
    # multiple testing correction
    print([class0_pval, class1_pval, class2_pval, class3_pval])
    corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval], method='fdr_bh', alpha=0.05)[1]
    print(corrected)

    # compute enrichment for each class against sibling background
    background = np.mean(sibs)
    background_all = (np.mean(class0) + np.mean(class1) + np.mean(class2) + np.mean(class3) + np.mean(sibs))/5
    class0_fe = np.mean(class0)/background
    class1_fe = np.mean(class1)/background
    class2_fe = np.mean(class2)/background
    class3_fe = np.mean(class3)/background
    sibs_fe = np.mean(sibs)/background_all

    # plot bars
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    #ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [np.mean(class0), np.mean(class1), np.mean(class2), np.mean(class3), np.mean(sibs)], color=['red', 'violet', 'limegreen', 'blue', 'black'], alpha=0.9)
    # plot fold enrichment
    ax.bar(['Low-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'High-ASD/High-Delays'], [class0_fe, class1_fe, class2_fe, class3_fe], color=['violet', 'red', 'limegreen', 'blue'], alpha=0.9)
    #ax.set_ylabel('CNV count per sample', fontsize=18)
    # rotate x-axis labels
    ax.set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('Fold enrichment', fontsize=18)
    ax.set_title('CNV enrichment across subgroups', fontsize=20)
    #for i, txt in enumerate([class0_pval, class1_pval, class2_pval, class3_pval]):
    #    if txt < 0.05:
    #        ax.annotate('p < 0.05', (i, np.mean(class0)), fontsize=8)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        # set color to black
        ax.spines[axis].set_color('black')
    fig.savefig('GFMM_WGS_Analysis_Plots/WES_CNVs_enrichment.png', bbox_inches='tight')
    plt.close()
 

def birth_pg_inf_analysis(impute=False):
    
    spids_with_inf = ['SP0034896', 'SP0108057', 'SP0116453', 'SP0220502', 'SP0245755', 'SP0260176', 'SP0336818', 'SP0368829']
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute) 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)

    # subset dnvs_pro to only include high confidence LoF variants
    dnvs_pro = dnvs_pro[(dnvs_pro['LoF'] == 1) & (dnvs_pro['LoF_flags'] == 1)]
    
    # retrieve spids_with_inf from dnvs_pro
    dnvs_pro = dnvs_pro[dnvs_pro['spid'].isin(spids_with_inf)]
    zero_pro = zero_pro[zero_pro['spid'].isin(spids_with_inf)]
    print(dnvs_pro.shape)
    print(zero_pro.shape)


def cell_marker_analysis(impute=False):
    # read in cell marker data - developing first trimester cell types discovered from 26 embryonic/fetal cells
    marker_genes = pd.read_csv('/mnt/home/alitman/ceph/Marker_Genes/616_clusters_marker_genes_processed.csv', header=0, index_col=False)
    # get list of marker genes for each cell type (AutoClass)
    # drop nan AutoClass values
    # groupby AutoClass and merge all marker genes into one list
    marker_genes = marker_genes.groupby('AutoClass')['TopLevelEnriched'].apply(lambda x: ','.join(x)).reset_index()
    gene_sets = []
    gene_set_names = []
    for i in range(len(marker_genes)):
        cell_type = str(marker_genes['AutoClass'][i]) #+ '_' + str(marker_genes.index[i])
        genes = marker_genes['TopLevelEnriched'][i].split(',')
        gene_sets.append(list(set(genes)))
        gene_set_names.append(cell_type)
    print("Number of cell types: ", len(gene_sets))
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set in gene_set_names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        corrected = [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', size='value', hue='Cluster', palette=colors, markers=markers, sizes=(100, 400), alpha=0.8)
    #for i, row in validation_subset.iterrows():
    #    if row['cluster'] == -1:
    #        ax.scatter(row['variable'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnPTVs', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_ptvs_celltype_markers_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_ptvs_celltype_markers_analysis.png', bbox_inches='tight')
    plt.close()

    # DENOVO MISSENSE VARIANTS
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set in gene_set_names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnMissense', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_missense_celltype_markers_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_missense_celltype_markers_analysis.png', bbox_inches='tight')
    plt.close()


def atlas_cell_marker_analysis(impute=False):
    # read in cell marker data - developing first trimester cell types discovered from 26 embryonic/fetal cells
    marker_genes = pd.read_csv('/mnt/home/alitman/ceph/Marker_Genes/brain_cell_markers.txt', sep='\t', header=0, index_col=False)
    # get list of marker genes for each cell type (AutoClass)
    # drop nan AutoClass values
    # groupby AutoClass and merge all marker genes into one list
    #marker_genes = marker_genes.groupby('AutoClass')['TopLevelEnriched'].apply(lambda x: ','.join(x)).reset_index()
    cell_to_genes = defaultdict(list)
    for i in range(len(marker_genes)):
        cell_type = str(marker_genes['cellName'][i])
        genes = marker_genes['cellMarker'][i].strip().split(', ')
        cell_to_genes[cell_type] += genes
    
    # convert lists to sets in cell_to_genes
    for cell in cell_to_genes:
        cell_to_genes[cell] = set(cell_to_genes[cell])
    print(cell_to_genes)
    gene_sets = list(cell_to_genes.values())
    gene_set_names = list(cell_to_genes.keys())
    print("Number of cell types: ", len(gene_sets))

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set in gene_set_names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    #for i, row in validation_subset.iterrows():
    #    if row['cluster'] == -1:
    #        ax.scatter(row['variable'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnPTVs', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_ptvs_CELLMARKER_celltype_markers_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_ptvs_CELLMARKER_celltype_markers_analysis.png', bbox_inches='tight')
    plt.close()

    # DENOVO MISSENSE VARIANTS
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set in gene_set_names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval], method='fdr_bh', alpha=0.05)[1]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,9))
    sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnMissense', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_missense_celltype_markers_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_missense_celltype_markers_analysis.png', bbox_inches='tight')
    plt.close()


def developmental_stages_cell_marker_analysis(impute=False):
    sheets = pd.ExcelFile('/mnt/home/alitman/ceph/Marker_Genes/cell_markers_developmental_stages.xlsx')
    
    gene_sets = []
    gene_set_names = []
    dev_stages = ['Res.Neonatal-Childhood', 'Res.Neonatal-Infancy',
       'Res.Neonatal-Adolescence', 'Res.Neonatal-Fetal', 'Res.Neonatal-Adult',
       'Res.Childhood-Infancy', 'Res.Childhood-Adolescence',
       'Res.Childhood-Fetal', 'Res.Childhood-Adult', 'Res.Infancy-Adolescence',
       'Res.Infancy-Fetal', 'Res.Infancy-Adult', 'Res.Adolescence-Fetal',
       'Res.Adolescence-Adult', 'Res.Fetal-Adult']
    for sheet in sheets.sheet_names:
        if sheet == 'O Number of nuclei':
            break
        df = pd.read_excel(sheets, sheet)
        # remove first letter from sheet name
        sheet = sheet[2:]
        
        #df = df[df['trend_class'].isin(['up'])]
        #gene_sets.append(df['gene_name'].tolist())
        #gene_set_names.append(sheet)
        for stage in dev_stages:
            if stage not in df.columns:
                continue
            if len(df[df[stage] != 0]) < 10:
                continue
            gene_sets.append(df[df[stage] == 1]['gene_name'].tolist())
            stage = stage.split('.')[1]
            gene_set_names.append(sheet + '_' + stage)

    print("Number of gene sets: ", len(gene_sets))
    print("Number of gene set names: ", len(gene_set_names))
    print(gene_set_names)
    # print lengths of gene sets (mean, median, min, max)
    gene_set_lengths = [len(gene_set) for gene_set in gene_sets]
    print("Mean gene set length: ", np.mean(gene_set_lengths))
    print("Median gene set length: ", np.median(gene_set_lengths))
    print("Min gene set length: ", np.min(gene_set_lengths))
    print("Max gene set length: ", np.max(gene_set_lengths))
        
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set in gene_set_names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
        #corrected = [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        # if no significant enrichment, continue
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]) < -np.log10(0.05):
            continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': 4}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df, all_pros_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue', 'purple']
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue', 'purple']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue', 4: 'purple'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays', 4: 'All Pros'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue', 4: 'purple'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays', 4: 'All Pros'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(0.05), 'gray', validation_subset['color'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,12))
    #sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    for i, row in validation_subset.iterrows():
        ax.scatter(row['Fold Enrichment'], row['variable'], s=200, c=row['color'], marker=row['marker'], alpha=0.8)

    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'All Pros']
    #handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnPTVs', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_ptvs_dev_stages_celltype_markers_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_ptvs_dev_stages_celltype_markers_analysis.png', bbox_inches='tight')
    plt.close()


def make_dev_stage_figure(impute=False):
    '''
    sheets = pd.ExcelFile('/mnt/home/alitman/ceph/Marker_Genes/cell_markers_developmental_stages.xlsx')
    # get all columns in the sheets in sheets that start with Res 
    dev_stages = []
    for sheet in sheets.sheet_names:
        if sheet == 'O Number of nuclei':
            break
        df = pd.read_excel(sheets, sheet)
        dev_stages += [col for col in df.columns if col.startswith('Res')]
    dev_stages = list(set(dev_stages))
    gene_sets = []
    gene_set_names = []
    all_stages = []
    cell_type_categories = []

    cell_to_category = {'Astro': 'Glia', 'ID2': 'Inhibitory_interneuron_CGE', 'L2-3_CUX2': 'Principal_excitatory_neuron', 'L4_RORB': 'Principal_excitatory_neuron',
                        'L5-6_THEMIS': 'Principal_excitatory_neuron', 'L5-6_TLE4': 'Principal_excitatory_neuron', 'LAMP5_NOS1': 'Inhibitory_interneuron_CGE',
                        'Micro': 'Glia', 'Oligo': 'Glia', 'OPC': 'Glia', 'PV': 'Inhibitory_interneuron_MGE', 'PV_SCUBE3': 'Inhibitory_interneuron_MGE',
                        'SST': 'Inhibitory_interneuron_MGE', 'VIP': 'Inhibitory_interneuron_CGE'}

    for stage in dev_stages:
        for sheet in sheets.sheet_names:
            if sheet == 'O Number of nuclei':
                break
            df = pd.read_excel(sheets, sheet)
            if stage not in df.columns:
                continue
            sheet = sheet[2:]
            # get both up and down regulated genes
            for trend in [1,-1]:
                if trend == 1:
                    gene_sets.append(df[df[stage] == trend]['gene_name'].tolist())
                    stage_temp = stage.split('.')[1]
                    gene_set_names.append(sheet + '_' + stage_temp)
                    all_stages.append(stage_temp)
                    cell_type_categories.append(cell_to_category[sheet])
                else:
                    gene_sets.append(df[df[stage] == trend]['gene_name'].tolist()) # grab downregulated genes
                    stage_temp = stage.split('.')[1]
                    stage_reversed = stage_temp.split('-')[1] + '-' + stage_temp.split('-')[0]
                    gene_set_names.append(sheet + '_' + stage_reversed) # stage reversed - these are now genes upregulated in the stage
                    all_stages.append(stage_reversed)
                    cell_type_categories.append(cell_to_category[sheet])
    print(len(gene_sets))
    print(len(gene_set_names))
    print(len(cell_type_categories))
    print(len(all_stages))
    
    # OPTIONAL: combine gene sets by cell type category
    new_gene_sets = []
    new_gene_set_names = []
    new_cell_type_categories = []
    new_stages = []
    for category in list(set(cell_type_categories)):
        for stage in list(set(all_stages)):
            gene_set = [] # gene set for each category and stage
            for i in range(len(gene_sets)):
                if (cell_type_categories[i] == category) and (all_stages[i] == stage):
                    gene_set += gene_sets[i]
            new_gene_sets.append(gene_set)
            new_gene_set_names.append(category + '_' + stage)
            new_cell_type_categories.append(category)
            new_stages.append(stage)
    gene_sets = new_gene_sets
    gene_set_names = new_gene_set_names
    cell_type_categories = new_cell_type_categories
    all_stages = new_stages
    print(len(gene_sets))
    print(len(gene_set_names))
    print(len(cell_type_categories))
    print(len(all_stages))

    # save gene sets, gene_set_names, cell_type_categories, all_stages to file
    with open('pickles/gene_sets.pkl', 'wb') as f:
        rick.dump(gene_sets, f)
    with open('pickles/gene_set_names.pkl', 'wb') as f:
        rick.dump(gene_set_names, f)
    with open('pickles/cell_type_categories.pkl', 'wb') as f:
        rick.dump(cell_type_categories, f)
    with open('pickles/all_stages.pkl', 'wb') as f:
        rick.dump(all_stages, f)
    '''
    # read in gene sets, gene_set_names, cell_type_categories, all_stages from file
    with open('pickles/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('pickles/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('pickles/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    with open('pickles/all_stages.pkl', 'rb') as f:
        all_stages = rick.load(f)   

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['violet', 'red', 'limegreen', 'blue', 'dimgray', 'purple']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'dimgray', 'purple']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []

    celltype_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for gene_set, stage in zip(gene_set_names, all_stages):
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh')[1]
        #corrected = [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        # if no significant enrichment, continue
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]) < -np.log10(0.05):
            continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background
        print(gene_set)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0, 'stage': stage}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1, 'stage': stage}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2, 'stage': stage}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3, 'stage': stage}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': 4, 'stage': stage}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1, 'stage': stage}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df, all_pros_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    if impute:
        colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue', 'purple']
    else:
        colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue', 'purple']
    markers = ['x', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue', 4: 'purple'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays', 4: 'All Pros'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue', 4: 'purple'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays', 4: 'All Pros'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(0.1), 'gray', validation_subset['color'])
    # map -1 to black again
    validation_subset['color'] = np.where(validation_subset['cluster'] == -1, 'black', validation_subset['color'])
    # randomly generate colors for each stage
    #validation_subset['stage_color'] = validation_subset['stage'].map({'Neonatal-Childhood': 'navy', 'Neonatal-Infancy': 'lightblue', 'Neonatal-Adolescence': 'yellow', 'Neonatal-Fetal': 'coral', 'Neonatal-Adult': 'lightlimegreen', 'Childhood-Infancy': 'darklimegreen', 'Childhood-Adolescence': 'fuchsia', 'Childhood-Fetal': 'darkviolet', 'Childhood-Adult': 'slateblue', 'Infancy-Adolescence': 'cornflowerblue', 'Infancy-Fetal': 'cyan', 'Infancy-Adult': 'gold', 'Adolescence-Fetal': 'maroon', 'Adolescence-Adult': 'rosybrown', 'Fetal-Adult': 'salmon',
    #                                                                    'Fetal-Infancy': 'olivedrab', 'Fetal-Adolescence': 'darkorange', 'Fetal-Childhood': 'sienna', 'Adult-Infancy': 'darkslategray', 'Adult-Adolescence': 'darkturquoise', 'Adult-Childhood': 'deeppink', 'Adolescence-Neonatal': 'khaki', 'Childhood-Neonatal': 'lightcoral', 'Infancy-Neonatal': 'lightsealimegreen', 'Fetal-Neonatal': 'lime', 'Adult-Neonatal': 'mediumvioletred', 'Adult-Fetal': 'pink'})
    validation_subset['stage_color'] = validation_subset['stage'].map({'Neonatal-Childhood': 'navy', 'Neonatal-Infancy': 'navy', 'Neonatal-Adolescence': 'navy', 'Neonatal-Fetal': 'navy', 'Neonatal-Adult': 'navy', 'Childhood-Infancy': 'navy', 'Childhood-Adolescence': 'navy', 'Childhood-Fetal': 'navy', 'Childhood-Adult': 'navy', 'Infancy-Adolescence': 'navy', 'Infancy-Fetal': 'navy', 'Infancy-Adult': 'navy', 'Adolescence-Fetal': 'navy', 'Adolescence-Adult': 'navy', 'Fetal-Adult': 'salmon',
                                                                        'Fetal-Infancy': 'salmon', 'Fetal-Adolescence': 'salmon', 'Fetal-Childhood': 'salmon', 'Adult-Infancy': 'navy', 'Adult-Adolescence': 'navy', 'Adult-Childhood': 'navy', 'Adolescence-Neonatal': 'navy', 'Childhood-Neonatal': 'navy', 'Infancy-Neonatal': 'navy', 'Fetal-Neonatal': 'salmon', 'Adult-Neonatal': 'navy', 'Adult-Fetal': 'navy'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,12))
    #sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', hue='color', sizes=(50, 300))
    for _, row in validation_subset.iterrows():
        #plt.scatter('stage', row['variable'], c=row['stage_color'], s=300)
        plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*200, c=row['color'])

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=45)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_6400_dev_stages_cellcat_ptvs_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_dev_stages_cellcat_ptvs_analysis.png', bbox_inches='tight')
    plt.close()


def make_go_enrichment_gene_trend_figure(impute=False):

    sheets = pd.ExcelFile('/mnt/home/alitman/ceph/Marker_Genes/go_enrichments_by_gene_trend.xlsx')

    g_to_trend = {'G1': 'up', 'G2': 'up', 'G3': 'up', 'G4': 'up', 'G5': 'trans_up', 'G6': 'trans_up', 'G7': 'trans_up', 'G8': 'trans_down', 'G9': 'trans_down',
                    'G10': 'down', 'G11': 'down', 'G12': 'down', 'G13': 'down', 'G14': 'down'}
    
    cell_to_category = {'Astro': 'Glia', 'ID2': 'Inhibitory_interneuron_CGE', 'L2-3_CUX2': 'Principal_excitatory_neuron', 'L4_RORB': 'Principal_excitatory_neuron',
                            'L5-6_THEMIS': 'Principal_excitatory_neuron', 'L5-6_TLE4': 'Principal_excitatory_neuron', 'LAMP5_NOS1': 'Inhibitory_interneuron_CGE',
                            'Micro': 'Glia', 'Oligo': 'Glia', 'OPC': 'Glia', 'PV': 'Inhibitory_interneuron_MGE', 'PV_SCUBE3': 'Inhibitory_interneuron_MGE',
                            'SST': 'Inhibitory_interneuron_MGE', 'VIP': 'Inhibitory_interneuron_CGE', 'LAMP5_CA1': 'Inhibitory_interneuron_CGE'}

    category_to_enrichment = {}
    agg_cells = defaultdict(pd.DataFrame)
    # go through every sheet, convert to trend, and for each cell type, extract top 10 enriched terms
    for sheet in sheets.sheet_names:
        df = pd.read_excel(sheets, sheet, index_col=0)
        sheet = sheet.split('-')[1].strip()
        trend = g_to_trend[sheet]
        print(sheet)
        print(trend)
        for cell in df.columns:
            category = cell_to_category[cell]
            name = category + '_' + trend
            if name not in agg_cells:
                agg_cells[name] = df[cell] # aggregate by both category and trend
            else:
                agg_cells[name] = pd.concat([agg_cells[name], df[cell]], axis=1)
    print(len(agg_cells))
    # take top 10 terms per category and trend
    for name, df in agg_cells.items():
        df = df.mean(axis=1)
        df = df.sort_values(ascending=False)
        df = df.head(10)
        # remove any terms with 0 enrichment
        df = df[df > 0]
        # remove enrichments < -np.log10(0.05)
        df = df[df > -np.log10(0.05)]
        category_to_enrichment[name] = df
   
    print(category_to_enrichment)
    # get union of all terms
    all_terms = set()
    for df in category_to_enrichment.values():
        all_terms = all_terms.union(set(df.index))

    # save category_to_enrichment to file
    with open('pickles/category_to_enrichment.pkl', 'wb') as f:
        rick.dump(category_to_enrichment, f)

def get_trend_celltype_gene_sets():
    sheets = pd.ExcelFile('/mnt/home/alitman/ceph/Marker_Genes/cell_markers_developmental_stages.xlsx')
    
    cell_to_category = {'Astro': 'Glia', 'ID2': 'Inhibitory_interneuron_CGE', 'L2-3_CUX2': 'Principal_excitatory_neuron', 'L4_RORB': 'Principal_excitatory_neuron',
                        'L5-6_THEMIS': 'Principal_excitatory_neuron', 'L5-6_TLE4': 'Principal_excitatory_neuron', 'LAMP5_NOS1': 'Inhibitory_interneuron_CGE',
                        'Micro': 'Glia', 'Oligo': 'Glia', 'OPC': 'Glia', 'PV': 'Inhibitory_interneuron_MGE', 'PV_SCUBE3': 'Inhibitory_interneuron_MGE',
                        'SST': 'Inhibitory_interneuron_MGE', 'VIP': 'Inhibitory_interneuron_CGE'}

    gene_sets = []
    gene_set_names = []
    gene_trends = ['down', 'trans_down', 'trans_up', 'up']
    trends = []
    cell_type_categories = []
    for trend in gene_trends:
        for sheet in sheets.sheet_names:
            if sheet == 'O Number of nuclei':
                break
            df = pd.read_excel(sheets, sheet)
            sheet = sheet[2:]
            
            df = df[df['trend_class'] == trend]
            name = sheet + '_' + trend
            gene_sets.append(df['gene_name'].tolist())
            gene_set_names.append(name)
            trends.append(trend)
            cell_type_categories.append(cell_to_category[sheet])
    
    # OPTIONAL: combine gene sets by cell type category (4 major cell types: PN, IN-MGE, IN-CGE, GLIA)
    new_gene_sets = []
    new_gene_set_names = []
    new_cell_type_categories = []
    new_trends = []
    for category in list(set(cell_type_categories)):
        for trend in list(set(trends)):
            gene_set = [] # gene set for each category and stage
            for i in range(len(gene_sets)):
                if (cell_type_categories[i] == category) and (trends[i] == trend):
                    gene_set += gene_sets[i]
            new_gene_sets.append(list(set(gene_set)))
            new_gene_set_names.append(category + '_' + trend)
            new_cell_type_categories.append(category)
            new_trends.append(trend)
    gene_sets = new_gene_sets
    gene_set_names = new_gene_set_names
    cell_type_categories = new_cell_type_categories
    trends = new_trends

    return gene_sets, gene_set_names, trends, cell_type_categories


def make_gene_trend_figure_inherited(impute=False, fdr=0.1):
     # rare inherited variants
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_gene_trends_rare_inherited.pkl', 'rb') as f:
    #    spid_to_num_ptvs = rick.load(f)
    #with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_gene_trends_rare_inherited_90patho.pkl', 'rb') as f:
    #    spid_to_num_missense = rick.load(f)

    # rare inherited + unobserved inherited variants
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_gene_trends_rare_inherited_noaf.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_gene_trends_rare_inherited_noaf_90patho.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    '''
    gene_sets, gene_set_names, trends, cell_type_categories = get_trend_celltype_gene_sets()
    # save as pickles
    with open('pickles/gene_sets.pkl', 'wb') as f:
        rick.dump(gene_sets, f)
    with open('pickles/gene_set_names.pkl', 'wb') as f:
        rick.dump(gene_set_names, f)
    with open('pickles/trends.pkl', 'wb') as f:
        rick.dump(trends, f)
    with open('pickles/cell_type_categories.pkl', 'wb') as f:
        rick.dump(cell_type_categories, f)
    '''
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
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' # 1588 sibs paired

    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in spid_to_class}
    print(len(pros_to_num_missense))
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in sibling_list}
    print(len(sibs_to_num_missense))

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    # LOAD DNVs
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    
    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    validation_subset = pd.DataFrame()
    prop_table = pd.DataFrame()
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        print(gene_set)
        i = gene_set_names.index(gene_set)
        # get number of PTVs for each spid in gene set
        class0 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0]
        class1 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1]
        class2 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2]
        class3 = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3]
        all_pros_data = class0 + class1 + class2 + class3
        sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]

        #total_inherited = np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3)
        total_inherited = np.sum(sibs)

        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0_dnv = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0_dnv = class0_dnv #+ zero_class0
        class1_dnv = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1_dnv = class1_dnv #+ zero_class1
        class2_dnv = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2_dnv = class2_dnv #+ zero_class2
        class3_dnv = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3_dnv = class3_dnv #+ zero_class3
        sibs_dnv = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs_dnv = sibs_dnv #+ zero_sibs['count'].astype(int).tolist()
        all_pros_data_dnv = class0_dnv + class1_dnv + class2_dnv + class3_dnv

        #total_dnvs = np.sum(class0_dnv) + np.sum(class1_dnv) + np.sum(class2_dnv) + np.sum(class3_dnv) 
        total_dnvs = np.sum(sibs_dnv)

        #print(total_dnvs)
        #print(total_inherited)

        # combine inherited and de novo PTVs (sum each spid's PTVs in gene set)
        # weight each class by the proportion of total inherited and de novo PTVs
        
        #class0 = [a/total_inherited + b/total_dnvs for a, b in zip(class0, class0_dnv)]
        #class1 = [a/total_inherited + b/total_dnvs for a, b in zip(class1, class1_dnv)]
        #class2 = [a/total_inherited + b/total_dnvs for a, b in zip(class2, class2_dnv)]
        #class3 = [a/total_inherited + b/total_dnvs for a, b in zip(class3, class3_dnv)]
        #sibs = [a/total_inherited + b/total_dnvs for a, b in zip(sibs, sibs_dnv)]
        #all_pros_data = [a/total_inherited + b/total_dnvs for a, b in zip(all_pros_data, all_pros_data_dnv)]
        
        # keep track of proportion of individuals with at least 1 PTV in gene set
        prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': len([i for i in class0 if i > 0])/num_class0, 'class1': len([i for i in class1 if i > 0])/num_class1, 'class2': len([i for i in class2 if i > 0])/num_class2, 'class3': len([i for i in class3 if i > 0])/num_class3, 'sibs': len([i for i in sibs if i > 0])/num_sibs}, index=[0])], axis=0)

        # get pvalue comparing each class to the rest
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        # get pvalue comparing each class to sibs using a t-test
        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]

        # multiple testing correction
        print([class0_pval, class1_pval, class2_pval, class3_pval])
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
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/(num_sibs)
        class0_fe = np.log2((np.sum(class0)/num_class0)/background)
        class1_fe = np.log2((np.sum(class1)/num_class1)/background)
        class2_fe = np.log2((np.sum(class2)/num_class2)/background)
        class3_fe = np.log2((np.sum(class3)/num_class3)/background)
        all_pros_fe = np.log2((np.sum(all_pros_data)/(num_class0+num_class1+num_class2+num_class3))/background)
        sibs_fe = np.log2((np.sum(sibs)/num_sibs)/background_all)

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
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'Low-ASD/Low-Delays', 3: 'High-ASD/High-Delays'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(fdr), 'gray', validation_subset['color'])
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(7,8))
    for _, row in validation_subset.iterrows():
        plt.scatter('Gene Trend', row['variable'], s=300, c=row['trend_color'])
    sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', sizes=(250, 500), hue='color', palette=colors, legend=False)
    
    # make legend outside plot
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=35, ha='right')
    plt.ylabel('')
    plt.xlabel('')
    plt.title('Rare Inherited PTVs combined with dnPTVs', fontsize=22)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_rareinherited_ptvs_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

    # plot prop table as heatmap
    prop_table = prop_table.set_index('variable')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    sns.heatmap(prop_table, cmap='coolwarm')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=45)
    plt.ylabel('')
    plt.xlabel('')
    plt.title('Proportion of individuals with at least 1 PTV (dn or inherited) in gene set', fontsize=22)
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_inherited_dn_ptvs_proportion.png', bbox_inches='tight')
    plt.close()
 

def make_gene_trend_figure(impute=False, consequence='lof', fdr=0.05):
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
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    if consequence == 'lof':
        consequences = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    elif consequence == 'missense':
        consequences = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    with open('pickles/category_to_enrichment.pkl', 'rb') as f:
            category_to_enrichment = rick.load(f)

    celltype_to_enrichment = {}
    class_to_go_enrichment = {}
    validation_subset = pd.DataFrame()
    prop_table = pd.DataFrame()
    class_to_cumulative_gene_set = defaultdict(list)
    genes_with_vars = defaultdict(list)
    genes_with_vars_classes = defaultdict(pd.DataFrame)
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        if consequence == 'lof':
            dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
            dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        elif consequence == 'missense':
            dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['am_class']
            dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        
        # print unique genes with variants in them from dnvs_pro and dnvs_sibs
        gene_set_with_vars = set(dnvs_pro[dnvs_pro['gene_set&consequence'] > 0]['name'].tolist())
        # extract genes and corresponding class from dnvs_pro
        gene_set_with_vars_class = dnvs_pro[dnvs_pro['gene_set&consequence'] > 0][['name', 'class']]
        
        gene_set_w_vars_sibs = set(dnvs_sibs[dnvs_sibs['gene_set&consequence'] > 0]['name'].tolist())
        
        gene_set_w_vars_sibs_class = dnvs_sibs[dnvs_sibs['gene_set&consequence'] > 0][['name']]
        gene_set_w_vars_sibs_class['class'] = 'sibs'

        gene_set_w_vars_all = list(set(gene_set_with_vars.union(gene_set_w_vars_sibs)))
        genes_with_vars[gene_set] = gene_set_w_vars_all

        # concat gene_set_with_vars_class and gene_set_w_vars_sibs_class
        gene_set_w_vars_all_class = pd.concat([gene_set_with_vars_class, gene_set_w_vars_sibs_class], axis=0)
        genes_with_vars_classes[gene_set] = gene_set_w_vars_all_class

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() #+ [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() #+ [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() #+ [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() #+ [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() #+ [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # keep track of proportion of individuals with at least 1 PTV in gene set
        #prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': len([i for i in class0 if i > 0])/num_class0, 'class1': len([i for i in class1 if i > 0])/num_class1, 'class2': len([i for i in class2 if i > 0])/num_class2, 'class3': len([i for i in class3 if i > 0])/num_class3, 'sibs': len([i for i in sibs if i > 0])/num_sibs}, index=[0])], axis=0)
        # keep track of total number of PTVs per group (norm by group size) in gene set
        prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': np.sum(class0)/num_class0, 'class1': np.sum(class1)/num_class1, 'class2': np.sum(class2)/num_class2, 'class3': np.sum(class3)/num_class3, 'sibs': np.sum(sibs)/num_sibs}, index=[0])], axis=0)
        # keep track of number of unique genes with PTVs in gene set
        # extract the genes
        '''
        genes = list(set(dnvs_pro[dnvs_pro[gene_set] == 1]['name'].tolist()))
        # get the number of unique genes in each class with PTVs in gene
        class0_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 0) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        class1_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 1) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        class2_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 2) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        class3_genes = len(set(dnvs_pro[(dnvs_pro['class'] == 3) & (dnvs_pro['name'].isin(genes)) & (dnvs_pro['gene_set&consequence'] > 0)]['name'].tolist()))
        sibs_genes = len(set(dnvs_sibs[(dnvs_sibs['name'].isin(genes)) & (dnvs_sibs['gene_set&consequence'] > 0)]['name'].tolist()))
        prop_table = pd.concat([prop_table, pd.DataFrame({'variable': gene_set, 'class0': class0_genes/num_class0, 'class1': class1_genes/num_class1, 'class2': class2_genes/num_class2, 'class3': class3_genes/num_class3, 'sibs': sibs_genes/num_sibs}, index=[0])], axis=0)
        '''
        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, alternative='greater')[1]
        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh')[1]
        print(corrected)
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        # UNCOMMENT TO SKIP INSIGNIFICANT FEATURES:
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]) < -np.log10(fdr):
            continue

        # get enrichment terms for this gene_set and trend
        enrichment = category_to_enrichment[category + '_' + trend]
        # look for which classes are significant
        if class0_pval > -np.log10(fdr):
            # nested dictionary
            class_to_go_enrichment[category + '.' + trend + '.class0'] = enrichment
            genes_w_ptvs = dnvs_pro[(dnvs_pro['class'] == 0) & (dnvs_pro['gene_set&consequence'] == 1)]['name'].to_list()
            class_to_cumulative_gene_set[0] += genes_w_ptvs
        if class1_pval > -np.log10(fdr):
            class_to_go_enrichment[category + '.' + trend + '.class1'] = enrichment
            genes_w_ptvs = dnvs_pro[(dnvs_pro['class'] == 1) & (dnvs_pro['gene_set&consequence'] == 1)]['name'].to_list()
            class_to_cumulative_gene_set[1] += genes_w_ptvs
        if class2_pval > -np.log10(fdr):
            class_to_go_enrichment[category + '.' + trend + '.class2'] = enrichment
            genes_w_ptvs = dnvs_pro[(dnvs_pro['class'] == 2) & (dnvs_pro['gene_set&consequence'] == 1)]['name'].to_list()
            class_to_cumulative_gene_set[2] += genes_w_ptvs
        if class3_pval > -np.log10(fdr):
            class_to_go_enrichment[category + '.' + trend + '.class3'] = enrichment
            genes_w_ptvs = dnvs_pro[(dnvs_pro['class'] == 3) & (dnvs_pro['gene_set&consequence'] == 1)]['name'].to_list()
            class_to_cumulative_gene_set[3] += genes_w_ptvs
        if sibs_pval > -np.log10(fdr):
            class_to_go_enrichment[category + '.' + trend + '.Siblings'] = enrichment
            genes_w_ptvs = dnvs_sibs[dnvs_sibs['gene_set&consequence'] == 1]['name'].to_list()
            class_to_cumulative_gene_set[-1] += genes_w_ptvs
        if all_pros_pval > -np.log10(fdr):
            class_to_go_enrichment[category + '.' + trend + '.AllProbands'] = enrichment
        
        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 3, 'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 2, 'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 1, 'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 0, 'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': -1, 'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -2, 'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
    
    '''
    # THIS IS FOR EXTRACTING THE ACTUAL GENES THAT ARE IMPACTED FOR EACH GROUP+TREND
    # convert genes_with_vars to a dataframe
    # orient to columns and complete with nan
    genes_with_vars = pd.DataFrame.from_dict(genes_with_vars, orient='index').reset_index()
    genes_with_vars = genes_with_vars.T
    genes_with_vars.columns = genes_with_vars.iloc[0]
    print(genes_with_vars.shape)
    genes_with_vars.to_csv('/mnt/home/alitman/SPARK_genomics/devDEGS_genes_with_vars.csv', index=False)
    
    # get genes_with_vars_classes into a large dataframe
    genes_with_vars_classes = pd.concat(genes_with_vars_classes.values(), axis=0)
    print(genes_with_vars_classes)
    genes_with_vars_classes.to_csv('/mnt/home/alitman/SPARK_genomics/devDEGS_genes_with_vars_classes.csv', index=False)
    '''

    # save class_to_go_enrichment to file
    #with open('pickles/class_to_go_enrichment_FDR10.pkl', 'wb') as f:
    #    rick.dump(class_to_go_enrichment, f)
    #print(class_to_go_enrichment)
    #exit()

    #for i in range(len(class_to_cumulative_gene_set.keys())):
    #    print(i)
    #    print(' '.join(list(set(class_to_cumulative_gene_set[i]))))

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
    
    # rename 'variable' column
    validation_subset['variable'] = validation_subset['variable'].map({'Principal_excitatory_neuron_down': 'Principal Excitatory Neuron: Down',
                                                                        'Inhibitory_interneuron_MGE_down': 'Inhibitory Interneuron MGE: Down',
                                                                        'Inhibitory_interneuron_CGE_down': 'Inhibitory Interneuron CGE: Down',
                                                                        'Glia_down': 'Glia: Down',
                                                                        'Principal_excitatory_neuron_trans_down': 'Principal Excitatory Neuron: Trans Down',
                                                                        'Inhibitory_interneuron_MGE_trans_down': 'Inhibitory Interneuron MGE: Trans Down',
                                                                        'Inhibitory_interneuron_CGE_trans_down': 'Inhibitory Interneuron CGE: Trans Down',
                                                                        'Glia_trans_down': 'Glia: Trans Down',
                                                                        'Principal_excitatory_neuron_trans_up': 'Principal Excitatory Neuron: Trans Up',
                                                                        'Inhibitory_interneuron_MGE_trans_up': 'Inhibitory Interneuron MGE: Trans Up',
                                                                        'Inhibitory_interneuron_CGE_trans_up': 'Inhibitory Interneuron CGE: Trans Up',
                                                                        'Glia_trans_up': 'Glia: Trans Up',
                                                                        'Principal_excitatory_neuron_up': 'Principal Excitatory Neuron: Up',
                                                                        'Inhibitory_interneuron_MGE_up': 'Inhibitory Interneuron MGE: Up',
                                                                        'Inhibitory_interneuron_CGE_up': 'Inhibitory Interneuron CGE: Up',
                                                                        'Glia_up': 'Glia: Up'})

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'purple', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'purple', 'blue', 'limegreen', 'violet', 'red']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'blue', 1: 'limegreen', 2: 'red', 3: 'violet'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'High-ASD/High-Delays', 3: 'Low-ASD/Low-Delays'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'blue', 1: 'limegreen', 2: 'red', 3: 'violet'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'ASD-Developmentally Delayed', 1: 'ASD-Social/RRB', 2: 'ASD-Higher Support Needs', 3: 'ASD-Lower Support Needs'})
    #validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(fdr), 'gray', validation_subset['color'])
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    #sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', hue='color', sizes=(50, 300))
    for _, row in validation_subset.iterrows():
        #plt.scatter('trend', row['variable'], c=row['trend_color'], s=300)
        if row['value'] < -np.log10(fdr):
            plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*220, c='white', linewidth=2.5, edgecolors=row['color'], alpha=0.9)
        else:
            plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*220, c=row['color']) #

    # get legend for bubble sizes
    for i in range(5):
        plt.scatter([], [], s=(i+1)*220, c='gray', label=str(i+1))
    #plt.scatter([], [], s=440, c='white', edgecolors='black', linewidth=2.5, label='Not significant')
    plt.legend(scatterpoints=1, labelspacing=1.1, title='Fold Enrichment', title_fontsize=23, fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
    

    # make legend outside plot
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Cluster', title_fontsize=14, fontsize=14)
    # make y and x ticklabels larger
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=20, rotation=35, ha='right')
    #plt.title('dnPTVs', fontsize=22)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    
    # color y labels
    ylabel_colors = ['navy', 'navy', 'navy', 'navy', 'cornflowerblue', 'gold', 'gold', 'coral']
    for i, label in enumerate(ax.get_yticklabels()):
        print(i, label)
        label.set_color(ylabel_colors[i])
        # make bold
        label.set_fontweight('bold')
    plt.ylabel('')
    
    if consequence == 'lof':
        if impute:
            plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_6400_gene_trends_ptvs_analysis.png', bbox_inches='tight')
        else:
            plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_ptvs_analysis.png', bbox_inches='tight')
    elif consequence == 'missense':
        if impute:
            plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_6400_gene_trends_missense_analysis.png', bbox_inches='tight')
        else:
            plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_missense_analysis.png', bbox_inches='tight')
    plt.close()

    # plot prop table as heatmap
    prop_table = prop_table.set_index('variable')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    sns.heatmap(prop_table, cmap='PuRd')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=18, rotation=45)
    plt.ylabel('')
    plt.xlabel('')
    #plt.title('Proportion of individuals with at least 1 dnPTV in gene set', fontsize=22)
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_ptvs_proportion.png', bbox_inches='tight')
    plt.close()


def make_gene_trend_figure_COMBINED(impute=False, fdr=0.05):    
    # read the pickles
    with open('pickles/gene_sets.pkl', 'rb') as f:
        gene_sets = rick.load(f)
    with open('pickles/gene_set_names.pkl', 'rb') as f:
        gene_set_names = rick.load(f)
    with open('pickles/trends.pkl', 'rb') as f:
        trends = rick.load(f)
    with open('pickles/cell_type_categories.pkl', 'rb') as f:
        cell_type_categories = rick.load(f)
    
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['lof_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['lof_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_pro['missense_consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['missense_consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_set_names)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()
    all_spids = num_class0 + num_class1 + num_class2 + num_class3

    with open('pickles/category_to_enrichment.pkl', 'rb') as f:
            category_to_enrichment = rick.load(f)

    celltype_to_enrichment = {}
    class_to_go_enrichment = {}
    validation_subset = pd.DataFrame()
    prop_table = pd.DataFrame()
    class_to_cumulative_gene_set = defaultdict(list)
    genes_with_vars = defaultdict(list)
    genes_with_vars_classes = defaultdict(pd.DataFrame)
    for gene_set, trend, category in zip(gene_set_names, trends, cell_type_categories):
        dnvs_pro['lof_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['lof_consequence'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
        dnvs_sibs['lof_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['lof_consequence'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        dnvs_pro['mis_gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['missense_consequence'] * dnvs_pro['am_class']
        dnvs_sibs['mis_gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['missense_consequence'] * dnvs_sibs['am_class']
    
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0_lof = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['lof_gene_set&consequence'].sum().tolist() 
        class0_mis = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = [sum(x) for x in zip(class0_lof, class0_mis)] + zero_class0
        class1_lof = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class1_mis = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = [sum(x) for x in zip(class1_lof, class1_mis)] + zero_class1
        class2_lof = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class2_mis = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = [sum(x) for x in zip(class2_lof, class2_mis)] + zero_class2
        class3_lof = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['lof_gene_set&consequence'].sum().tolist()
        class3_mis = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = [sum(x) for x in zip(class3_lof, class3_mis)] + zero_class3
        sibs_lof = dnvs_sibs.groupby('spid')['lof_gene_set&consequence'].sum().tolist() 
        sibs_mis = dnvs_sibs.groupby('spid')['mis_gene_set&consequence'].sum().tolist()
        sibs = [sum(x) for x in zip(sibs_lof, sibs_mis)] + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        sibs_rest_of_sample = class0 + class1 + class2 + class3
        class0_pval = ttest_ind(class0, sibs, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, alternative='greater')[1]
        print(gene_set)
        print([class0_pval, class1_pval, class2_pval, class3_pval])
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh')[1]
        print(corrected)
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        # UNCOMMENT TO SKIP INSIGNIFICANT FEATURES:
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]) < -np.log10(fdr):
            continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        all_pros_fe = (np.sum(all_pros_data)/all_spids)/background
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])

        class0_df = pd.DataFrame({'variable': gene_set, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 3, 'trend': trend}, index=[0])
        class1_df = pd.DataFrame({'variable': gene_set, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 2, 'trend': trend}, index=[0])
        class2_df = pd.DataFrame({'variable': gene_set, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 1, 'trend': trend}, index=[0])
        class3_df = pd.DataFrame({'variable': gene_set, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 0, 'trend': trend}, index=[0])
        all_pros_df = pd.DataFrame({'variable': gene_set, 'value': all_pros_pval, 'Fold Enrichment': all_pros_fe, 'cluster': -1, 'trend': trend}, index=[0])
        sibs_df = pd.DataFrame({'variable': gene_set, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -2, 'trend': trend}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, all_pros_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        celltype_to_enrichment[gene_set] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
    
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
    print(validation_subset.head(35))

    # BUBBLE PLOT
    if impute:
        colors = ['black', 'purple', 'violet', 'red', 'limegreen', 'blue']
    else:
        colors = ['black', 'purple', 'blue', 'limegreen', 'violet', 'red']
    markers = ['x', 'o', 'o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({-2: 'x', -1: 'o', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    # rename cluster labels
    if impute:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'blue', 1: 'limegreen', 2: 'red', 3: 'violet'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'High-ASD/High-Delays', 3: 'Low-ASD/Low-Delays'})
    else:
        validation_subset['color'] = validation_subset['cluster'].map({-2: 'black', -1: 'purple', 0: 'blue', 1: 'limegreen', 2: 'red', 3: 'violet'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-2: 'Siblings', -1: 'All Probands', 0: 'Low-ASD/High-Delays', 1: 'High-ASD/Low-Delays', 2: 'High-ASD/High-Delays', 3: 'Low-ASD/Low-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(fdr), 'gray', validation_subset['color'])
    validation_subset['trend_color'] = validation_subset['trend'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    #sns.scatterplot(data=validation_subset, x='Cluster', y='variable', size='Fold Enrichment', hue='color', sizes=(50, 300))
    for _, row in validation_subset.iterrows():
        #plt.scatter('trend', row['variable'], c=row['trend_color'], s=300)
        plt.scatter(row['Cluster'], row['variable'], s=row['Fold Enrichment']*220, c=row['color']) #

    # get legend for bubble sizes
    for i in range(5):
        plt.scatter([], [], s=(i+1)*200, c='gray', label=str(i+1))
    plt.legend(scatterpoints=1, labelspacing=1.1, title='Fold Enrichment', title_fontsize=22, fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
    
    # make legend outside plot
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Cluster', title_fontsize=14, fontsize=14)
    # make y and x ticklabels larger
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=20, rotation=35, ha='right')
    #plt.title('dnPTVs', fontsize=22)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')

    # color y labels
    '''
    ylabel_colors = ['navy', 'navy', 'navy', 'navy', 'cornflowerblue', 'gold', 'gold', 'coral']
    for i, label in enumerate(ax.get_yticklabels()):
        print(i, label)
        label.set_color(ylabel_colors[i])
        # make bold
        label.set_fontweight('bold')
    plt.ylabel('')
    '''

    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_gene_trends_ptvs_combined_missense_analysis.png', bbox_inches='tight')
    plt.close()


def make_gene_trend_go_term_figure(impute=False):

    with open('pickles/class_to_go_enrichment_FDR10.pkl', 'rb') as f:
        class_to_go_enrichment = rick.load(f)

    # how many unique terms across all values?
    unique_terms = set()
    for key, value in class_to_go_enrichment.items():
        unique_terms = unique_terms.union(set(value.index))
    print(len(unique_terms))

    # make a dataframe with all terms and their enrichments for each class
    df = pd.DataFrame(index=list(unique_terms))
    for key, value in class_to_go_enrichment.items():
        df[key] = value
    print(df.head)

    # take transpose of df
    df = df.T
    print(df.head)

    # order columns by GO term category
    column_order = ['trans-synaptic signaling', 'chemical synaptic transmission', 'cell-cell signaling', 'cell communication', 'anterograde trans-synaptic signaling',
       'metal ion transport', 'synaptic signaling', 'regulation of membrane potential', 'signaling', 'inorganic ion transmembrane transport', 'neurogenesis', 'generation of neurons',
       'neuron differentiation', 'neuron development', 'neuron projection development', 'regulation of plasma membrane bounded cell projection organization', 
       'multicellular organism development', 'developmental process', 'nervous system development', 'anatomical structure development', 'system development',
       'cellular process', 'regulation of biological process',
       'multicellular organismal process', 'cellular aromatic compound metabolic process', 'cellular metabolic process', 'nucleic acid metabolic process',
       'metabolic process', 'organic substance metabolic process',
       'cellular nitrogen compound metabolic process', 'organic cyclic compound metabolic process', 'mitotic cell cycle process', 'primary metabolic process',
       'nitrogen compound metabolic process', 'monovalent inorganic cation transport', 'regulation of cellular process', 'biological regulation']

    # group together
    group_terms = [['trans-synaptic signaling', 'chemical synaptic transmission', 'anterograde trans-synaptic signaling', 'synaptic signaling'], ['cell-cell signaling', 'metal ion transport', 'regulation of membrane potential', 'inorganic ion transmembrane transport', 'monovalent inorganic cation transport'],
                    ['cell communication', 'signaling'], ['multicellular organism development', 'developmental process', 'nervous system development', 'anatomical structure development', 'system development'],
                    ['cellular process', 'regulation of biological process', 'regulation of cellular process', 'biological regulation', 'regulation of plasma membrane bounded cell projection organization', 'mitotic cell cycle process', 'multicellular organismal process'],
                    ['cellular aromatic compound metabolic process', 'cellular metabolic process', 'nucleic acid metabolic process', 'metabolic process', 'organic substance metabolic process', 'cellular nitrogen compound metabolic process', 'organic cyclic compound metabolic process', 'primary metabolic process', 'nitrogen compound metabolic process'],
                    ['neurogenesis', 'generation of neurons', 'neuron differentiation', 'neuron development', 'neuron projection development']]
    group_terms_new_names = ['synaptic signaling/transmission', 'cell signaling and transport', 'cell communication', 'development', 'regulation of cellular/biological process', 'metabolic process', 'neurogenesis and neuron development']
    
    cols = [x for x in column_order if x in df.columns]
    df = df[cols]
    #df = df[column_order]

    for group, name in zip(group_terms, group_terms_new_names):
        group = [x for x in group if x in df.columns]
        df[f'{name}'] = df[group].mean(axis=1)
        # drop old columns
        df = df.drop(columns=group)
    
    column_order = ['synaptic signaling/transmission', 'cell signaling and transport',
       'development', 'regulation of cellular/biological process',
       'neurogenesis and neuron development'
       ]
    
    df = df[column_order]

    #x_label_colors = ['black', 'purple', 'purple', 'purple', 'purple', 'purple',
    #   'purple', 'purple', 'purple', 'purple', 'purple', 'darkorange', 'darkorange',
    #   'darkorange', 'darkorange', 'darkorange', 'darkorange', 
    #   'turquoise', 'turquoise', 'turquoise', 'turquoise', 'turquoise',
    #   'peru', 'peru',
    #   'peru', 'peru', 'peru', 'peru',
    #   'peru', 'peru',
    #   'peru', 'peru', 'peru', 'peru',
    #   'peru', 'peru', 'peru', 'peru']
    x_label_colors = ['black', 'purple',
       'turquoise', 'tomato', 'peru', 'darkorange'
       ]

    print(len(column_order))
    print(len(x_label_colors))

    # get list of classes from 26 rows
    categories = df.index.tolist()
    classes = [x.split('.')[-1] for x in categories]
    trends = [x.split('.')[1] for x in categories]
    celltypes = [x.split('.')[0] for x in categories]
    print(categories)
    print(classes)
    print(trends)

    df['classes'] = classes
    df['cluster'] = df['classes'].map({'class0': 0, 'class1': 1, 'class2': 2, 'class3': 3, 'Siblings': -2, 'AllProbands': -1})
    df['trends'] = trends
    df['celltypes'] = celltypes

    # make one plot for each celltype
    ct = np.unique(celltypes)
    for celltype in ct:
        copy_df = df.copy()
        print(copy_df.shape)

        copy_df = copy_df[copy_df['celltypes'] == celltype]
        print(copy_df.shape)

        # drop -1 cluster and AllProbands rows
        copy_df = copy_df[copy_df['classes'] != 'AllProbands']
        copy_df = copy_df[copy_df['cluster'] != -1]

        # sort by class
        copy_df = copy_df.sort_values(by='cluster')
        print(copy_df.columns)

        # BUBBLE PLOT
        if impute:
            colors = ['black', 'purple', 'violet', 'red', 'limegreen', 'blue']
        else:
            colors = ['black', 'purple', 'red', 'violet', 'limegreen', 'blue']
        copy_df['trend_color'] = copy_df['trends'].map({'down': 'navy', 'trans_down': 'lightblue', 'trans_up': 'yellow', 'up': 'coral'})
        copy_df['cell_color'] = copy_df['celltypes'].map({'Glia': 'lightcoral', 'Inhibitory_interneuron_CGE': 'lightsealimegreen', 'Principal_excitatory_neuron': 'mediumvioletred', 'Inhibitory_interneuron_MGE': 'khaki'})
        copy_df['class_color'] = copy_df['cluster'].map({-2: colors[0], -1: colors[1], 0: colors[2], 1: colors[3], 2: colors[4], 3: colors[5]})
        copy_df = copy_df.fillna(0)
        
        print(df.columns)

        plt.style.use('seaborn-v0_8-whitegrid')
        #fig, ax = plt.subplots(figsize=(25,8.5)) # for all terms
        fig, ax = plt.subplots(figsize=(14,7.3)) # for grouped terms
        for _, row in copy_df.iterrows():
            #plt.scatter('class', row.name, s=250, c=row['class_color'])
            plt.scatter('trend', row.name, s=300, c=row['trend_color'])
            #plt.scatter('celltype', row.name, s=250, c=row['cell_color'])
            for i in range(len(row[:-7])):
                plt.scatter(i+1, row.name, s=row[i]*180, c=row['class_color'])
            
        # make xticklabels and offset by 1
        # first tick label = trend
        plt.xticks(range(0, len(row[:-7])+1), ['trend'] + list(row.index[:-7]), fontsize=18, rotation=35, ha='right')
        plt.yticks(fontsize=18)

        for label, color in zip(ax.get_xticklabels(), x_label_colors):
            label.set_color(color)

        plt.title(f'{celltype}', fontsize=26)

        # legend with bubble size
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='-log10(p-value)', title_fontsize=14, fontsize=14)

        # make borders thicker
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black')
        
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_{celltype}_gene_trends_go_term_analysis.png', bbox_inches='tight')
        plt.close()


def process_reactome():
    reactome = '/mnt/home/alitman/ceph/Marker_Genes/Reactome_2022.gmt'
    gene_sets = []
    go_terms = []
    with open(reactome, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_set_name = parts[0]
            gene_set_genes = parts[2:]
            gene_sets.append(gene_set_genes)
            go_terms.append(gene_set_name)
    
    return gene_sets, go_terms


def process_kegg():
    kegg = '/mnt/home/alitman/ceph/Marker_Genes/KEGG_2021_Human.gmt'
    gene_sets = []
    go_terms = []
    with open(kegg, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_set_name = parts[0]
            gene_set_genes = parts[2:]
            gene_sets.append(gene_set_genes)
            go_terms.append(gene_set_name)

    return gene_sets, go_terms


def reactome_analysis(impute=False):
    gene_sets, go_terms = process_reactome()
    #gene_sets, go_terms = process_kegg()

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute) # DENOVO VARIANTS
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    # annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(go_terms)):
        dnvs_pro[go_terms[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[go_terms[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    tick_labels = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []
    
    # for each GO_term, count number of PTVs for each SPID which fall within genes in that GO_term
    go_term_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for go_term in go_terms:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[go_term] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[go_term] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
        #corrected = [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]) < -np.log10(0.05):
            continue
        #if go_term != 'Pathways of neurodegeneration':
        #    continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(go_term)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])
        #if np.max([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]) < 5:
        #    continue

        class0_df = pd.DataFrame({'variable': go_term, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': go_term, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': go_term, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': go_term, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': go_term, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        go_term_to_enrichment[go_term] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    markers = ['x', 'o', 'o', 'o', 'o']
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(0.05), 'gray', validation_subset['color'])
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,12))
    for i, row in validation_subset.iterrows():
        ax.scatter(row['Fold Enrichment'], row['variable'], s=220, c=row['color'], marker=row['marker'], alpha=0.8)
    #sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    for i, row in validation_subset.iterrows():
        if row['cluster'] == -1:
            ax.scatter(row['Fold Enrichment'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    #handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnPTVs', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_plot_ptvs_REACTOME_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_plot_ptvs_REACTOME_analysis.png', bbox_inches='tight')
    plt.close()
    exit()

    # PLOT dnMISSENSE VARIANTS
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute) # DENOVO VARIANTS
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
        
    # annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(go_terms)):
        dnvs_pro[go_terms[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[go_terms[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    FE = []
    pvals = []
    ref_colors_notimputed = ['red', 'violet', 'limegreen', 'blue', 'black']
    ref_colors_imputed = ['violet', 'red', 'limegreen', 'blue', 'black']
    if impute:
        ref_colors = ref_colors_imputed
    else:
        ref_colors = ref_colors_notimputed
    colors = []
    
    # for each GO_term, count number of PTVs for each SPID which fall within genes in that GO_term
    go_term_to_enrichment = {}
    validation_subset = pd.DataFrame()
    for go_term in go_terms:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[go_term] * dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[go_term] * dnvs_sibs['consequence'] * dnvs_sibs['am_class']
        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
        class0 = dnvs_pro[dnvs_pro['class'] == 0].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class0 = zero_pro[zero_pro['mixed_pred'] == 0]['count'].astype(int).tolist()
        class0 = class0 + zero_class0
        class1 = dnvs_pro[dnvs_pro['class'] == 1].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class1 = zero_pro[zero_pro['mixed_pred'] == 1]['count'].astype(int).tolist()
        class1 = class1 + zero_class1
        class2 = dnvs_pro[dnvs_pro['class'] == 2].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class2 = zero_pro[zero_pro['mixed_pred'] == 2]['count'].astype(int).tolist()
        class2 = class2 + zero_class2
        class3 = dnvs_pro[dnvs_pro['class'] == 3].groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        zero_class3 = zero_pro[zero_pro['mixed_pred'] == 3]['count'].astype(int).tolist()
        class3 = class3 + zero_class3
        sibs = dnvs_sibs.groupby('spid')['gene_set&consequence'].sum().tolist() + [1]
        sibs = sibs + zero_sibs['count'].astype(int).tolist()
        all_pros_data = class0 + class1 + class2 + class3

        # get p-values comparing each class to sibs using a t-test
        class0_rest_of_sample = class1 + class2 + class3 + sibs
        class1_rest_of_sample = class0 + class2 + class3 + sibs
        class2_rest_of_sample = class0 + class1 + class3 + sibs
        class3_rest_of_sample = class0 + class1 + class2 + sibs 
        sibs_rest_of_sample = class0 + class1 + class2 + class3

        class0_pval = ttest_ind(class0, sibs, equal_var=False, alternative='greater')[1]
        class1_pval = ttest_ind(class1, sibs, equal_var=False, alternative='greater')[1]
        class2_pval = ttest_ind(class2, sibs, equal_var=False, alternative='greater')[1]
        class3_pval = ttest_ind(class3, sibs, equal_var=False, alternative='greater')[1]
        sibs_pval = ttest_ind(sibs, sibs_rest_of_sample, equal_var=False, alternative='greater')[1]
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # multiple testing correction
        corrected = multipletests([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval], method='fdr_bh', alpha=0.05)[1]
        #corrected = [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval, all_pros_pval]
        class0_pval = -np.log10(corrected[0])
        class1_pval = -np.log10(corrected[1])
        class2_pval = -np.log10(corrected[2])
        class3_pval = -np.log10(corrected[3])
        sibs_pval = -np.log10(corrected[4])
        all_pros_pval = -np.log10(corrected[5])
        if np.max([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]) < -np.log10(0.05):
            continue
        #if go_term != 'Pathways of neurodegeneration':
        #    continue

        background_all = (np.sum(class0) + np.sum(class1) + np.sum(class2) + np.sum(class3) + np.sum(sibs))/(num_class0 + num_class1 + num_class2 + num_class3 + num_sibs)
        background = np.sum(sibs)/num_sibs # add psuedocount
        class0_fe = (np.sum(class0)/num_class0)/background
        class1_fe = (np.sum(class1)/num_class1)/background
        class2_fe = (np.sum(class2)/num_class2)/background
        class3_fe = (np.sum(class3)/num_class3)/background
        sibs_fe = (np.sum(sibs)/num_sibs)/background_all
        print(go_term)
        print([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe])
        print([class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval])
        #if np.max([class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]) < 5:
        #    continue

        class0_df = pd.DataFrame({'variable': go_term, 'value': class0_pval, 'Fold Enrichment': class0_fe, 'cluster': 0}, index=[0])
        class1_df = pd.DataFrame({'variable': go_term, 'value': class1_pval, 'Fold Enrichment': class1_fe, 'cluster': 1}, index=[0])
        class2_df = pd.DataFrame({'variable': go_term, 'value': class2_pval, 'Fold Enrichment': class2_fe, 'cluster': 2}, index=[0])
        class3_df = pd.DataFrame({'variable': go_term, 'value': class3_pval, 'Fold Enrichment': class3_fe, 'cluster': 3}, index=[0])
        sibs_df = pd.DataFrame({'variable': go_term, 'value': sibs_pval, 'Fold Enrichment': sibs_fe, 'cluster': -1}, index=[0])
        validation_subset = pd.concat([validation_subset, sibs_df, class0_df, class1_df, class2_df, class3_df], axis=0)

        go_term_to_enrichment[go_term] = [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        # append fold enrichment and pvalue to lists
        FE += [class0_fe, class1_fe, class2_fe, class3_fe, sibs_fe]
        pvals += [class0_pval, class1_pval, class2_pval, class3_pval, sibs_pval]

    # BUBBLE PLOT
    markers = ['x', 'o', 'o', 'o', 'o']
    if impute:
        colors = ['black', 'violet', 'red', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        colors = ['black', 'red', 'violet', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({-1: 'black', 0: 'red', 1: 'violet', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({-1: 'Siblings', 0: 'High-ASD/High-Delays', 1: 'Low-ASD/Low-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(0.05), 'gray', validation_subset['color'])
    validation_subset['marker'] = validation_subset['cluster'].map({-1: 'x', 0: 'o', 1: 'o', 2: 'o', 3: 'o'})

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9,12))
    for i, row in validation_subset.iterrows():
        ax.scatter(row['Fold Enrichment'], row['variable'], s=220, c=row['color'], marker=row['marker'], alpha=0.8)
    #sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.8, ax=ax)
    for i, row in validation_subset.iterrows():
        if row['cluster'] == -1:
            ax.scatter(row['Fold Enrichment'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
    plt.xlabel('Fold Enrichment', fontsize=20)
    plt.ylabel('')
    plt.yticks(fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    #handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
    plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'dnMissense', fontsize=24)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    if impute:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_6400_plot_missense_REACTOME_analysis.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_BUBBLE_plot_missense_REACTOME_analysis.png', bbox_inches='tight')
    plt.close()


def compute_odds_ratios(impute=False):
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    print(gene_set_names)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    satterstrom_gene_set = []
    gene_set_of_interest = 'sfari_genes'
    for i in range(len(gene_sets)):
        if gene_set_names[i] == gene_set_of_interest:
            print(gene_sets[i][:-3])
            satterstrom_gene_set += gene_sets[i][:-3]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    print(len(satterstrom_gene_set))

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    dnvs_pro['final_consequence'] = dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags'] # filter to high confidence LoF variants
    dnvs_sibs['final_consequence'] = dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
    
    odds_ratios = []
    class_to_gene_set = {}
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        print(class_id)
        
        # FOR THE ENTIRE GENE SET
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(satterstrom_gene_set)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['final_consequence'] == 1]
        case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
        case_variant_absent_count = class_count - case_variant_present_count

        # get genes with >0 LoF PTVs for class_id
        # how many unique spids have a LoF PTV in the gene set?
        spids = lof_gene_vars_for_class['spid'].unique()
        
        clinical_lab_results = pd.read_csv(f'{BASE_PHENO_DIR}/clinical_lab_results-2022-06-03.csv')
        res_spids = clinical_lab_results['subject_sp_id'].unique().tolist()

        # get intersection betwen spids and res_spids
        # Convert spids and res_spids to sets
        spids_set = set(spids)
        res_spids_set = set(res_spids)

        # Find the intersection
        intersection_spids = spids_set.intersection(res_spids_set)

        print(f'intersection spids: {len(intersection_spids)}')

        print(f'num in class: {class_count}')
        print(f'spid count: {case_variant_present_count}')
        class_to_gene_set[class_id] = lof_gene_vars_for_class['name'].unique()

        # now siblings
        gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'].isin(satterstrom_gene_set)]
        lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['final_consequence'] == 1]
        sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique()
        sibs_case_variant_absent_count = num_sibs - sibs_case_variant_present_count

        table = [[case_variant_present_count, case_variant_absent_count],
                    [sibs_case_variant_present_count, sibs_case_variant_absent_count]]

        # compute odds ratio
        odds_ratio, _ = fisher_exact(table)
        odds_ratios.append(odds_ratio)
        
    # plot bar plot for odds ratios
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6,8))
    plt.bar([0,0.5,1,1.5], odds_ratios, color=['violet', 'red', 'limegreen', 'blue'], width=0.35, alpha=0.8, linewidth=2, edgecolor='black')
    plt.xticks([0,0.5,1,1.5], ['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=19, rotation=30, ha='right')
    plt.yticks(fontsize=22)
    plt.ylabel('Odds Ratio', fontsize=30)
    plt.title(f'dnPTVs', fontsize=30)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_odds_ratios_{gene_set_of_interest}_LoF_PTVs.png', bbox_inches='tight')
    plt.close()

    # analyze class_to_gene_set
    # print number of genes in each class
    for class_id in [0,1,2,3]:
        print(class_id)
        print(len(class_to_gene_set[class_id]))
        print(' '.join(class_to_gene_set[class_id]))
    # print intersection between classes 2 and 3
    print(set(class_to_gene_set[2]).intersection(set(class_to_gene_set[3])))

    intersection = set(class_to_gene_set[2]).intersection(set(class_to_gene_set))
    class2_genes = set(class_to_gene_set[2]) - intersection
    class3_genes = set(class_to_gene_set[3]) - intersection


def compute_odds_ratio_per_gene(impute=False):
    # 1. LoF PTV in Satterstrom genes

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    
    gene_sets, gene_set_names = get_gene_sets()
    print(gene_set_names)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    satterstrom_gene_set = []
    gene_set_of_interest = 'lof_genes'
    for i in range(len(gene_sets)):
        if gene_set_names[i] == gene_set_of_interest:
            print(gene_sets[i][:-3])
            satterstrom_gene_set += gene_sets[i][:-3]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    print(len(satterstrom_gene_set))

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    # we will have to iterate through every group, then every gene and compute odds ratios for every class and gene
    class_to_odds_ratios = defaultdict(list) # map class to list of odds ratios (one for each gene)

    dnvs_pro['final_consequence'] = dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags'] # filter to high confidence LoF variants
    dnvs_sibs['final_consequence'] = dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']
    
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        print(class_id)
        
        for gene in satterstrom_gene_set:
            # compute case_variant_present (number of individuals in class with a LoF variant in the gene)
            gene_vars = dnvs_pro[dnvs_pro['name'] == gene]
            gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
            lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['final_consequence'] == 1]
            case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
            case_variant_absent_count = class_count - case_variant_present_count
        
            # now siblings
            gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'] == gene]
            lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['final_consequence'] == 1]
            sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique() + 1
            sibs_case_variant_absent_count = num_sibs - sibs_case_variant_present_count 

            table = [[case_variant_present_count, case_variant_absent_count], 
                     [sibs_case_variant_present_count, sibs_case_variant_absent_count]]
            
            # compute odds ratio
            odds_ratio, _ = fisher_exact(table)
            
            if odds_ratio > 0:
                class_to_odds_ratios[class_id].append(odds_ratio)
        
    # print mean of each class
    for class_id in [0,1,2,3]:
        print(np.std(class_to_odds_ratios[class_id])) 

    # Convert the nested dictionary to a DataFrame
    # Find the length of the longest list in class_to_odds_ratios
    max_length = max(len(v) for v in class_to_odds_ratios.values())

    # Pad shorter lists with np.nan
    for class_id in class_to_odds_ratios:
        length = len(class_to_odds_ratios[class_id])
        if length < max_length:
            class_to_odds_ratios[class_id] = class_to_odds_ratios[class_id] + [np.nan] * (max_length - length)

    # Now you can convert class_to_odds_ratios to a DataFrame
    df = pd.DataFrame(class_to_odds_ratios)
    df = df.reset_index().melt(id_vars='index', var_name='class', value_name='odds_ratio')
    print(df.head(25))

    # plot boxplots for each class of odds ratios per gene
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='class', y='odds_ratio', data=df, palette=['violet', 'red', 'limegreen', 'blue'])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    plt.title(f'Odds Ratios per Gene for LoF PTVs in {gene_set_of_interest}', fontsize=24)
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_odds_ratios_per_gene_{gene_set_of_interest}.png', bbox_inches='tight')


def compute_odds_ratios_missense(impute=False, synonymous=False, add_lof=True):

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
    # subset to target Consequences 
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']
    consequences_benign = ['synonymous_variant']
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']

    # annotate dnvs_pro and dnvs_sibs with consequence (binary)
    if synonymous:
        dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
        dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_benign else 0)
    else:
        dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
        dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_missense else 0)
        dnvs_pro['consequence_lof'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
        dnvs_sibs['consequence_lof'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
        
    gene_sets, gene_set_names = get_gene_sets()
    print(gene_set_names)
    
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    satterstrom_gene_set = []
    gene_set_of_interest = ['all_genes'] # 'sfari_genes1', 'sfari_genes2'
    for i in range(len(gene_sets)):
        if gene_set_names[i] in gene_set_of_interest:
            print(gene_sets[i][:-3])
            satterstrom_gene_set += gene_sets[i][:-3]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    print(len(satterstrom_gene_set))

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    # we will have to iterate through every group, then every gene and compute odds ratios for every class and gene
    class_to_odds_ratios = defaultdict(list) # map class to list of odds ratios (one for each gene)

    if synonymous:
        dnvs_pro['final_consequence'] = dnvs_pro['consequence']
        dnvs_sibs['final_consequence'] = dnvs_sibs['consequence']
    else:
        dnvs_pro['final_consequence'] = dnvs_pro['consequence'] * dnvs_pro['am_class']
        dnvs_sibs['final_consequence'] = dnvs_sibs['consequence'] * dnvs_pro['am_class']
        dnvs_pro['final_consequence_lof'] = dnvs_pro['consequence_lof'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
        dnvs_sibs['final_consequence_lof'] = dnvs_sibs['consequence_lof'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']
        
    odds_ratios = []
    class_to_gene_set = {}
    class_to_gene_set_log_fold_change = defaultdict(dict)
    num_mutations = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        print(class_id)
        # FOR THE ENTIRE GENE SET
        gene_vars = dnvs_pro[dnvs_pro['name'].isin(satterstrom_gene_set)]
        gene_vars_for_class = gene_vars[gene_vars['class'] == class_id]
        mis_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['final_consequence'] == 1]
        case_variant_present_count = mis_gene_vars_for_class['spid'].nunique()
        case_variant_absent_count = class_count - case_variant_present_count

        # now siblings
        gene_vars_sibs = dnvs_sibs[dnvs_sibs['name'].isin(satterstrom_gene_set)]
        mis_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['final_consequence'] == 1]
        sibs_case_variant_present_count = mis_gene_vars_sibs['spid'].nunique()
        sibs_case_variant_absent_count = num_sibs - sibs_case_variant_present_count

        # get the counts for lof and sum
        if not synonymous:
            lof_gene_vars_for_class = gene_vars_for_class[gene_vars_for_class['final_consequence_lof'] == 1]
            lof_case_variant_present_count = lof_gene_vars_for_class['spid'].nunique()
            lof_gene_vars_sibs = gene_vars_sibs[gene_vars_sibs['final_consequence_lof'] == 1]
            lof_sibs_case_variant_present_count = lof_gene_vars_sibs['spid'].nunique()
        else:
            lof_case_variant_present_count = 0
            lof_sibs_case_variant_present_count = 0

        # extend num_mutations to include number of mutations for each person
        num_missense = mis_gene_vars_for_class['spid'].value_counts().tolist()
        num_spids = mis_gene_vars_for_class['spid'].nunique()
        # get average number of missense per spid
        print(np.sum(num_missense)/num_spids)
        num_mutations.extend(mis_gene_vars_for_class['spid'].value_counts().tolist())
        if add_lof:
            num_mutations.extend(lof_gene_vars_for_class['spid'].value_counts().tolist())
        
        if add_lof:
            total_case_variant_present_count = case_variant_present_count + lof_case_variant_present_count
            total_case_variant_absent_count = class_count - total_case_variant_present_count
            total_sibs_case_variant_present_count = sibs_case_variant_present_count + lof_sibs_case_variant_present_count
            total_sibs_case_variant_absent_count = num_sibs - total_sibs_case_variant_present_count
        else:
            total_case_variant_present_count = case_variant_present_count
            total_case_variant_absent_count = case_variant_absent_count
            total_sibs_case_variant_present_count = sibs_case_variant_present_count
            total_sibs_case_variant_absent_count = sibs_case_variant_absent_count

        # get fold change for class over sibs for each gene 
        table = [[total_case_variant_present_count, total_case_variant_absent_count],
                    [total_sibs_case_variant_present_count, total_sibs_case_variant_absent_count]]

        # compute odds ratio
        odds_ratio, _ = fisher_exact(table)
        odds_ratios.append(odds_ratio)

        if not synonymous:
            spids = []
            genes = []
            if add_lof:
                spids.extend(mis_gene_vars_for_class['spid'].unique())
                spids.extend(lof_gene_vars_for_class['spid'].unique())
                genes.extend(mis_gene_vars_for_class['name'].unique())
                genes.extend(lof_gene_vars_for_class['name'].unique())
            else:
                spids.extend(mis_gene_vars_for_class['spid'].unique())
                genes.extend(mis_gene_vars_for_class['name'].unique())
            
            spids = list(set(spids))
            class_to_gene_set[class_id] = list(set(genes))
            print(f'spids: {len(spids)}')

            clinical_lab_results = pd.read_csv(f'{BASE_PHENO_DIR}/clinical_lab_results-2022-06-03.csv')
            res_spids = clinical_lab_results['subject_sp_id'].unique().tolist()

            # get intersection betwen spids and res_spids
            # Convert spids and res_spids to sets
            spids_set = set(spids)
            res_spids_set = set(res_spids)

            # Find the intersection
            intersection_spids = spids_set.intersection(res_spids_set)

            print(f'intersection spids: {len(intersection_spids)}')


    # print mean of each class
    for class_id in [0,1,2,3]:
        print(np.mean(class_to_odds_ratios[class_id])) 
    
    # plot bar plot for odds ratios
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6,8))
    plt.bar([0,0.5,1,1.5], odds_ratios, color=['violet', 'red', 'limegreen', 'blue'], width=0.35, alpha=0.8, linewidth=2, edgecolor='black')
    plt.xticks([0,0.5,1,1.5], ['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=19, rotation=30, ha='right')
    plt.yticks(fontsize=22)
    plt.ylabel('Odds Ratio', fontsize=27)
    if synonymous:
        plt.title(f'dnSynonymous', fontsize=30)
    else:
        plt.title(f'dnPTVs + dnMissense', fontsize=30)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    if synonymous:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_odds_ratios_{gene_set_of_interest}_synonymous.png', bbox_inches='tight')
    else:
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_odds_ratios_{gene_set_of_interest}_missense.png', bbox_inches='tight')
    plt.close()

    # analyze class_to_gene_set
    # print number of genes in each class
    for class_id in [0,1,2,3]:
        print(class_id)
        print(len(class_to_gene_set[class_id]))
        print(' '.join(class_to_gene_set[class_id]))
    # print intersection between classes 2 and 3
    intersection = set(class_to_gene_set[2]).intersection(set(class_to_gene_set[3]))
    print(f'length of intersection: {len(intersection)}')
    class2_genes = set(class_to_gene_set[2]) - intersection
    class3_genes = set(class_to_gene_set[3]) - intersection
    print(' '.join(intersection))
    print(' '.join(class2_genes))
    print(' '.join(class3_genes))

    # plot log fold change for each gene
    for class_id in [0,1,2,3]:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8,10))
        plt.bar(class_to_gene_set_log_fold_change[class_id].keys(), class_to_gene_set_log_fold_change[class_id].values(), color='limegreen')
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=22)
        plt.ylabel('log2 Fold Change', fontsize=27)
        plt.title(f'Log Fold Change for {gene_set_of_interest} in class {class_id}', fontsize=24)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black')
        plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_log_fold_change_{gene_set_of_interest}_class_{class_id}.png', bbox_inches='tight')
        plt.close()

    # plot num_mutations as a frequency plot
    fig, ax = plt.subplots(figsize=(7,8))
    plt.hist(num_mutations, bins=range(0, 10), color='limegreen', alpha=0.8, edgecolor='black', linewidth=1.4)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Number of Mutations', fontsize=27)
    plt.ylabel('Frequency', fontsize=27)
    plt.title(f'Number of Mutations per Individual in {gene_set_of_interest}', fontsize=24)
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_num_mutations_{gene_set_of_interest}.png', bbox_inches='tight')
    plt.close()


def compute_odds_ratios_inherited(impute=False):
    # LOAD FILTERED RARE VARIANTS
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited_90patho.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    '''
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_ptvs_LOFTEE_rare_inherited_noaf.pkl', 'rb') as f:
        spid_to_num_ptvs = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spid_to_num_missense_ALPHAMISSENSE_rare_inherited_noaf_90patho.pkl', 'rb') as f:
        spid_to_num_missense = rick.load(f)
    '''
    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=False, header=0) # 6400 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=False, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt' # 1588 sibs paired
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))

    pros_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in spid_to_class}
    pros_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in spid_to_class}
    print(len(pros_to_num_missense))
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()
    sibs_to_num_ptvs = {k: v for k, v in spid_to_num_ptvs.items() if k in sibling_list}
    sibs_to_num_missense = {k: v for k, v in spid_to_num_missense.items() if k in sibling_list}
    print(len(sibs_to_num_missense))

    # get number of spids in each class from spid_to_num_ptvs
    num_class0 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 0])
    num_class1 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 1])
    num_class2 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 2])
    num_class3 = len([k for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == 3])
    num_sibs = len(sibs_to_num_ptvs)

    gene_sets, gene_set_names = get_gene_sets()
    print(gene_set_names)
    gene_set_of_interest = 'sfari_genes'

    odds_ratios = []
    class_to_gene_set = {}
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        for i in range(len(gene_sets)):
            if gene_set_names[i] == gene_set_of_interest:
                # get number of PTVs for each spid in gene set in class_id
                #class_counts = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == class_id]
                class_counts = [v[i] for k, v in pros_to_num_ptvs.items() if spid_to_class[k] == class_id]
                #sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]
                sibs = [v[i] for k, v in sibs_to_num_ptvs.items()]

                # count how many items in class_counts != 0 and == 0
                case_variant_present_count = len([i for i in class_counts if i > 0])
                case_variant_absent_count = class_count - case_variant_present_count

                sibs_case_variant_present_count = len([i for i in sibs if i > 0])
                sibs_case_variant_absent_count = num_sibs - sibs_case_variant_present_count

                table = [[case_variant_present_count, case_variant_absent_count],
                        [sibs_case_variant_present_count, sibs_case_variant_absent_count]]

                # compute odds ratio
                odds_ratio, _ = fisher_exact(table)
                odds_ratios.append(odds_ratio)
        
    # plot 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,9))
    plt.bar([0,1,2,3], odds_ratios, color=['violet', 'red', 'limegreen', 'blue'])
    plt.xticks([0,1,2,3], ['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=19)
    plt.yticks(fontsize=22)
    plt.ylabel('Odds Ratio', fontsize=27)
    plt.title(f'Inherited LoF PTVs in {gene_set_of_interest}', fontsize=24)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    plt.savefig(f'GFMM_WGS_Analysis_Plots/WES_odds_ratios_inherited_{gene_set_of_interest}_missense.png', bbox_inches='tight')
    plt.close()


def compare_constrained_with_oe(impute=False):
    constrained_file = '/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gnomad.v4.0.constraint_metrics.tsv'
    genes = pd.read_csv(constrained_file, sep='\t', header=0)
    print(genes)
    genes = genes[genes['gene'] != np.nan]
    genes = genes.drop(['transcript', 'mane_select', 'constraint_flags'], axis=1)
    # groupby gene (first column) and take average of all columns
    genes = genes.groupby('gene').mean()
    print(genes)

    # GET LOEUF AVG VALUES PER GENE
    loeuf = genes[['lof.oe_ci.upper']]
    #print(loeuf)

    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    # get gene sets
    highest = loeuf[loeuf['lof.oe_ci.upper'] <= 0.1].index.tolist()
    medium = loeuf[(loeuf['lof.oe_ci.upper'] > 0.1) & (loeuf['lof.oe_ci.upper'] <= 0.6)].index.tolist()
    low = loeuf[(loeuf['lof.oe_ci.upper'] > 0.6) & (loeuf['lof.oe_ci.upper'] < 1)].index.tolist()
    
    deciles = []
    names = []
    for i in np.arange(0.2,1,0.1):
        # get genes with loeuf between i-1 and i 
        deciles.append(loeuf[(loeuf['lof.oe_ci.upper'] >= i-0.1) & (loeuf['lof.oe_ci.upper'] < i)].index.tolist())
        names.append(i)

    # get lof consequences
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    dnvs_pro['consequence'] = dnvs_pro['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)
    dnvs_sibs['consequence'] = dnvs_sibs['Consequence'].apply(lambda x: 1 if x in consequences_lof else 0)

    gene_sets = [highest, medium, low]
    gene_set_names = ['pli_highest', 'pli_medium', 'pli_low']
    # for each gene set, annotate dnvs_pro and dnvs_sibs with gene set membership (binary)
    for i in range(len(gene_sets)):
        dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
        dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    for gene_set in ['pli_highest', 'pli_medium', 'pli_low']:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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

        # ALL PROS VS. ALL SIBS ANALYSIS
        all_pros_data = dnvs_pro.groupby('spid')['gene_set&consequence'].sum().tolist()
        all_pros_avg = np.mean(all_pros_data)
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        print(all_pros_pval)
        # plot bar of average number of PTVs per proband and siblings
        '''
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(['Siblings', 'Probands'], [np.mean(sibs), all_pros_avg], color=['black', 'purple'], alpha=0.85, edgecolor='black', width=0.7, linewidth=1)
        ax.set_ylabel('dnPTVs per sample', fontsize=14)
        if gene_set == 'pli_highest':
            ax.set_title('pLI >= 0.995', fontsize=16)
        elif gene_set == 'pli_medium':
            ax.set_title('0.5 <= pLI < 0.995', fontsize=16)
        elif gene_set == 'pli_low':
            ax.set_title('0.5 <= pLI < 0.9', fontsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color('black')
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_avg_ptvs_constrained_all_pros_vs_sibs_' + gene_set + '.png', bbox_inches='tight')
        plt.close()
        '''

        # get average number of PTVs per spid in each class
        class0_avg = np.mean(class0)
        class1_avg = np.mean(class1)
        class2_avg = np.mean(class2)
        class3_avg = np.mean(class3)
        sibs_avg = np.mean(sibs)

        # get pvalues comparing each class to siblings using a t-test
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
        print(gene_set)
        print(corrected)

        means = []
        means.append(np.mean(sibs))
        means.append(np.mean(class0))
        means.append(np.mean(class1))
        means.append(np.mean(class2))
        means.append(np.mean(class3))
        stds = []
        stds.append(np.std(sibs)/np.sqrt(num_sibs))
        stds.append(np.std(class0)/np.sqrt(num_class0))
        stds.append(np.std(class1)/np.sqrt(num_class1))
        stds.append(np.std(class2)/np.sqrt(num_class2))
        stds.append(np.std(class3)/np.sqrt(num_class3))

        # plot average number of PTVs per spid in each class
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
        if impute:
            ax.bar(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], [class1_avg, class0_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'dimgray'], alpha=0.95, edgecolor='black', width=0.6, linewidth=1)
        else:
            # make bars closer together
            ax.bar([0,1,2,3,4], [class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg], color=['violet', 'red', 'limegreen', 'blue', 'dimgray'], alpha=0.95, edgecolor='black', width=0.7, linewidth=1)
        ax.set_ylabel('dnLoF per sample', fontsize=16)
        # rotate x-axis labels
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(30)
        #    tick.set_fontsize(14)
        #    tick.set_ha('right')
        if gene_set == 'pli_highest':
            ax.set_title('LOEUF ≤ 0.25', fontsize=16)
        elif gene_set == 'pli_medium':
            ax.set_title('0.25 ≤ LOEUF < 0.6', fontsize=16)
        elif gene_set == 'pli_low':
            ax.set_title('0.6 ≤ LOEUF < 1', fontsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_OE' + gene_set + '.png', bbox_inches='tight')
        plt.close()

        # PLOT AS MEAN+S.E.
        fig, ax = plt.subplots(figsize=(6,5.5))
        # plot props as scatter
        x_values = list(np.arange(len(means)))
        print(f'x_values: {x_values}')
        y_values = means
        print(f'y values: {y_values}')
        print(f'std: {stds}')
        colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
        for i in range(len(x_values)):
            plt.errorbar(x_values[i], y_values[i], yerr=stds[i], fmt='o', color=colors[i], markersize=20)
        plt.xlabel('')
        plt.ylabel('dnLoF per offspring', fontsize=20)
        ax.set_xticks(x_values)
        #ax.set_xticklabels(['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=16, rotation=90)
        if gene_set == 'pli_highest':
            ax.set_title('LOEUF ≤ 0.25', fontsize=16)
        elif gene_set == 'pli_medium':
            ax.set_title('0.25 ≤ LOEUF < 0.6', fontsize=16)
        elif gene_set == 'pli_low':
            ax.set_title('0.6 ≤ LOEUF < 1', fontsize=16)
        ax.set_axisbelow(True)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        ax.grid(color='gray', linestyle='-', linewidth=0.5)
        fig.savefig('GFMM_WGS_Analysis_Plots/WES_OE_LOEUF_' + gene_set + '.png', bbox_inches='tight')
        plt.close()

    # PLOT DECILES
    for i in range(len(names)):
        dnvs_pro[names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in deciles[i] else 0)
        dnvs_sibs[names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in deciles[i] else 0)

    unique_data = []
    for gene_set in names:
        dnvs_pro['gene_set&consequence'] = dnvs_pro[gene_set] * dnvs_pro['consequence'] * dnvs_pro['LoF'] * dnvs_pro['LoF_flags']
        dnvs_sibs['gene_set&consequence'] = dnvs_sibs[gene_set] * dnvs_sibs['consequence'] * dnvs_sibs['LoF'] * dnvs_sibs['LoF_flags']

        # for each spid in each class, sum the number of PTVs for each gene in the gene set. if a person has no PTVs in the gene set, the sum will be 0
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

        # ALL PROS VS. ALL SIBS ANALYSIS
        all_pros_data = dnvs_pro.groupby('spid')['gene_set&consequence'].sum().tolist()
        all_pros_avg = np.mean(all_pros_data)
        all_pros_pval = ttest_ind(all_pros_data, sibs, equal_var=False, alternative='greater')[1]
        
        # get average number of PTVs per spid in each class
        class0_avg = np.mean(class0)
        class1_avg = np.mean(class1)
        class2_avg = np.mean(class2)
        class3_avg = np.mean(class3)
        sibs_avg = np.mean(sibs)
        unique_data.append([class0_avg, class1_avg, class2_avg, class3_avg, sibs_avg])

        # get pvalues comparing each class to siblings using a t-test
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

    num_data_points = len(unique_data)

    # Width of each bar
    bar_width = 0.15

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x positions for each group of bars
    x_positions = range(num_data_points)

    # Plot each group of bars
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']
    for i in range(5):
        # Compute x positions for bars in this group
        x_values = [x + i * bar_width for x in x_positions]
        
        # Extract data for this group
        y_values = [item[i] for item in unique_data]
        
        # Plot bars
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    # Set xticks and labels
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels([round(x, 2) for x in names])

    # Add legend
    #ax.legend()
    # add thick borders
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')

    # Set labels and title
    ax.set_xlabel('LOEUF Decile', fontsize=18)
    ax.set_ylabel('Average PTVs per Sample', fontsize=18)
    #ax.set_title('Bar Plot with Multiple Bars per Data Point')

    # Show plot
    plt.tight_layout()
    # Show the plot
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_OE_deciles.png', bbox_inches='tight')


def odds_ratios_sfari_figure(impute=False):
    dnvs_pro, dnvs_sibs, zero_pro, zero_sibs = load_dnvs(imputed=impute)
    
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
    gene_set_of_interest = ['fmrp_genes'] # , 'sfari_genes1', 'sfari_genes2'
    for i in range(len(gene_sets)):
        if gene_set_names[i] in gene_set_of_interest:
            satterstrom_gene_set += gene_sets[i]
            dnvs_pro[gene_set_names[i]] = dnvs_pro['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
            dnvs_sibs[gene_set_names[i]] = dnvs_sibs['name'].apply(lambda x: 1 if x in gene_sets[i] else 0)
    print(len(satterstrom_gene_set))

    # get number of spids in each class
    num_class0 = dnvs_pro[dnvs_pro['class'] == 0]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 0]['spid'].nunique()
    num_class1 = dnvs_pro[dnvs_pro['class'] == 1]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 1]['spid'].nunique()
    num_class2 = dnvs_pro[dnvs_pro['class'] == 2]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 2]['spid'].nunique()
    num_class3 = dnvs_pro[dnvs_pro['class'] == 3]['spid'].nunique() + zero_pro[zero_pro['mixed_pred'] == 3]['spid'].nunique()
    num_sibs = dnvs_sibs['spid'].nunique() + zero_sibs['spid'].nunique()

    # we will have to iterate through every group, then every gene and compute odds ratios for every class and gene
    class_to_odds_ratios = defaultdict(list) # map class to list of odds ratios (one for each gene)

    dnvs_pro['syn_final_consequence'] = dnvs_pro['synonymous']
    dnvs_sibs['syn_final_consequence'] = dnvs_sibs['synonymous']
    dnvs_pro['mis_final_consequence'] = dnvs_pro['missense'] * dnvs_pro['am_class']
    dnvs_sibs['mis_final_consequence'] = dnvs_sibs['missense'] * dnvs_pro['am_class']
    dnvs_pro['lof_final_consequence'] = dnvs_pro['lof'] * dnvs_pro['LoF'] #* dnvs_pro['LoF_flags']
    dnvs_sibs['lof_final_consequence'] = dnvs_sibs['lof'] * dnvs_sibs['LoF'] #* dnvs_sibs['LoF_flags']

    odds_ratios = defaultdict(list)
    class_to_gene_set = {}
    class_to_gene_set_log_fold_change = defaultdict(dict)
    num_mutations = []
    props = []
    pvals_lof = []
    pvals_syn = []
    for class_id, class_count in zip([0,1,2,3], [num_class0, num_class1, num_class2, num_class3]):
        print(class_id)
        # FOR THE ENTIRE GENE SET
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
        print(class_id, odds_ratio_mis)

        # compute lof, total, and syn odds ratios
        table_lof = [[lof_case_variant_present_count, lof_case_variant_absent_count],
                    [lof_sibs_case_variant_present_count, lof_sibs_case_variant_absent_count]]
        odds_ratio_lof, pval_lof = fisher_exact(table_lof)
        print(class_id, odds_ratio_lof, pval_lof)
        pvals_lof.append(pval_lof)

        # get fold change for class over sibs for each gene 
        table = [[total_case_variant_present_count, total_case_variant_absent_count],
                    [total_sibs_case_variant_present_count, total_sibs_case_variant_absent_count]]

        # compute odds ratio
        odds_ratio_total, _ = fisher_exact(table)

        table = [[syn_case_variant_present_count, syn_case_variant_absent_count],
                    [syn_sibs_case_variant_present_count, syn_sibs_case_variant_absent_count]]
        odds_ratio_syn, pval_syn = fisher_exact(table)
        print(class_id, odds_ratio_syn, pval_syn)
        pvals_syn.append(pval_syn)

        odds_ratios['lof'].append(odds_ratio_lof)
        odds_ratios['total'].append(odds_ratio_total)
        odds_ratios['syn'].append(odds_ratio_syn)
        odds_ratios['mis'].append(odds_ratio_mis)

    # FDR CORRECTION on pvals_lof and pvals_syn
    corrected_lof = multipletests(pvals_lof, method='fdr_bh', alpha=0.05)[1]
    corrected_syn = multipletests(pvals_syn, method='fdr_bh', alpha=0.05)[1]
    print(corrected_lof)
    print(corrected_syn)

    props.append(mis_sibs_case_variant_present_count/num_sibs)
    print(props)
    fig, ax = plt.subplots(figsize=(7,5))
    # plot props as scatter
    x_values = [0, 1, 2, 3, 4]
    y_values = props
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']

    plt.scatter(x_values, y_values, color=colors, s=200)
    plt.xlabel('')
    plt.ylabel('Proportion of dnMis carriers', fontsize=16)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], fontsize=16, rotation=90)
    plt.title('dnMis', fontsize=18)
    # put grid behind
    ax.set_axisbelow(True)
    # make y axis numbers larger
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('GFMM_WGS_Analysis_Plots/props_scatter.png', bbox_inches='tight')
    plt.close()

    # plot
    odds_ratios_for_plotting = [odds_ratios['lof']] # odds_ratios['mis'],
    
    # Width of each bar
    bar_width = 0.15

    # Set up the theme, figure and axis
    #plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(3, 6))

    # Set x positions for each group of bars
    x_positions = range(len(odds_ratios_for_plotting))

    # Plot each group of bars
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        # Compute x positions for bars in this group
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        
        # Extract data for this group
        y_values = [item[i] for item in odds_ratios_for_plotting]
        
        # Plot bars
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    # Set xticks and labels
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels(['dnLoF'], fontsize=22)

    # Add legend
    #ax.legend()

    # Set labels and title
    ax.set_xlabel('')
    # make y axis numbers larger
    plt.yticks(fontsize=13)
    ax.set_ylabel('Odds ratio', fontsize=18)
    ax.set_title('FMRP genes', fontsize=18)

    # make border bold
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')

    # add grid behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-')

    # Show plot
    plt.tight_layout()
    # Show the plot
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_odds_ratios_SFARI_genes_together.png', bbox_inches='tight')
    plt.close()

    # PLOT TOGETHER with SYN
    # plot
    odds_ratios_for_plotting = [odds_ratios['lof'], odds_ratios['syn']] # odds_ratios['mis'],
    # Width of each bar
    bar_width = 0.15

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(5.5, 6))

    # Set x positions for each group of bars
    x_positions = range(len(odds_ratios_for_plotting))

    # Plot each group of bars
    colors = ['violet', 'red', 'limegreen', 'blue', 'dimgray']
    for i in range(len(odds_ratios_for_plotting[0])):
        # Compute x positions for bars in this group
        x_values = [x + i * bar_width+0.075 for x in x_positions]
        # Extract data for this group
        y_values = [item[i] for item in odds_ratios_for_plotting]
        # Plot bars
        ax.bar(x_values, y_values, width=bar_width, label=f'Value {i+1}', color=colors[i])

    # Set xticks and labels
    ax.set_xticks([x + 2 * bar_width for x in x_positions])
    ax.set_xticklabels(['dnLoF', 'dnSyn'], fontsize=22)
    plt.yticks(fontsize=13)
    # Add legend
    #ax.legend()

    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Odds ratio', fontsize=18)
    ax.set_title('Satterstrom Genes', fontsize=18)

    # make border bold
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')

    # add grid behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-')

    # Show plot
    plt.tight_layout()
    # Show the plot
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_odds_ratios_wsyn_SFARI_genes_together.png', bbox_inches='tight')
    plt.close()


def GO_figure():

    num_top_terms = 3

    class2_biol_processes = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class2_top_pathways.csv')
    class2_mol_functions = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class2_top_mol_functions.csv')
    class3_biol_processes = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class3_top_pathways.csv')
    class3_mol_functions = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class3_top_mol_functions.csv')

    class0_biol_processes = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class0_top_pathways.csv')
    class0_mol_functions = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class0_top_mol_functions.csv')
    class1_biol_processes = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class1_top_pathways.csv')
    class1_mol_functions = pd.read_csv('/mnt/home/alitman/ceph/GO_enrichment_analysis/class1_top_mol_functions.csv')

    class2_biol_processes = class2_biol_processes.sort_values(by=['Fold Enrichment'], ascending=False)
    class2_biol_processes = class2_biol_processes.iloc[:num_top_terms, :]
    class2_mol_functions = class2_mol_functions.sort_values(by=['Fold Enrichment'], ascending=False)
    class2_mol_functions = class2_mol_functions.iloc[:num_top_terms, :]
    class3_biol_processes = class3_biol_processes.sort_values(by=['Fold Enrichment'], ascending=False)
    class3_biol_processes = class3_biol_processes.iloc[:num_top_terms, :]
    class3_mol_functions = class3_mol_functions.sort_values(by=['Fold Enrichment'], ascending=False)
    class3_mol_functions = class3_mol_functions.iloc[:num_top_terms, :]
    class0_biol_processes = class0_biol_processes.sort_values(by=['Fold Enrichment'], ascending=False)
    class0_biol_processes = class0_biol_processes.iloc[:num_top_terms, :]
    class0_mol_functions = class0_mol_functions.sort_values(by=['Fold Enrichment'], ascending=False)
    class0_mol_functions = class0_mol_functions.iloc[:num_top_terms, :]
    class1_biol_processes = class1_biol_processes.sort_values(by=['Fold Enrichment'], ascending=False)
    class1_biol_processes = class1_biol_processes.iloc[:num_top_terms, :]
    class1_mol_functions = class1_mol_functions.sort_values(by=['Fold Enrichment'], ascending=False)
    class1_mol_functions = class1_mol_functions.iloc[:num_top_terms, :]

    # concat class2 
    class2_enrich = pd.concat([class2_biol_processes, class2_mol_functions], axis=0)
    class3_enrich = pd.concat([class3_biol_processes, class3_mol_functions], axis=0)
    class0_enrich = pd.concat([class0_biol_processes, class0_mol_functions], axis=0)
    class1_enrich = pd.concat([class1_biol_processes, class1_mol_functions], axis=0)

    # process
    class2_enrich['Enrichment FDR'] = class2_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class3_enrich['Enrichment FDR'] = class3_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class0_enrich['Enrichment FDR'] = class0_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    class1_enrich['Enrichment FDR'] = class1_enrich['Enrichment FDR'].apply(lambda x: -np.log10(x))
    # split each 'pathway' by space and remove the first word (the go term id)
    class2_enrich['Pathway'] = class2_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class3_enrich['Pathway'] = class3_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class3_enrich = class3_enrich.replace('double-strand break repair via alternative nonhomologous end joining', 'double-strand break repair') # simplify term
    class0_enrich['Pathway'] = class0_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    class1_enrich['Pathway'] = class1_enrich['Pathway'].apply(lambda x: ' '.join(x.split()[1:]))
    
    # make a bubble plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(13.5,16))
    
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("limegreen", as_cmap=True), s=450, data=class2_enrich, edgecolor='black', linewidth=1, ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax2.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax2.set_title('ASD-Social/RRB', fontsize=28)
    ax2.tick_params(labelsize=18)
    #ax0.set_yticklabels(clean_labels_class2)
    #plt.xticks(fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_color('black')
    #plt.savefig('GFMM_WGS_Analysis_Plots/WES_GO_enrichment_class2.png', bbox_inches='tight')
    #plt.close()
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # color first 6 y ticks with dark orange, last 6 with purple
    for i in range(num_top_terms):
        ax2.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax2.get_yticklabels()[i].set_color('purple')
    
    # for class3
    # make a bubble plot for each
    #plt.style.use('seaborn-v0_8-whitegrid')
    #fig, ax = plt.subplots(figsize=(5.5,9))
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("blue", as_cmap=True), s=450, data=class3_enrich, edgecolor='black', linewidth=1, ax=ax3)
    #for _, row in class2_enrich.iterrows():
    #    plt.scatter(row['Fold Enrichment'], row['Pathway'], s=row['Enrichment FDR']*220, color='limegreen', alpha=0.8, edgecolor='black', linewidth=1)
    # legend outside plot
    handles, labels = ax3.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax3.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    ax3.set_title('ASD-Developmentally Delayed', fontsize=28)
    #ax1.set_yticklabels(clean_labels_class3)
    #plt.yticks(fontsize=18)
    #plt.xticks(fontsize=20)
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

    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("violet", as_cmap=True), s=450, data=class0_enrich, edgecolor='black', linewidth=1, ax=ax0)
    #for _, row in class2_enrich.iterrows():
    #    plt.scatter(row['Fold Enrichment'], row['Pathway'], s=row['Enrichment FDR']*220, color='limegreen', alpha=0.8, edgecolor='black', linewidth=1)
    handles, labels = ax0.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax0.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))

    ax0.set_title('ASD-Low Support Needs', fontsize=28)
    ax0.tick_params(labelsize=18)
    #ax0.set_yticklabels(clean_labels_class2)
    #plt.xticks(fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax0.spines[axis].set_linewidth(1.5)
        ax0.spines[axis].set_color('black')
    #plt.savefig('GFMM_WGS_Analysis_Plots/WES_GO_enrichment_class2.png', bbox_inches='tight')
    #plt.close()
    ax0.set_xlabel('')
    ax0.set_ylabel('')
    # color first 6 y ticks with dark orange, last 6 with purple
    for i in range(num_top_terms):
        ax0.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax0.get_yticklabels()[i].set_color('purple')
    
    sns.scatterplot(x='Fold Enrichment', y='Pathway', hue='Enrichment FDR', palette=sns.light_palette("red", as_cmap=True), s=450, data=class1_enrich, edgecolor='black', linewidth=1, ax=ax1)
    #for _, row in class2_enrich.iterrows():
    #    plt.scatter(row['Fold Enrichment'], row['Pathway'], s=row['Enrichment FDR']*220, color='limegreen', alpha=0.8, edgecolor='black', linewidth=1)
    handles, labels = ax1.get_legend_handles_labels()
    rounded_labels = [f'{float(label):.1f}' for label in labels]
    ax1.legend(handles, rounded_labels, scatterpoints=1, labelspacing=1.1, title='-log10(FDR)', title_fontsize=18, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))

    ax1.set_title('ASD-High Support Needs', fontsize=28)
    ax1.tick_params(labelsize=18)
    #ax0.set_yticklabels(clean_labels_class2)
    #plt.xticks(fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color('black')
    #plt.savefig('GFMM_WGS_Analysis_Plots/WES_GO_enrichment_class2.png', bbox_inches='tight')
    #plt.close()
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # color first 6 y ticks with dark orange, last 6 with purple
    for i in range(num_top_terms):
        ax1.get_yticklabels()[i].set_color('darkorange')
    for i in range(num_top_terms, 2*num_top_terms):
        ax1.get_yticklabels()[i].set_color('purple')
    
    # add legend: dark orange = Biological Processes, purple = Molecular functions
    #ax2.text(1.05, 0.2, 'Biological Process', fontsize=19, color='darkorange', transform=ax0.transAxes)
    #ax2.text(1.05, 0.25, 'Molecular Function', fontsize=19, color='purple', transform=ax0.transAxes)

    fig.tight_layout()
    plt.savefig('GFMM_WGS_Analysis_Plots/WES_GO_enrichment_figure.png', bbox_inches='tight')
    plt.close()
    
    
if __name__ == "__main__":
    GO_figure(); exit()
    #compare_constrained_with_oe(impute=False); exit()
    #odds_ratios_sfari_figure(impute=False); exit()
    #compute_odds_ratios_inherited(impute=True); exit()
    compute_odds_ratios_missense(impute=False, synonymous=False, add_lof=True); exit()
    #compute_odds_ratio_per_gene(impute=True); exit()
    #compute_odds_ratios(impute=False); exit()
    #make_go_enrichment_gene_trend_figure(impute=False); exit()
    #make_dev_stage_figure(impute=False); exit()
    #make_gene_trend_go_term_figure(impute=False); exit()
    #make_gene_trend_figure(impute=False, consequence='lof'); exit() # MAIN FIGURE
    #make_gene_trend_figure_COMBINED(impute=False); exit()
    #make_gene_trend_figure_inherited(impute=False); exit()
    #reactome_analysis(impute=True); exit()
    #developmental_stages_cell_marker_analysis(impute=True); exit()
    #atlas_cell_marker_analysis(); exit()
    #cell_marker_analysis(impute=False); exit()
    #birth_pg_inf_analysis(impute=True); exit()
    #get_inherited_variant_count(); exit()
    #analyze_CNVs(); exit()
    #combine_inherited_vep_files(); exit()
    #compare_constrained_gene_sets(impute=False, all_pros=False); exit()
    #get_count_burden_figure(impute=True); exit()
    #GO_term_analysis(impute=True); exit()
    #impute_probands_into_classes(); exit()
    #plot_proportions(impute=False); exit()
    #volcano_noncoding(impute=False)
    #volcano_inherited(impute=False, all_pros=False)
    #volcano_missense(impute=False, all_pros=False)
    volcano_lof(impute=False, all_pros=False); exit()
    #get_genetic_diagnoses()