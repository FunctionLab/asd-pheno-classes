import os
from collections import defaultdict

import pandas as pd
import numpy as np
import pickle as rick
import hail as hl

from utils import get_gene_sets


with open('gene_sets/gene_ensembl_ID_to_name.pkl', 'rb') as f:
        ENSEMBL_TO_GENE_NAME = rick.load(f)


def get_WES_trios():
    wes_spids = '../Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes_spids = pd.read_csv(wes_spids, sep='\t')
    wes_spids = wes_spids[['father', 'mother', 'spid']]
    wes_spids = wes_spids[(wes_spids['father'] != '0') & (wes_spids['mother'] != '0')]

    deepvar_dir = 'SFARI/SPARK/pub/iWES_v2/variants/deepvariant/gvcf/'
    gatk_dir = 'SFARI/SPARK/pub/iWES_v2/variants/gatk/gvcf/'
    wes_spids.columns = ['FID', 'MID', 'SPID']
    ids = wes_spids
    # check if all SPIDs are in the deepvar dir in the form of {SPID}.gvcf.gz
    for fid, mid, spid in zip(ids['FID'], ids['MID'], ids['SPID']):
        # if spid is not in any of the deepvar dirs, remove it from ids
        # add i to the directory: {deepvar_dir}{i}/{spid}.gvcf.gz
        # and check if it exists in any of those for both deepvar and gatk
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{spid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        # do the same for the gatk dir - must exist in both to be considered
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{spid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{mid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{mid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{deepvar_dir}{i}/{fid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        for i in range(0, 11):
            if os.path.exists(f'{gatk_dir}{i}/{fid}.gvcf.gz'):
                break
            elif i == 10:
                ids = ids[ids['SPID'] != spid]
        
    ids.to_csv('data/processed_spark_trios_WES2.txt', sep='\t', header=False, index=False)


def get_paired_sibs():
    file = '../Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes = pd.read_csv(file, sep='\t')
    sibs = wes[wes['asd'] == 1]
    spids_for_model = pd.read_csv('../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0) # 5280 PROBANDS
    probands = spids_for_model.index.tolist()
    sibling_spids = []
    for i, row in wes.iterrows():
        if row['spid'] in probands:
            fid = row['father']
            mid = row['mother']
            # get all siblings with FID/MID
            if fid == '0' and mid == '0':
                continue
            if fid == '0':
                siblings = sibs[sibs['mother'] == mid]['spid'].tolist()
            elif mid == '0':
                siblings = sibs[sibs['father'] == fid]['spid'].tolist()
            else:
                siblings = sibs[(sibs['father'] == fid) & (sibs['mother'] == mid)]['spid'].tolist()
            sibling_spids.extend(siblings)
    sibling_spids = list(set(sibling_spids))
    with open('../PhenotypeClasses/data/WES_5392_siblings_spids.txt', 'w') as f:
        for item in sibling_spids:
            f.write("%s\n" % item)


def process_DNVs():
    data_dir = 'data/WES_V2_data/calling_denovos_data/output/'
    subdirs = os.listdir(data_dir)
    var_to_spid = defaultdict(list) # dictionary with variant ID as key and list of SPIDs as value
    SPID_to_vars = defaultdict(list) # dictionary with SPID as key and list of variant IDs as value
    spid_to_count = defaultdict(int) # dictionary with SPID as key and number of DNVs as value
    spids = []
    missing = 0
    for subdir in subdirs:
        if os.path.exists(f'{data_dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf'):
            try:
                dnv = pd.read_csv(f'{data_dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf', sep='\t', comment='#', header=None)
                for i, row in dnv.iterrows():
                    var_id = row[2]
                    spid = str(subdir)
                    var_to_spid[var_id].append(spid)
                    SPID_to_vars[spid].append(var_id)
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
    threshold = mean + 3*sd
    # FILTER: remove SPIDs with more than 3SD DNVs above the mean
    spid_to_count = {k: v for k, v in spid_to_count.items() if v <= threshold}
    SPID_to_vars = {k: v for k, v in SPID_to_vars.items() if k in spid_to_count.keys()}

    # iterate through SPID_to_vars and remove non-singleton variants
    for spid, vars in SPID_to_vars.items():
        for var in vars:
            spids = var_to_spid[var]
            if len(spids) > 1:
                SPID_to_vars[spid].remove(var)
    for spid, vars in SPID_to_vars.items():
        spid_to_count[spid] = len(vars)

    # update var_to_spid to only include singletons
    var_to_spid = {}
    for spid, vars in SPID_to_vars.items():
        for var in vars:
            var_to_spid[var] = spid
    
    spid_to_count = pd.DataFrame.from_dict(spid_to_count, orient='index')
    spid_to_count.columns = ['count']
    spid_to_count.index.name = 'SPID'
    spid_to_count = spid_to_count.reset_index()
    spid_to_count.to_csv('data/SPID_to_DNV_count.txt', sep='\t', index=False)

    with open('data/var_to_spid.pkl', 'wb') as f:
        rick.dump(var_to_spid, f)
    with open('data/SPID_to_vars.pkl', 'wb') as f2:
        rick.dump(SPID_to_vars, f2)


def fetch_rare_vars_with_hail():
    hl.init()

    gnomad_v4 = hl.read_table('gs://gcp-public-data--gnomad/release/4.1/ht/exomes/gnomad.exomes.v4.1.sites.ht')

    rare_variants_ht = gnomad_v4.filter(gnomad_v4.freq[0].AF < 0.01)

    rare_variants_ht = rare_variants_ht.annotate(
        variant_id = hl.str("chr") + hl.str(rare_variants_ht.locus.contig) + "_" + 
                    hl.str(rare_variants_ht.locus.position) + "_" + 
                    hl.str(rare_variants_ht.alleles[0]) + "_" + 
                    hl.str(rare_variants_ht.alleles[1])
    )

    rare_variants_ht = rare_variants_ht.select('variant_id', AF=rare_variants_ht.freq[0].AF)
    rare_variants_ht = rare_variants_ht.key_by('variant_id')

    output_tsv_path = 'data/rare_variants.tsv.bgz'
    rare_variants_ht.export(output_tsv_path)
    hl.stop()


def combine_inherited_vep_files():
    rare_variants_df = pd.read_csv('data/rare_variants.tsv.bgz', sep='\t', compression='gzip')
    variant_to_af = dict(zip(rare_variants_df['variant_id'], rare_variants_df['AF']))

    directory = 'inherited_vep_predictions_plugins_filtered/' # filtered repeats + centromeres
    files = [f for f in os.listdir(directory) if f.endswith('.vcf')]
    spids = [f.split('.')[0] for f in files]

    spid_to_num_ptvs = {}
    spid_to_num_missense = {}

    gene_sets, gene_set_names = get_gene_sets()
    consequences_lof = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'transcript_ablation']
    consequences_missense = ['missense_variant', 'inframe_deletion', 'inframe_insertion', 'protein_altering_variant']

    gfmm_labels = pd.read_csv('../PhenotypeValidations/data/SPARK_SSC_combined_cohort_phenotypes.csv', index_col=False, header=0)
    gfmm_ids = gfmm_labels.iloc[:, 0].tolist()

    sibling_list = '../PhenotypeValidations/data/WES_5392_siblings_spids.txt'
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    sibling_list = sibling_list['spid'].tolist()

    gfmm_ids = gfmm_ids + sibling_list

    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    for i in range(len(files)):
        if spids[i] not in gfmm_ids:
            continue
        cols = ['Uploaded_variation', 'Location', 'Allele', 'Gene', 'Feature', 'Feature_type', 'Consequence', 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 'Existing_variation', 'Extra']
        df = pd.read_csv(directory + files[i], sep='\t', comment='#', header=None, names=cols, index_col=False)
        
        df = df[['Uploaded_variation', 'Gene', 'Consequence', 'Extra']]
        df['AF'] = df['Uploaded_variation'].map(variant_to_af)
        df = df.dropna(subset=['AF']) # only keep rare variants (af<0.01 from gnomAD exomes)
        
        # filter PTVs and missense variants with plugins LOFTEE, AM
        df['am_class'] = df['Extra'].str.extract(r'am_class=(.*?);')
        df['am_class'] = df['am_class'].apply(lambda x: 1 if x in ['likely_pathogenic'] else 0)
        df['LoF'] = df['Extra'].str.extract(r'LoF=(.*?);')
        df['LoF'] = df['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
        df = df.drop('Extra', axis=1)

        df['name'] = df['Gene'].map(ensembl_to_gene)
        
        ptv_counts = []
        missense_counts = []
        for gene_set in gene_sets:
            num_ptvs = df[(df['name'].isin(gene_set)) & (df['Consequence'].isin(consequences_lof)) & (df['LoF'] == 1)].shape[0]
            num_missense = df[(df['name'].isin(gene_set)) & (df['Consequence'].isin(consequences_missense)) & (df['am_class'] == 1)].shape[0]
            ptv_counts.append(num_ptvs)
            missense_counts.append(num_missense)
        spid_to_num_ptvs[spids[i]] = ptv_counts
        spid_to_num_missense[spids[i]] = missense_counts
        
    with open('data/spid_to_num_lof_rare_inherited_gnomad_only.pkl', 'wb') as f:
        rick.dump(spid_to_num_ptvs, f)
    with open('data/spid_to_num_missense_rare_inherited_gnomad_only.pkl', 'wb') as f:
        rick.dump(spid_to_num_missense, f)
