

def load_AF():
    '''
    Load allele frequencies for WES variants.
    '''
    af = pd.read_csv('/mnt/ceph/SFARI/SPARK/pub/iWES_v2/variants/deepvariant/iWES_v2.deepvariant.pvcf_variants.tsv', sep='\t', header=0, index_col=None)
    af['id'] = af['chrom'].astype(str) + '_' + af['pos'].astype(str) + '_' + af['ref'].astype(str) + '_' + af['alt'].astype(str)
    af = af.drop(['chrom', 'pos', 'ref', 'alt'], axis=1)
    af['af'] = pd.to_numeric(af['af'], errors='coerce')
    return af


def load_dnvs():
    file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/VEP_most_severe_consequence_LOFTEE_DNV_calls_filtered_WES_v2.vcf' # filtered out centromeres and repeats
    dnvs = pd.read_csv(file, sep='\t', comment='#', header=0, index_col=None)
    dnvs = dnvs[['Uploaded_variation', 'Consequence', 'Gene', 'Extra']]
    dnvs['Consequence'] = dnvs['Consequence'].str.split(',').str[0]
    dnvs = dnvs.rename({'Uploaded_variation': 'id'}, axis='columns')
    ensembl_to_gene = dict(zip(ENSEMBL_TO_GENE_NAME['Gene'], ENSEMBL_TO_GENE_NAME['name']))
    dnvs['name'] = dnvs['Gene'].map(ensembl_to_gene)
    dnvs = dnvs.dropna(subset=['name'])

    # parse out extra information
    dnvs['am_class'] = dnvs['Extra'].str.extract(r'am_class=(.*?);')
    dnvs['am_class'] = dnvs['am_class'].apply(lambda x: 1 if x in ['likely_pathogenic'] else 0)
    dnvs['am_pathogenicity'] = dnvs['Extra'].str.extract(r'am_pathogenicity=([\d.]+)').astype(float)
    dnvs['am_pathogenicity'] = dnvs['am_pathogenicity'].apply(lambda x: 1 if x>=0.9 else 0)
    dnvs['LoF'] = dnvs['Extra'].str.extract(r'LoF=(.*?);')
    dnvs['LoF'] = dnvs['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
    dnvs['LoF_flags'] = dnvs['Extra'].str.extract(r'LoF_flags=(.*?);')
    dnvs['LoF_flags'] = dnvs['LoF_flags'].fillna(1)
    dnvs['LoF_flags'] = dnvs['LoF_flags'].apply(lambda x: 1 if x in ['SINGLE_EXON',1] else 0)
    dnvs = dnvs.drop('Extra', axis=1)

    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_vars.pkl', 'rb') as f:
        SPID_to_vars = rick.load(f)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/var_to_spid.pkl', 'rb') as handle:
        var_to_spid = rick.load(handle)
    
    dnvs['spid'] = dnvs['id'].map(var_to_spid)
    dnvs = dnvs.dropna(subset=['spid'])

    master = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    master = pd.read_csv(master, sep='\t')
    master = master[['spid', 'asd']]
    spid_to_asd = dict(zip(master['spid'], master['asd']))
    dnvs['asd'] = dnvs['spid'].map(spid_to_asd)
    dnvs = dnvs.dropna(subset=['asd'])
    
    dnvs_sibs = dnvs[dnvs['asd'] == 1]
    dnvs_pro = dnvs[dnvs['asd'] == 2]

    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0) # 5391 probands
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))
    dnvs_pro['class'] = dnvs_pro['spid'].map(spid_to_class)
    dnvs_pro = dnvs_pro.dropna(subset=['class'])

    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    dnvs_sibs = pd.merge(dnvs_sibs, sibling_list, how='inner', on='spid')

    # extract individuals with no DNVs
    count_file = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_DNV_count.txt'
    counts = pd.read_csv(count_file, sep='\t', index_col=False)
    counts = counts.rename(columns={'SPID': 'spid'})
    zero = counts[counts['count'] == 0]
    zero = zero.merge(master[['spid', 'asd']], on='spid')
    zero_pros = zero.merge(gfmm_labels[['spid', 'mixed_pred']], on='spid').drop('asd', axis=1)
    zero_sibs = zero.merge(sibling_list, on='spid').drop('asd', axis=1)

    return dnvs_pro, dnvs_sibs, zero_pros, zero_sibs


def get_gene_sets():
    sfari_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/SFARI_genes.csv', header=0, index_col=False).rename({'gene-symbol': 'name'}, axis='columns')
    lof_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/Constrained_PLIScoreOver0.9.bed', sep='\t', index_col=None)
    fmrp_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/FMRP_targets_Darnell2011.bed', sep='\t', index_col=None)
    asd_risk_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/ASD_risk_genes_TADA_FDR0.3.bed', sep='\t', index_col=None)
    brain_expressed_genes = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/BrainExpressed_Kang2011.bed', sep='\t', index_col=None)
    satterstrom = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/satterstrom_2020_102_ASD_genes.csv', header=0, index_col=False).rename({'gene': 'name'}, axis='columns')
    
    sfari_genes1 = list(sfari_genes[sfari_genes['gene-score'] == 1]['name'])
    sfari_genes2 = list(sfari_genes[sfari_genes['gene-score'] == 2]['name'])
    sfari_syndromic = list(sfari_genes[sfari_genes['syndromic'] == 1]['name'])
    sfari_genes = list(sfari_genes['name'])
    lof_genes = list(lof_genes['name'])
    fmrp_genes = list(fmrp_genes['name'])
    asd_risk_genes = list(asd_risk_genes['name'])
    brain_expressed_genes = list(brain_expressed_genes['name'])
    satterstrom = list(satterstrom['name'])
    all_genes = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v29.annotation.protein_coding_genes.hg38.bed', sep='\t', index_col=None, header=None)
    all_genes.columns = ['chr', 'start', 'end', 'gene', 'name', 'strand']
    all_genes = list(all_genes['name'])

    pli_table = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/pLI_table.txt', sep='\t', index_col=False)
    # get genes with pLI >= 0.995
    pli_genes_high = pli_table[pli_table['pLI'] >= 0.995]['gene'].tolist()
    # get genes with pli 0.5-0.995
    pli_genes_low = pli_table[(pli_table['pLI'] < 0.995) & (pli_table['pLI'] >= 0.5)]['gene'].tolist()

    gene_list = [all_genes, lof_genes, fmrp_genes, asd_risk_genes, brain_expressed_genes, satterstrom, sfari_genes, sfari_genes1, sfari_genes2, sfari_syndromic, pli_genes_high, pli_genes_low]
    gene_list_names = ['all_genes', 'lof_genes', 'fmrp_genes', 'asd_risk_genes','brain_expressed_genes', 'asd_coexpression_networks', 'psd_genes', 'satterstrom', 'sfari_genes', 'sfari_genes1', 'sfari_genes2', 'sfari_syndromic', 'pli_genes_highest', 'pli_genes_medium', 'ddg2p', 'liver_genes']

    return gene_list, gene_list_names
