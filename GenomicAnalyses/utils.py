import pandas as pd
import pickle as rick
import numpy as np


with open('data/gene_ensembl_ID_to_name.pkl', 'rb') as f:
        ENSEMBL_TO_GENE_NAME = rick.load(f)


def load_AF():
    '''
    Load allele frequencies for WES variants.
    '''
    af = pd.read_csv('SPARK/pub/iWES_v2/variants/deepvariant/iWES_v2.deepvariant.pvcf_variants.tsv', sep='\t', header=0, index_col=None)
    af['id'] = af['chrom'].astype(str) + '_' + af['pos'].astype(str) + '_' + af['ref'].astype(str) + '_' + af['alt'].astype(str)
    af = af.drop(['chrom', 'pos', 'ref', 'alt'], axis=1)
    af['af'] = pd.to_numeric(af['af'], errors='coerce')
    return af


def load_dnvs():
    file = 'data/VEP_most_severe_consequence_LOFTEE_DNV_calls_filtered_WES_v2.vcf' # filtered out centromeres and repeats
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
    dnvs['LoF'] = dnvs['Extra'].str.extract(r'LoF=(.*?);')
    dnvs['LoF'] = dnvs['LoF'].apply(lambda x: 1 if x == 'HC' else 0)
    dnvs = dnvs.drop('Extra', axis=1)

    with open('data/SPID_to_vars.pkl', 'rb') as f:
        SPID_to_vars = rick.load(f)
    with open('data/var_to_spid.pkl', 'rb') as handle:
        var_to_spid = rick.load(handle)
    
    dnvs['spid'] = dnvs['id'].map(var_to_spid)
    dnvs = dnvs.dropna(subset=['spid'])

    master = '../Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    master = pd.read_csv(master, sep='\t')
    master = master[['spid', 'asd']]
    spid_to_asd = dict(zip(master['spid'], master['asd']))
    dnvs['asd'] = dnvs['spid'].map(spid_to_asd)
    dnvs = dnvs.dropna(subset=['asd'])
    
    dnvs_sibs = dnvs[dnvs['asd'] == 1]
    dnvs_pro = dnvs[dnvs['asd'] == 2]

    gfmm_labels = pd.read_csv('../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=False, header=0)
    
    gfmm_labels = gfmm_labels.rename(columns={'subject_sp_id': 'spid'})
    gfmm_labels = gfmm_labels[['spid', 'mixed_pred']]
    spid_to_class = dict(zip(gfmm_labels['spid'], gfmm_labels['mixed_pred']))
    dnvs_pro['class'] = dnvs_pro['spid'].map(spid_to_class)
    dnvs_pro = dnvs_pro.dropna(subset=['class'])

    sibling_list = '../PhenotypeValidations/data/WES_5392_siblings_spids.txt'
    sibling_list = pd.read_csv(sibling_list, sep='\t', header=None)
    sibling_list.columns = ['spid']
    dnvs_sibs = pd.merge(dnvs_sibs, sibling_list, how='inner', on='spid')

    # extract individuals with no DNVs
    count_file = 'data/SPID_to_DNV_count.txt'
    counts = pd.read_csv(count_file, sep='\t', index_col=False)
    counts = counts.rename(columns={'SPID': 'spid'})
    zero = counts[counts['count'] == 0]
    zero = zero.merge(master[['spid', 'asd']], on='spid')
    zero_pros = zero.merge(gfmm_labels[['spid', 'mixed_pred']], on='spid').drop('asd', axis=1)
    zero_sibs = zero.merge(sibling_list, on='spid').drop('asd', axis=1)

    return dnvs_pro, dnvs_sibs, zero_pros, zero_sibs


def get_gene_sets():
    sfari_genes = pd.read_csv('gene_sets/SFARI_genes.csv', header=0, index_col=False).rename({'gene-symbol': 'name'}, axis='columns')
    lof_genes = pd.read_csv('gene_sets/Constrained_PLIScoreOver0.9.bed', sep='\t', index_col=None)
    chd8_genes = pd.read_csv('gene_sets/CHD8_targets_Cotney2015_Sugathan2014.bed', sep='\t', index_col=None)
    fmrp_genes = pd.read_csv('gene_sets/FMRP_targets_Darnell2011.bed', sep='\t', index_col=None)
    dd = pd.read_csv('gene_sets/Developmental_delay_DDD.bed', sep='\t', index_col=None)
    asd_risk_genes = pd.read_csv('gene_sets/ASD_risk_genes_TADA_FDR0.3.bed', sep='\t', index_col=None)
    haplo_genes = pd.read_csv('gene_sets/haploinsufficiency_hesc_2022_ST.csv', header=0, index_col=False).rename({'Symbol': 'name'}, axis='columns')
    brain_expressed_genes = pd.read_csv('gene_sets/BrainExpressed_Kang2011.bed', sep='\t', index_col=None)
    asd_coexpression_networks = pd.read_csv('gene_sets/ASD_coexpression_networks_Willsey2013.bed', sep='\t', index_col=None)
    psd_genes = pd.read_csv('gene_sets/PSD_Genes2Cognition.bed', sep='\t', index_col=None)
    satterstrom = pd.read_csv('gene_sets/satterstrom_2020_102_ASD_genes.csv', header=0, index_col=False).rename({'gene': 'name'}, axis='columns')
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
    all_genes = pd.read_csv('/mnt/home/alitman/ceph/Genome_Annotation_Files_hg38/gencode.v29.annotation.protein_coding_genes.hg38.bed', sep='\t', index_col=None, header=None)
    all_genes.columns = ['chr', 'start', 'end', 'gene', 'name', 'strand']
    all_genes = list(all_genes['name'])

    df = pd.DataFrame({'name': all_genes})
    df['all_genes'] = 1
    df['lof_genes'] = df['name'].apply(lambda x: 1 if x in lof_genes else 0)
    df['fmrp_genes'] = df['name'].apply(lambda x: 1 if x in fmrp_genes else 0)
    df['asd_risk_genes'] = df['name'].apply(lambda x: 1 if x in asd_risk_genes else 0)
    df['brain_expressed_genes'] = df['name'].apply(lambda x: 1 if x in brain_expressed_genes else 0)
    df['satterstrom'] = df['name'].apply(lambda x: 1 if x in satterstrom else 0)
    df['sfari_genes1'] = df['name'].apply(lambda x: 1 if x in sfari_genes1 else 0)
    df.to_csv('data/Supp_Table_4.csv', index=False)

    pli_table = pd.read_csv('/mnt/home/alitman/ceph/DIS_Tissue_Analysis_Variant_Sets/gene_sets/pLI_table.txt', sep='\t', index_col=False)
    pli_genes_higher = pli_table[pli_table['pLI'] >= 0.995]['gene'].tolist()
    pli_genes_lower = pli_table[(pli_table['pLI'] < 0.995) & (pli_table['pLI'] >= 0.5)]['gene'].tolist()

    gene_list = [all_genes, lof_genes, chd8_genes, fmrp_genes, dd_genes, asd_risk_genes, haplo_genes, brain_expressed_genes, asd_coexpression_networks, psd_genes, satterstrom, sfari_genes, sfari_genes1, sfari_genes2, sfari_syndromic, pli_genes_higher, pli_genes_lower]
    gene_list_names = ['all_genes', 'lof_genes', 'chd8_genes', 'fmrp_genes', 'dd_genes', 'asd_risk_genes', 'haplo_genes', 'brain_expressed_genes', 'asd_coexpression_networks', 'psd_genes', 'satterstrom', 'sfari_genes', 'sfari_genes1', 'sfari_genes2', 'sfari_syndromic', 'pli_genes_higher', 'pli_genes_lower']

    return gene_list, gene_list_names


def get_trend_celltype_gene_sets():
    sheets = pd.ExcelFile('gene_sets/cell_markers_developmental_stages.xlsx')
    
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
    
    # combine gene sets by cell type category (4 major cell types: PN, IN-MGE, IN-CGE, GLIA)
    new_gene_sets = []
    new_gene_set_names = []
    new_cell_type_categories = []
    new_trends = []
    for category in list(set(cell_type_categories)):
        for trend in list(set(trends)):
            gene_set = []
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


def pad_lists(dict_of_lists):
    max_len = max(len(lst) for lst in dict_of_lists.values())
    padded_dict = {k: lst + [np.nan] * (max_len - len(lst)) for k, lst in dict_of_lists.items()}
    return padded_dict
