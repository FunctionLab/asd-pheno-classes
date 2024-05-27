

def get_WES_trios():
    wes_spids = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes_spids = pd.read_csv(wes_spids, sep='\t')
    wes_spids = wes_spids[['father', 'mother', 'spid']]
    wes_spids = wes_spids[(wes_spids['father'] != '0') & (wes_spids['mother'] != '0')]

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
        
    ids.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/spark_trios_WES2.txt', sep='\t', header=False, index=False)


def get_paired_sibs():
    file = '/mnt/home/nsauerwald/ceph/SPARK/Mastertables/SPARK.iWES_v2.mastertable.2023_01.tsv'
    wes = pd.read_csv(file, sep='\t')
    sibs = wes[wes['asd'] == 1]
    spids_for_model = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0) # 5280 PROBANDS
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
    with open('/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt', 'w') as f:
        for item in sibling_spids:
            f.write("%s\n" % item)


def process_DNVs():
    dir = '/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/output/'
    subdirs = os.listdir(dir)
    var_to_spid = defaultdict(list) # dictionary with variant ID as key and list of SPIDs as value
    SPID_to_vars = defaultdict(list) # dictionary with SPID as key and list of variant IDs as value
    spid_to_count = defaultdict(int) # dictionary with SPID as key and number of DNVs as value
    spids = []
    missing = 0
    for subdir in subdirs:
        if os.path.exists(f'{dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf'):
            try:
                dnv = pd.read_csv(f'{dir}{subdir}/{subdir}.glnexus.family.combined_intersection_filtered_gq_20_depth_10.vcf', sep='\t', comment='#', header=None)
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
    # FILTER: remove SPIDs with more than 3SD DNVs
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
    spid_to_count.to_csv('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_DNV_count.txt', sep='\t', index=False)

    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/var_to_spid.pkl', 'wb') as handle:
        rick.dump(var_to_spid, handle, protocol=rick.HIGHEST_PROTOCOL)
    with open('/mnt/home/alitman/ceph/WES_V2_data/calling_denovos_data/SPID_to_vars.pkl', 'wb') as f:
        rick.dump(SPID_to_vars, f, rick.HIGHEST_PROTOCOL)
