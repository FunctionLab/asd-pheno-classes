from data_utils import process_DNVs, get_paired_sibs, fetch_rare_vars_with_hail, combine_inherited_vep_files


if __name__ == "__main__":
    # 1. Preprocess DNVs to get var_to_spid and SPID_to_vars dictionaries
    process_DNVs()

    # 2. Extract IDs of paired siblings for analysis
    get_paired_sibs()

    # 3. Fetch rare variants with hail
    fetch_rare_vars_with_hail()

    # 4. Get rare inherited variant counts
    combine_inherited_vep_files()

    print("Done.")
