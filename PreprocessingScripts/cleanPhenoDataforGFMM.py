import pandas as pd
import numpy as np

from latent_class_analysis import split_columns


def get_main_spark_data_for_GFMM():
    '''CONSTRUCT SPARK PHENOTYPE MATRIX FOR GFMM CLUSTERING MODEL.'''
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'

    clinical_lab_results = pd.read_csv(f'{BASE_PHENO_DIR}/clinical_lab_results-2022-06-03.csv')
    clinical_lab_results = clinical_lab_results.set_index('subject_sp_id', drop=True)

    ### BASIC MEDICAL SCREENING
    bmsdf = pd.read_csv(f'{BASE_PHENO_DIR}/basic_medical_screening_2022-12-12.csv')
    bmsdf = bmsdf.set_index('subject_sp_id',drop=True)
    bmsdf = bmsdf.loc[(bmsdf['age_at_eval_years'] <= 18) & (bmsdf['age_at_eval_years'] >= 4)]
    
    ### OPTIONAL: convert age_at_eval_years to a dummy variable
    # separate into 3 groups: 0 (9 and under), 1 (10-14), 2 (14-18)
    #age_to_categ = {4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:1, 11:1, 12:1, 13:1, 14:2, 15:2, 16:2, 17:2, 18:2}
    #bmsdf = bmsdf.replace({'age_at_eval_years': age_to_categ})

    bmsdf = bmsdf.drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'age_at_eval_months', 'eval_year','current_depend_adult','gest_age','gen_test','gen_dx1_self_report','gen_dx2_self_report',
                                                            'gen_dx_oth_calc_self_report','gen_test_aut_dd','gen_test_cgh_cma','gen_test_chrom_karyo','gen_test_ep','gen_test_fish_angel','gen_test_fish_digeorge','gen_test_fish_williams','gen_test_fish_oth',
                                                            'gen_test_frax','gen_test_id','gen_test_mecp2','gen_test_nf1','gen_test_noonan','gen_test_pten','gen_test_tsc','gen_test_unknown','gen_test_wes','gen_test_wgs','gen_test_oth_calc','med_cond_birth',
                                                            'med_cond_birth_def','med_cond_growth','med_cond_neuro','med_cond_visaud','prev_study_agre','prev_study_asc','prev_study_charge','prev_study_earli','prev_study_marbles','prev_study_mssng','prev_study_seed',
                                                            'prev_study_ssc','prev_study_vip','sleep_eat_toilet','prev_study_calc','basic_medical_measure_validity_flag'],axis=1).fillna(0)
    colswithstrings = bmsdf.columns[bmsdf.dtypes == 'string']
    bmsdf = bmsdf.replace({'Male':1, 'Female':0})
    bmsdf['sex'] = bmsdf['sex'].astype(int)
    bmsdf['asd'] = bmsdf['asd'].astype(int)

    ### SCQ
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv', header=0, index_col=None)
    scqdf = scqdf.loc[(scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) & (scqdf['age_at_eval_years'] >= 4)] # 
    scqdf = scqdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','current_depend_adult','age_at_eval_months','scq_measure_validity_flag','eval_year','missing_values','summary_score', 'q01_phrases'],axis=1) 
    scqdf = scqdf.replace({'Male':1, 'Female':0})
    scqdf['sex'] = scqdf['sex'].astype(int)

    ### BACKGROUND HISTORY
    bhcdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_child_2022-12-12.csv') ## asd child
    bhsdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv') ## sibling
    
    bhcdf = bhcdf.loc[(bhcdf['age_at_eval_years'] <= 18)&(bhcdf['age_at_eval_years'] >= 4)]
    bhsdf = bhsdf.loc[(bhsdf['age_at_eval_years'] <= 18)&(bhsdf['age_at_eval_years'] >= 4)]
    
    bhcdf = bhcdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','sex','current_depend_adult','age_at_eval_months','age_at_eval_years'], axis=1)
    bhsdf = bhsdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','sex','current_depend_adult','age_at_eval_months','age_at_eval_years'], axis=1)

    bhdf = pd.concat([bhcdf, bhsdf], join='inner') ## concat ASD and sibling
    bhdf = bhdf.drop(['hand', 'survey_version','eval_year','gender','child_lives_with','child_lives_with_v2','mother_highest_education','father_highest_education','annual_household_income','zygosity','twin_asd','twin_partic','twin_mult_birth','bghx_validity_flag','child_grade_school','sped_y_n'],axis=1)
    bhdf = bhdf[~bhdf.index.duplicated(keep=False)]

    ### RBSR (repetitive behaviors)
    rbsr = pd.read_csv(f'{BASE_PHENO_DIR}/rbsr_2022-12-12.csv')
    rbsr = rbsr.loc[(rbsr['age_at_eval_years'] <= 18) & (rbsr['missing_values'] < 1) & (rbsr['age_at_eval_years'] >= 4)] # 
    rbsr = rbsr.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'asd', 'current_depend_adult', 'age_at_eval_months', 'age_at_eval_years', 'rbsr_validity_flag', 'overall_score', 'overall_number_items', 'total_final_score',
                                                        'missing_values', 'eval_year'], axis=1)

    ### CBCL 2
    cbcl_1 = pd.read_csv(f'{BASE_PHENO_DIR}/cbcl_1_5_2022-12-12.csv')
    cbcl_1 = cbcl_1.loc[(cbcl_1['age_at_eval_years'] <= 18) & (cbcl_1['age_at_eval_years'] >= 4)]
    cbcl_1 = cbcl_1.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'current_depend_adult', 'asd', 'age_at_eval_months', 'age_at_eval_years', 'cbcl_validity_flag', 'emotionally_reactive_raw_score', 'emotionally_reactive_percentile', 'emotionally_reactive_range', 'anxious_depressed_raw_score', 'anxious_depressed_percentile', 'anxious_depressed_range', 'somatic_complaints_raw_score', 'somatic_complaints_percentile', 'somatic_complaints_range', 'withdrawn_raw_score', 'withdrawn_percentile', 'withdrawn_range', 'sleep_problems_raw_score', 'sleep_problems_percentile', 'sleep_problems_range', 'attention_problems_raw_score',
                                                            'attention_problems_percentile', 'attention_problems_range', 'aggressive_behavior_raw_score', 'aggressive_behavior_percentile', 'aggressive_behavior_range', 'internalizing_problems_raw_score', 'internalizing_problems_percentile', 'internalizing_problems_range', 'externalizing_problems_raw_score', 'externalizing_problems_percentile', 'externalizing_problems_range', 'total_problems_raw_score', 'total_problems_percentile', 'total_problems_range', 'stress_problems_raw_score', 'stress_problems_percentile', 'stress_problems_range', 'other_problems_raw_score', 'dsm5_depressive_problems_raw_score', 'dsm5_depressive_problems_percentile',
                                                            'dsm5_depressive_problems_range', 'dsm5_anxiety_problems_raw_score', 'dsm5_anxiety_problems_percentile', 'dsm5_anxiety_problems_range',
                                                            'dsm5_autism_spectrum_problems_raw_score', 'dsm5_autism_spectrum_problems_percentile', 'dsm5_autism_spectrum_problems_problems_range', 'dsm5_attention_deficit_hyperactivity_raw_score', 'dsm5_attention_deficit_hyperactivity_percentile', 'dsm5_attention_deficit_hyperactivity_range', 'dsm5_oppositional_defiant_raw_score', 'dsm5_oppositional_defiant_percentile', 'dsm5_oppositional_defiant_range'], axis=1)
    #cbcl_1 = cbcl_1[['']]
    cbcl_2 = pd.read_csv(f'{BASE_PHENO_DIR}/cbcl_6_18_2022-12-12.csv')
    #cbcl_2 = cbcl_2.loc[(cbcl_2['age_at_eval_years'] <= 18) & (cbcl_2['age_at_eval_years'] >= 4)]
    cbcl_2 = cbcl_2.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'current_depend_adult', 'asd', 'age_at_eval_months', 'age_at_eval_years', 'cbcl_validity_flag', 'q056_h_other',
                                                            'anxious_depressed_raw_score', 'anxious_depressed_percentile', 'anxious_depressed_range', 'withdrawn_depressed_raw_score', 'withdrawn_depressed_percentile', 'withdrawn_depressed_range',
                                                            'somatic_complaints_raw_score', 'somatic_complaints_percentile', 'somatic_complaints_range', 'social_problems_raw_score', 'social_problems_percentile', 'social_problems_range', 'thought_problems_raw_score',
                                                            'thought_problems_percentile', 'thought_problems_range', 'attention_problems_raw_score', 'attention_problems_percentile', 'attention_problems_range', 'rule_breaking_behavior_raw_score', 'rule_breaking_behavior_percentile',
                                                            'rule_breaking_behavior_range', 'aggressive_behavior_raw_score', 'aggressive_behavior_percentile', 'aggressive_behavior_range', 'internalizing_problems_raw_score', 'internalizing_problems_percentile', 'internalizing_problems_range',
                                                            'externalizing_problems_raw_score', 'externalizing_problems_percentile', 'externalizing_problems_range', 'total_problems_raw_score', 'total_problems_percentile', 'total_problems_range', 'other_problems_raw_score', 'obsessive_compulsive_problems_raw_score', 'obsessive_compulsive_problems_percentile', 'obsessive_compulsive_problems_range',
                                                            'sluggish_cognitive_tempo_raw_score', 'sluggish_cognitive_tempo_percentile', 'sluggish_cognitive_tempo_range', 'stress_problems_raw_score', 'stress_problems_percentile', 'stress_problems_range', 'dsm5_conduct_problems_raw_score', 'dsm5_conduct_problems_percentile', 'dsm5_conduct_problems_range', 'dsm5_somatic_problems_raw_score',
                                                            'dsm5_somatic_problems_percentile', 'dsm5_somatic_problems_range', 'dsm5_oppositional_defiant_raw_score', 'dsm5_oppositional_defiant_percentile', 'dsm5_oppositional_defiant_range', 'dsm5_attention_deficit_hyperactivity_raw_score', 'dsm5_attention_deficit_hyperactivity_percentile', 'dsm5_attention_deficit_hyperactivity_range',
                                                            'dsm5_anxiety_problems_raw_score', 'dsm5_anxiety_problems_percentile', 'dsm5_anxiety_problems_range', 'dsm5_depressive_problems_raw_score', 'dsm5_depressive_problems_percentile', 'dsm5_depressive_problems_range', 'reading_eng_language', 'history_social_studies', 'arithmetic_math', 'science'], axis=1)
    
    # CONVERT CATEGORICAL VARIABLES TO NUMERICAL VARIABLES
    cbcl_2 = cbcl_2.replace('above_average', 0)
    cbcl_2 = cbcl_2.replace('average', 1)
    cbcl_2 = cbcl_2.replace('below_average', 2)
    cbcl_2 = cbcl_2.replace('failing', 3)
    # close_friends
    cbcl_2 = cbcl_2.replace('none', 3)
    cbcl_2 = cbcl_2.replace('1', 2)
    cbcl_2 = cbcl_2.replace('2_3', 1)
    cbcl_2 = cbcl_2.replace('4_more', 0)
    # contact friends outside school
    cbcl_2 = cbcl_2.replace('less_1', 2)
    cbcl_2 = cbcl_2.replace('1_2', 1)
    cbcl_2 = cbcl_2.replace('3_more', 0)
    # other questions
    cbcl_2 = cbcl_2.replace('worse', 2)
    cbcl_2 = cbcl_2.replace('average', 1)
    cbcl_2 = cbcl_2.replace('better', 0)
    cbcl_2 = cbcl_2.replace('has no brothers or sisters', 1)
    cbcl_2 = cbcl_2[~cbcl_2.index.duplicated(keep=False)]
    
    ## OPTIONAL: only keep subscale scores from CBCL
    #cbcl_2 = cbcl_2[['anxious_depressed_t_score','withdrawn_depressed_t_score','social_problems_t_score','attention_problems_t_score','internalizing_problems_t_score','total_problems_t_score','obsessive_compulsive_problems_t_score','stress_problems_t_score','dsm5_attention_deficit_hyperactivity_t_score','dsm5_anxiety_problems_t_score','dsm5_depressive_problems_t_score']]
    
    ### VINELAND
    vineland = pd.read_csv(f'{BASE_PHENO_DIR}/vineland-3_2022-12-12.csv')
    vineland = vineland.loc[(vineland['age_at_eval_years'] <= 18) & (vineland['age_at_eval_years'] >= 4)]
    vineland = vineland.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'sex', 'asd', 'current_depend_adult', 'age_at_eval_months', 'age_at_eval_years', 'abc_percentile', 'communication_percentile', 'dls_percentile', 'soc_percentile', 'motor_percentile', 'lowest_sub_v', 'measure_code', 'express_raw_score', 'express_est', 'express_growth', 'express_age', 'receptive_raw_score', 'receptive_age', 'receptive_est', 'receptive_growth', 'written_raw_score', 'written_age', 'written_est', 'written_growth', 'community_raw_score', 'community_age', 'community_est', 'community_growth', 'domestic_raw_score', 'domestic_age', 'domestic_est', 'domestic_growth','personal_raw_score', 'personal_age', 'personal_est', 'personal_growth', 'coping_raw_score', 'coping_age',
                                                                'coping_est', 'coping_growth', 'interpersonal_raw_score', 'interpersonal_age', 'interpersonal_est', 'interpersonal_growth', 'pla_raw_score', 'pla_age', 'pla_est',
                                                                'pla_growth', 'fine_raw_score', 'fine_age', 'fine_est', 'fine_growth', 'gross_raw_score', 'gross_age', 'gross_est', 'gross_growth','external_raw_score', 'external_est',
                                                                'internal_raw_score', 'internal_est', 'motor_adaptive_level', 'abc_adaptive_level', 'communication_adaptive_level', 'dls_adaptive_level', 'soc_adaptive_level',
                                                                'express_v_score', 'receptive_v_score', 'written_v_score', 'community_v_score', 'domestic_v_score', 'personal_v_score', 'coping_v_score', 'interpersonal_v_score', 'pla_v_score', 'fine_v_score', 'gross_v_score', 'external_v_score', 
                                                                'internal_v_score', 'communication_standard', 'abc_standard', 'dls_standard','soc_standard'], axis=1) 
    #                                                               # 'communication_standard', 'abc_standard', 'dls_standard','soc_standard', 'motor_standard'
    # encode categorical variables:
    #vineland['abc_adaptive_level'] = vineland['abc_adaptive_level'].astype('category').cat.codes
    #vineland['communication_adaptive_level'] = vineland['communication_adaptive_level'].astype('category').cat.codes
    #vineland['dls_adaptive_level'] = vineland['dls_adaptive_level'].astype('category').cat.codes
    #vineland['soc_adaptive_level'] = vineland['soc_adaptive_level'].astype('category').cat.codes
    #vineland['motor_adaptive_level'] = vineland['motor_adaptive_level'].astype('category').cat.codes
    #vineland=vineland.replace('>252',253) ## express_age has '>252' which throws an error
    #vineland=vineland.replace('>264',265) ## line to fix receptive age and written age.
    #vineland=vineland.replace('>118',119) ## line to fix fine age and gross age
    vineland=vineland.replace('<36',35) # replace strings with ints
    vineland=vineland.replace('<24',23) # replace strings with ints
    vineland = vineland[~vineland.index.duplicated(keep=False)]

    ### PREDICTED ID
    prediq = pd.read_csv(f'{BASE_PHENO_DIR}/predicted_iq_experimental_2022-12-12.csv') ## predicted iq
    prediq = prediq.set_index('subject_sp_id',drop=True).drop(['family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'asd', 'ml_predicted_cog_impair'], axis=1)
    prediq = prediq[~prediq.index.duplicated(keep=False)]

    ### AREA DEPRIVATION INDEX
    adi = pd.read_csv(f'{BASE_PHENO_DIR}/area_deprivation_index_2022-12-12.csv') ## ADI
    adi = adi.set_index('subject_sp_id',drop=True).drop(['family_sf_id', 'adi_version'], axis=1)
    adi = adi[~adi.index.duplicated(keep=False)]
    adi.loc[adi["adi_national_rank_percentile"] == "GQ", "adi_national_rank_percentile"] = 0
    adi.loc[adi["adi_national_rank_percentile"] == "GQ-PH", "adi_national_rank_percentile"] = 0
    adi.loc[adi["adi_national_rank_percentile"] == "QDI", "adi_national_rank_percentile"] = 0
    adi.loc[adi["adi_national_rank_percentile"] == "PH", "adi_national_rank_percentile"] = 0
    adi.loc[adi["adi_state_rank_decile"] == "GQ", "adi_state_rank_decile"] = 0
    adi.loc[adi["adi_state_rank_decile"] == "GQ-PH", "adi_state_rank_decile"] = 0
    adi.loc[adi["adi_state_rank_decile"] == "QDI", "adi_state_rank_decile"] = 0
    adi.loc[adi["adi_state_rank_decile"] == "PH", "adi_state_rank_decile"] = 0

    # merge all data
    finaldf = pd.concat([scqdf, bhdf, rbsr, cbcl_2],axis=1,join='inner')
    finaldf = finaldf.loc[:,~finaldf.columns.duplicated()] 
    
    # drop features with > 0.1 proportion missing values
    #print(features with >0.1 missing)
    print(finaldf.columns[finaldf.isna().sum()/finaldf.shape[0] > 0.1])
    finaldf = finaldf.loc[:, finaldf.isna().sum()/finaldf.shape[0] < 0.1]
    print(finaldf.shape)
    '''
    # FEATURE SELECTION - remove features with low variance
    from sklearn.feature_selection import VarianceThreshold
    threshold = 0.15
    finaldf_copy = finaldf.copy()
    column_names = finaldf.columns
    selector = VarianceThreshold(threshold=threshold)
    finaldf = selector.fit_transform(finaldf)
    # turn back into a dataframe
    cols = selector.get_support(indices=True)
    finaldf = pd.DataFrame(finaldf, columns=column_names[cols])
    # get back sex column
    finaldf.index = finaldf_copy.index
    #finaldf = finaldf.merge(finaldf_copy[['sex']], left_index=True, right_index=True)
    #print(finaldf.columns); exit()
    
    # drop binary features with missing values
    continuous, categorical, binary = split_columns(finaldf)
    binary_df = finaldf[binary]
    # remove binary features with > 1% missing values
    binary_missing = binary_df.columns[binary_df.isna().sum()/binary_df.shape[0] > 0.01]
    finaldf = finaldf.drop(binary_missing, axis=1)
    '''
    # SAVE CLEAN DATA - DROP NULL VALUES
    clean_df = finaldf.dropna(axis=0)
    print(clean_df['age_at_eval_years'].value_counts())
    print(clean_df.shape)
    print(clean_df['sex'].value_counts())
    clean_df.to_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t')
    exit()
    
    # plot proportion missing values vs. count of features histogram
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hist(finaldf.isna().sum()/finaldf.shape[0], bins=30, color='blue')
    plt.ylabel('Count of Features', fontsize=14)
    plt.xlabel('Proportion Missing Values', fontsize=14)
    plt.savefig('/mnt/home/alitman/SPARK/GFMM_all_figures/imputed_cohort_missing_values_hist.png')
    plt.close()

    # plot number of people (x) vs. number of features (y) with missing values
    fig, ax = plt.subplots(figsize=(8, 5))
    # iterate through and count number of missing features for each person
    missing = []
    for i in range(finaldf.shape[0]):
        missing.append(finaldf.iloc[i].isna().sum())
    plt.hist(missing, bins=30, color='purple')
    plt.ylabel('Count of People', fontsize=14)
    plt.xlabel('Count of Features with Missing Values', fontsize=14)
    plt.savefig('/mnt/home/alitman/SPARK/GFMM_all_figures/imputed_cohort_missing_values_people_hist.png')
    plt.close()

    # GET IMPUTED COHORT
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler
    k = 10
    scaler = MinMaxScaler()
    imputer = KNNImputer(n_neighbors=k, weights='distance', metric='nan_euclidean')
    features = finaldf.columns # save feature names
    idx = finaldf.index
    finaldf = scaler.fit_transform(finaldf)
    finaldf = imputer.fit_transform(finaldf)
    # 3. Inverse transform (undo scaling)
    finaldf = scaler.inverse_transform(finaldf)
    finaldf = pd.DataFrame(finaldf, columns=features, index=idx)
    print(finaldf.columns[finaldf.isna().sum()/finaldf.shape[0] > 0.01])
    print(finaldf.shape)

    # round each column up or down to the nearest integer
    finaldf = finaldf.round(0)
    finaldf.to_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_7032_imputed_cohort.txt',sep='\t')


if __name__ == '__main__':
    get_main_spark_data_for_GFMM()
