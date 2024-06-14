import numpy as np


def sabic(model, X, Y=None):
    """Sample-Sized Adjusted BIC.

    References
    ----------
    Sclove SL. Application of model-selection criteria to some problems in multivariate analysis. Psychometrika. 1987;52(3):333–343.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    ssa_bic : float
    """
    n = X.shape[0]

    return -2 * model.score(X, Y) * n + model.n_parameters * np.log(
        n * ((n + 2) / 24)
    )


def c_aic(model, X, Y=None):
    """Consistent AIC.

    References
    ----------
    Bozdogan, H. 1987. Model selection and Akaike’s information criterion (AIC):
    The general theory and its analytical extensions. Psychometrika 52: 345–370.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    caic : float
        The lower the better.
    """
    n = X.shape[0]
    return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1)


def awe(model, X, Y=None):
        """Approximate weight of evidence. (Banfield & Raftery (1993))

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        awe : float
        """
        n = X.shape[0]
        return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1.5)


def scramble_column(column):
    return np.random.permutation(column)


def get_cross_cohort_SPARK_data():
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'

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

    bhdf = pd.concat([bhcdf, bhsdf], join='inner') 
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
    cbcl_2 = pd.read_csv(f'{BASE_PHENO_DIR}/cbcl_6_18_2022-12-12.csv')
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
    ## only keep subscale scores from CBCL
    cbcl_2 = cbcl_2[['anxious_depressed_t_score', 'withdrawn_depressed_t_score', 'somatic_complaints_t_score', 'social_problems_t_score', 'thought_problems_t_score', 'attention_problems_t_score', 'rule_breaking_behavior_t_score', 'aggressive_behavior_t_score', 'internalizing_problems_t_score', 'externalizing_problems_t_score', 'total_problems_t_score', 'obsessive_compulsive_problems_t_score', 'sluggish_cognitive_tempo_t_score', 'stress_problems_t_score', 'dsm5_conduct_problems_t_score', 'dsm5_somatic_problems_t_score', 'dsm5_oppositional_defiant_t_score', 'dsm5_attention_deficit_hyperactivity_t_score', 'dsm5_anxiety_problems_t_score', 'dsm5_depressive_problems_t_score']]
    
    # merge all data
    finaldf = pd.concat([scqdf, bhdf, rbsr, cbcl_2],axis=1,join='inner')
    finaldf = finaldf.loc[:,~finaldf.columns.duplicated()] 
    
    # drop features with > 0.1 proportion missing values and dropna
    finaldf = finaldf.loc[:, finaldf.isna().sum()/finaldf.shape[0] < 0.1]
    finaldf = finaldf.dropna(axis=0)
    
    return finaldf