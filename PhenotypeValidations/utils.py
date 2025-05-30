import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, binomtest, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two groups.

    Parameters:
    - group1: array-like, containing data for group 1
    - group2: array-like, containing data for group 2

    Returns:
    - d: Cohen's d
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    d = mean_diff / pooled_std
    return d


def adjust_pvalues(p_values, method):
    return multipletests(p_values, method=method)[1]


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


def get_cross_cohort_SPARK_data():
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'

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


def generate_ssc_data():
    ssc_data_dir = '../SSC_Phenotype_Dataset/Proband_Data'
    
    cbcl = pd.read_csv(f'{ssc_data_dir}/cbcl_6_18.csv').set_index('individual',drop=True).drop(['measure', 'activities_total', 'add_adhd_total',
                                                                                                'affective_problems_total', 'aggressive_behavior_total', 'anxiety_problems_total', 'anxious_depressed_total',
                                                                                                'attention_problems_total', 'conduct_problems_total', 'externalizing_problems_total', 'internalizing_problems_total',
                                                                                                'oppositional_defiant_total', 'rule_breaking_total', 'school_total', 'social_problems_total', 'social_total',
                                                                                                'somatic_complaints_total', 'somatic_prob_total', 'thought_problems_total', 'total_competence_total',
                                                                                                'total_problems_total', 'withdrawn_total', 'activities_t_score', 'school_t_score', 'total_competence_t_score',
                                                                                                'social_t_score'], axis=1)
    # rename cbcl scores to match spark
    cbcl.rename(columns={'add_adhd_t_score': 'dsm5_attention_deficit_hyperactivity_t_score', 'affective_problems_t_score': 'dsm5_depressive_problems_t_score', 'aggressive_behavior_t_score': 'aggressive_behavior_t_score',
                        'anxiety_problems_t_score': 'dsm5_anxiety_problems_t_score', 'anxious_depressed_t_score': 'anxious_depressed_t_score', 'attention_problems_t_score': 'attention_problems_t_score',
                        'conduct_problems_t_score': 'dsm5_conduct_problems_t_score', 'externalizing_problems_t_score': 'externalizing_problems_t_score', 'internalizing_problems_t_score': 'internalizing_problems_t_score',
                        'oppositional_defiant_t_score': 'dsm5_oppositional_defiant_t_score', 'rule_breaking_t_score': 'rule_breaking_behavior_t_score', 'social_problems_t_score': 'social_problems_t_score',
                        'somatic_complaints_t_score': 'somatic_complaints_t_score', 'thought_problems_t_score': 'thought_problems_t_score',
                        'total_problems_t_score': 'total_problems_t_score', 'withdrawn_t_score': 'withdrawn_depressed_t_score', 'somatic_prob_t_score': 'dsm5_somatic_problems_t_score'}, inplace=True)
    
    rbsr_scores = pd.read_csv(f'{ssc_data_dir}/rbs_r.csv').set_index('individual',drop=True).drop(['measure', 'overall_number_items', 'overall_score', 'status', 'iii_compulsive_behavior_items', 'ii_self_injurious_items',
                                                                                                    'i_stereotyped_behavior_items', 'iv_ritualistic_behavior_items',
                                                                                                    'vi_restricted_behavior_items', 'v_sameness_behavior_items'], axis=1) 
    rbsr_raw = pd.read_csv(f'{ssc_data_dir}/rbs_r_raw.csv').set_index('individual',drop=True).drop(['measure'], axis=1) 
    rbsr_raw.rename(columns={'q39_insists_palce': 'q39_insists_time'}, inplace=True)
    
    scq_raw = pd.read_csv(f'{ssc_data_dir}/scq_life_recode.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq_raw.replace('yes', 1, inplace=True)
    scq_raw.replace('no', 0, inplace=True)
    scq_raw.rename(columns={'q08_hits_self_object': 'q08_hits_self_against_object', 'q09_hits_self_object': 'q09_hits_self_with_object',
                            'q28_communicatiion': 'q28_communication'}, inplace=True)
    
    # get total scq score
    scq = pd.read_csv(f'{ssc_data_dir}/scq_life.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq.rename(columns={'summary_score': 'final_score'}, inplace=True)
    
    core_descriptive = pd.read_csv(f'{ssc_data_dir}/ssc_core_descriptive.csv').set_index('individual', drop=True).drop(['ssc_diagnosis_full_scale_iq', 'ssc_diagnosis_nonverbal_iq', 'ssc_diagnosis_verbal_iq', 'measure', 'abc_total_score', 'adi_r_b_comm_verbal_total', 'adi_r_comm_b_non_verbal_total', 'adi_r_cpea_dx', 'adi_r_evidence_onset', 'adi_r_rrb_c_total', 'adi_r_soc_a_total', 'ados_communication_social', 'ados_css', 'ados_module', 'ados_restricted_repetitive', 'ados_social_affect', 'cbcl_2_5_externalizing_t_score', 'cbcl_2_5_internalizing_t_score', 'cbcl_6_18_externalizing_t_score', 'cbcl_6_18_internalizing_t_score', 'cpea_dx', 'diagnosis_ados', 'ethnicity', 'family_type', 'febrile_seizures', 'non_febrile_seizures', 'pregnancy_optimality_code', 'pregnancy_optimality_code_intrapartal', 'pregnancy_optimality_code_neonatal', 'pregnancy_optimality_code_prenatal', 'pregnancy_optimality_intrapartal', 'pregnancy_optimality_neonatal', 'pregnancy_optimality_prenatal', 'pregnancy_optimality_total', 'puberty_ds_progress', 'puberty_ds_total', 'race', 'rbs_r_overall_score', 'regression', 'regression_loss', 'regression_no_insert', 'srs_parent_raw_total', 'srs_parent_t_score', 'srs_teacher_raw_total', 'srs_teacher_t_score', 'ssc_diagnosis_full_scale_iq_type', 'ssc_diagnosis_nonverbal_iq_type', 'ssc_diagnosis_nvma', 'ssc_diagnosis_verbal_iq_type', 'ssc_diagnosis_vma', 'vineland_ii_composite_standard_score'], axis=1)
    core_descriptive.replace('male', 1, inplace=True)
    core_descriptive.replace('female', 0, inplace=True)
    core_descriptive.rename(columns={'age_at_ados': 'age_at_eval_years'}, inplace=True)
    core_descriptive['age_at_eval_years'] = core_descriptive['age_at_eval_years']/12
    
    bh = pd.read_csv('../SSC_Phenotype_Dataset/SSC_background_hx_clean.csv').set_index('individual', drop=True)
    bh = bh.drop(['age_gave_up_bottle', 'age_started_solid_foods'], axis=1) # not in SPARK model
    
    # merge data
    finaldf = pd.concat([core_descriptive, bh, scq_raw, scq, rbsr_raw, rbsr_scores, cbcl], axis=1, join='inner')
    #finaldf = finaldf.loc[:, finaldf.isna().sum()/finaldf.shape[0] < 0.1]
    finaldf = finaldf.dropna(axis=0)
    finaldf = finaldf.astype('float32')
    #print(finaldf.shape); exit()

    return finaldf


def split_columns(feature_subset):
    '''
    Given list of features (strings), return stratified lists of continuous, binary, and categorical lists containing the feature subset.
    '''
    binary_columns = ['sex', 'attn_behav', 'behav_adhd', 'behav_conduct', 'behav_intermitt_explos', 'behav_odd',
                      'birth_def_bone', 'birth_def_bone_club', 'birth_def_bone_miss', 'birth_def_bone_polydact',
                      'birth_def_bone_spine', 'birth_def_cleft_lip', 'birth_def_cleft_palate', 'birth_def_cns',
                      'birth_def_cns_brain', 'birth_def_cns_myelo', 'birth_def_fac', 'birth_def_gastro',
                      'birth_def_gi_esoph_atres', 'birth_def_gi_hirschprung', 'birth_def_gi_intest_malrot',
                      'birth_def_gi_pylor_sten', 'birth_def_thorac', 'birth_def_thorac_cdh', 'birth_def_thorac_heart',
                      'birth_def_thorac_lung', 'birth_def_urogen', 'birth_def_urogen_hypospad',
                      'birth_def_urogen_renal', 'birth_def_urogen_renal_agen', 'birth_def_urogen_uter_agen',
                      'birth_def_oth_calc', 'birth_etoh_subst', 'birth_ivh', 'birth_oth_calc', 'birth_oxygen',
                      'birth_pg_inf', 'birth_prem', 'cog_med', 'dev_id', 'dev_lang', 'dev_lang_dis', 'dev_ld',
                      'dev_motor', 'dev_mutism', 'dev_soc_prag', 'dev_speech', 'eating_probs', 'eating_disorder',
                      'encopres', 'enures', 'etoh_subst', 'feeding_dx', 'growth_low_wt', 'growth_macroceph',
                      'growth_microceph', 'growth_obes', 'growth_short', 'growth_oth_calc', 'mood_anx', 'mood_bipol',
                      'mood_dep', 'mood_dmd', 'mood_hoard', 'mood_ocd', 'mood_or_anx', 'mood_sep_anx', 'mood_soc_anx',
                      'neuro_inf', 'neuro_lead', 'neuro_sz', 'neuro_tbi', 'neuro_oth_calc', 'pers_dis',
                      'prev_study_oth_calc', 'psych_oth_calc', 'schiz', 'sleep_dx', 'sleep_probs', 'tics',
                      'visaud_blind', 'visaud_catar', 'visaud_deaf', 'visaud_strab', 'q01_phrases', 'q02_conversation',
                      'q03_odd_phrase', 'q04_inappropriate_question', 'q05_pronouns_mixed', 'q06_invented_words',
                      'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate', 'q10_hand_tool',
                      'q11_interest_preoccupy', 'q12_parts_object', 'q13_interests_intensity', 'q14_senses',
                      'q15_odd_ways', 'q16_complicated_movements', 'q17_injured_deliberately', 'q18_objects_carry',
                      'q19_best_friend', 'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted',
                      'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 'q28_things_interested',
                      'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention', 'q33_range_expressions',
                      'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
                      'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games', 'derived_cog_impair', 'ml_predicted_cog_impair']
    categorical_columns = ['q01_whole_body', 'q02_head', 'q03_hand_finger', 'q04_locomotion', 'q05_object_usage',
                           'q06_sensory', 'q07_hits_self_body', 'q08_hits_self_against_object',
                           'q09_hits_self_with_object', 'q10_bites_self', 'q11_pulls', 'q12_rubs', 'q13_inserts_finger',
                           'q14_skin_picking', 'q15_arranging', 'q16_complete', 'q17_washing', 'q18_checking',
                           'q19_counting', 'q20_hoarding', 'q21_repeating', 'q22_touch_tap', 'q23_eating', 'q24_sleep',
                           'q25_self_care', 'q26_travel', 'q27_play', 'q28_communication', 'q29_things_same_place',
                           'q30_objects', 'q31_becomes_upset', 'q32_insists_walking', 'q33_insists_sitting',
                           'q34_dislikes_changes', 'q35_insists_door', 'q36_likes_piece_music', 'q37_resists_change',
                           'q38_insists_routine', 'q39_insists_time', 'q40_fascination_subject',
                           'q41_strongly_attached', 'q42_preoccupation', 'q43_fascination_movement', 'close_friends',
                           'contact_friends_outside_school', 'gets_along_siblings', 'gets_along_other_kids',
                           'behave_with_parents', 'play_work_alone', 'reading_eng_language', 'history_social_studies',
                           'arithmetic_math', 'science', 'q001_acts_young', 'q002_drinks_alcohol', 'q003_argues',
                           'q004_fails_to_finish', 'q005_very_little_enjoyment', 'q006_bowel_movements_outside',
                           'q007_brag_boast', 'q008_concentrate', 'q009_obsessions', 'q010_restless',
                           'q011_too_dependent', 'q012_loneliness', 'q013_confused', 'q014_cries_a_lot',
                           'q015_cruelty_animals', 'q016_cruelty_others', 'q017_daydreams', 'q018_harms_self',
                           'q019_demands_attention', 'q020_destroys_own_things', 'q021_destroys_others_things',
                           'q022_disobedient_home', 'q023_disobedient_school', 'q024_doesnt_eat_well',
                           'q025_doesnt_get_along_others', 'q026_guilty_misbehaving', 'q027_jealous',
                           'q028_breaks_rules', 'q029_fears', 'q030_fears_school', 'q031_fears_bad', 'q032_perfect',
                           'q033_fears_no_one_loves', 'q034_out_to_get', 'q035_feels_worthless', 'q036_accident_prone',
                           'q037_fights', 'q038_teased', 'q039_hangs_around_trouble', 'q040_hears_voices',
                           'q041_impulsive', 'q042_rather_alone', 'q043_lying', 'q044_bites_fingernails',
                           'q045_nervous_tense', 'q046_twitching', 'q047_nightmares', 'q048_not_liked',
                           'q049_constipated', 'q050_anxious', 'q051_dizzy', 'q052_feels_too_guilty', 'q053_overeating',
                           'q054_overtired', 'q055_overweight', 'q056_a_aches', 'q056_b_headache', 'q056_c_nausea',
                           'q056_d_eyes', 'q056_e_rashes', 'q056_f_stomachaches',
                           'q056_g_vomiting', 'q056_h_other', 'q057_attacks', 'q058_picks_skin',
                           'q059_sex_parts_public', 'q060_sex_parts_too_much', 'q061_poor_work', 'q062_clumsy',
                           'q063_rather_older_kids', 'q064_rather_younger_kids', 'q065_refuses_to_talk',
                           'q066_repeats_acts', 'q067_runs_away_home', 'q068_screams_a_lot', 'q069_secretive',
                           'q070_sees_things', 'q071_self_conscious', 'q072_sets_fires', 'q073_sexual_problems',
                           'q074_clowning', 'q075_too_shy', 'q076_sleeps_less', 'q077_sleeps_more',
                           'q078_easily_distracted', 'q079_speech_problem', 'q080_stares_blankly', 'q081_steals_home',
                           'q082_steals_outside', 'q083_stores_many_things', 'q084_strange_behavior',
                           'q085_strange_ideas', 'q086_stubborn', 'q087_changes_mood', 'q088_sulks', 'q089_suspicious',
                           'q090_obscene_language', 'q091_talks_killing_self', 'q092_talks_walks_sleep',
                           'q093_talks_too_much', 'q094_teases_a_lot', 'q095_tantrums', 'q096_thinks_sex_too_much',
                           'q097_threatens', 'q098_thumb_sucking', 'q099_tobacco', 'q100_trouble_sleeping',
                           'q101_skips_school', 'q102_underactive', 'q103_unhappy', 'q104_unusually_loud', 'q105_drugs',
                           'q106_vandalism', 'q107_wets_self', 'q108_wets_bed', 'q109_whining',
                           'q110_wishes_to_be_opp_sex', 'q111_withdrawn', 'q112_worries']
    continuous_columns = ['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
                          'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos',
                          'combined_phrases_age_mos', 'bladder_trained_age_mos', 'bowel_trained_age_mos',
                          'repeat_grade', 'not_able', 'adi_national_rank_percentile', 'adi_state_rank_decile',
                          'anxious_depressed_t_score', 'withdrawn_depressed_t_score', 'somatic_complaints_t_score',
                          'social_problems_t_score', 'thought_problems_t_score', 'attention_problems_t_score',
                          'rule_breaking_behavior_t_score', 'aggressive_behavior_t_score',
                          'internalizing_problems_t_score', 'externalizing_problems_t_score', 'total_problems_t_score',
                          'obsessive_compulsive_problems_t_score', 'sluggish_cognitive_tempo_t_score',
                          'stress_problems_t_score', 'dsm5_conduct_problems_t_score', 'dsm5_somatic_problems_t_score',
                          'dsm5_oppositional_defiant_t_score', 'dsm5_attention_deficit_hyperactivity_t_score',
                          'dsm5_anxiety_problems_t_score', 'dsm5_depressive_problems_t_score',
                          'i_stereotyped_behavior_score', 'ii_self_injurious_score', 'iii_compulsive_behavior_score',
                          'iv_ritualistic_behavior_score', 'v_sameness_behavior_score', 'vi_restricted_behavior_score', 'final_score', 'social_domain', 'communication_domain']

    continuous_columns_subset = []
    categorical_columns_subset = []
    binary_columns_subset = []
    for feature in feature_subset:
        if feature in continuous_columns:
            continuous_columns_subset.append(feature)
        elif feature in categorical_columns:
            categorical_columns_subset.append(feature)
        elif feature in binary_columns:
            binary_columns_subset.append(feature)

    return continuous_columns_subset, binary_columns_subset, categorical_columns_subset


def get_feature_enrichments(mixed_data, summarize=False):
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary
            binary_features.append(feature)
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            subset_class3 = int(np.sum(class3[feature]))

            sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
            sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue

            sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
            sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue

            sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
            sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue

            sf3 = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='greater').pvalue
            sf3_less = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='less').pvalue

            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background
            fe3 = (subset_class3/total_in_class3)/background

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            feature_to_pval[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less, sf3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        elif len(unique) > 2:  ## continuous or categorical
            pval_class0 = ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class3 = ttest_ind(class3[feature],
                                          pd.concat([class0[feature], class1[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue

            pval_class0_less = ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class3_less = ttest_ind(class3[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, pval_class3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        else:
            continue
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2, 3]
    fold_enrichments['cluster'] = [0, 1, 2, 3]
    mean_values['cluster'] = [0, 1, 2, 3]
    pval_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)

    for tested_class in range(4):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1  

        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(8), index=mixed_data.columns)
        for i in range(4):
            enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+4] = val
        
        df = df[[0,4,1,5,2,6,3,7]] # rearrange columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted']
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d)
        fold_enrichments = fold_enrichments.drop('cluster', axis=1).T.dropna(axis=0)
        fold_enrichments = pd.concat([fold_enrichments[0], fold_enrichments[0],
                                      fold_enrichments[1], fold_enrichments[1],
                                      fold_enrichments[2], fold_enrichments[2],
                                      fold_enrichments[3], fold_enrichments[3]], axis=1)
        fold_enrichments.columns = df.columns[1:]
        df_binary = df.copy()
        df_binary = df_binary.set_index('feature')
        df_binary = df_binary.applymap(lambda x: 1 if x < 0.05 else 0)
        df_binary = df_binary * fold_enrichments
        df_binary = df_binary.groupby(lambda x:x.split('_')[0], axis=1).sum()
        df_binary = df_binary.drop([col for col in df_binary.columns if 'depleted' in col], axis=1)
        df_binary.columns = [col.split('_')[0] for col in df_binary.columns]
        df_binary = df_binary.replace(0, np.nan)
        df_binary = df_binary.drop('mixed_pred', axis=0)
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector


def match_class_labels(gt_df, exp_df):
    """
    Match class labels from experimental run to ground truth based on overlap.
    If a class gets assigned twice, the class with the highest proportion overlap keeps the assignment,
    and the remaining unmatched class is assigned to the class that remained unassigned.

    Parameters:
    gt_df (pd.DataFrame): DataFrame with ground truth labels, must have a column named 'mixed_pred'.
    exp_df (pd.DataFrame): DataFrame with experimental labels, must have a column named 'mixed_pred'.

    Returns:
    pd.DataFrame: A new DataFrame with experimental labels reassigned based on overlap with ground truth.
    """
    
    # Check if the required column exists
    if 'mixed_pred' not in gt_df.columns or 'mixed_pred' not in exp_df.columns:
        raise ValueError("Both DataFrames must contain a 'mixed_pred' column.")
    
    # Get unique class labels
    gt_classes = gt_df['mixed_pred'].unique()
    exp_classes = exp_df['mixed_pred'].unique()

    # Initialize a dictionary to store overlaps
    overlap_dict = {exp_class: {gt_class: 0 for gt_class in gt_classes} for exp_class in exp_classes}

    # Calculate overlap between experimental and ground truth classes
    for exp_class in np.arange(len(exp_classes)):
        for gt_class in np.arange(len(gt_classes)):
            overlap = len(gt_df[gt_df['mixed_pred'] == gt_class].index.intersection(exp_df[exp_df['mixed_pred'] == exp_class].index))
            exp_class_size = len(exp_df[exp_df['mixed_pred'] == exp_class].index)  # normalize by class size
            overlap = overlap / exp_class_size
            overlap_dict[exp_class][gt_class] = overlap

    # Create a mapping for experimental classes to ground truth classes based on max overlap
    label_mapping = {}
    assigned_gt_classes = {}
    
    for exp_class, overlaps in overlap_dict.items():
        max_gt_class = max(overlaps, key=overlaps.get)
        
        # If the max_gt_class is already assigned to another exp_class
        if max_gt_class in assigned_gt_classes:
            # Compare the overlaps of the conflicting classes
            other_exp_class = assigned_gt_classes[max_gt_class]
            if overlap_dict[other_exp_class][max_gt_class] < overlap_dict[exp_class][max_gt_class]:
                # Assign max_gt_class to the current exp_class
                label_mapping[exp_class] = max_gt_class
                # The previous assignment gets unassigned
                del label_mapping[other_exp_class]
            else:
                # Keep the previous assignment and unassign the current exp_class
                continue
        else:
            label_mapping[exp_class] = max_gt_class
        
        # Track which experimental classes were assigned to ground truth
        assigned_gt_classes[max_gt_class] = exp_class

    # Handle any unassigned experimental or ground truth classes
    unassigned_exp_classes = set(exp_classes) - set(label_mapping.keys())
    unassigned_gt_classes = set(gt_classes) - set(label_mapping.values())

    # Assign remaining experimental class to remaining unassigned ground truth class
    if unassigned_exp_classes and unassigned_gt_classes:
        for exp_class, gt_class in zip(unassigned_exp_classes, unassigned_gt_classes):
            label_mapping[exp_class] = gt_class

    # Reassign experimental labels based on the mapping
    exp_df['matched_labels'] = exp_df['mixed_pred'].map(label_mapping)
    exp_df = exp_df.drop('mixed_pred', axis=1)
    exp_df.rename(columns={'matched_labels': 'mixed_pred'}, inplace=True)

    return exp_df


def get_correlation(spark_labels, ssc_labels):
    # get feature enrichments
    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(spark_labels, summarize=True)
    
    features_to_exclude = fold_enrichments.copy() # Fold enrichments + Cohen's d values filtered by significance
    # take abs value of cohen's d
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # numerically get features to exclude:
    # (1) features with no significant enrichments in any class
    # (2) features with all cohen's d values < 0.2 or FE < 1.5
    # get features where all classes is nan
    binary_features = ['repeat_grade', 'q01_phrases', 'q02_conversation', 'q03_odd_phrase', 'q04_inappropriate_question', 'q05_pronouns_mixed', 'q06_invented_words', 'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate', 'q10_hand_tool', 'q11_interest_preoccupy', 'q12_parts_object', 'q13_interests_intensity', 'q14_senses', 'q15_odd_ways', 'q16_complicated_movements', 'q17_injured_deliberately', 'q18_objects_carry', 'q19_best_friend', 'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention', 'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively', 'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    # GET FEATURES TO EXCLUDE
    nan_features = features_to_exclude.loc[(features_to_exclude['class0'].isna()) & 
                                            (features_to_exclude['class1'].isna()) & 
                                            (features_to_exclude['class2'].isna()) & 
                                            (features_to_exclude['class3'].isna())]
    low_features_continuous = features_to_exclude.loc[~features_to_exclude['feature'].isin(binary_features)]
    low_features_continuous = features_to_exclude.loc[(features_to_exclude['class0'] < 0.2) & (features_to_exclude['class1'] < 0.2) 
                                            & (features_to_exclude['class2'] < 0.2) & (features_to_exclude['class3'] < 0.2)] 
    low_features_binary = features_to_exclude.loc[features_to_exclude['feature'].isin(binary_features)]
    low_features_binary = low_features_binary.loc[(low_features_binary['class0'] < 1.5) & (low_features_binary['class1'] < 1.5)
                                            & (low_features_binary['class2'] < 1.5) & (low_features_binary['class3'] < 1.5)]
    features_to_exclude = pd.concat([nan_features, low_features_continuous, low_features_binary])
    features_to_exclude = features_to_exclude['feature'].unique()

    # read in feature_to_category mapping
    features_to_category = pd.read_csv('data/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))

    df = df_enriched_depleted.copy()
    df = df.fillna('NaN')
    # drop feature category column
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    
    df = df.loc[~df['feature'].isin(features_to_exclude)] # remove non-contributory features
    
    # annotate each feature with its category
    df['feature_category'] = df['feature'].map(feature_to_category)
    # drop features with no category
    df = df.dropna(subset=['feature_category'])
    df = df.replace('NaN', 1)

    # dictionary of feature_category to feature names
    feature_category_to_features = dict()
    for category in df['feature_category'].unique():
        feature_category_to_features[category] = df.loc[df['feature_category'] == category, 'feature'].to_list()
    
    # convert to float
    df['class0_enriched'] = df['class0_enriched'].astype(float)
    df['class0_depleted'] = df['class0_depleted'].astype(float)
    df['class1_enriched'] = df['class1_enriched'].astype(float)
    df['class1_depleted'] = df['class1_depleted'].astype(float)
    df['class2_enriched'] = df['class2_enriched'].astype(float)
    df['class2_depleted'] = df['class2_depleted'].astype(float)
    df['class3_enriched'] = df['class3_enriched'].astype(float)
    df['class3_depleted'] = df['class3_depleted'].astype(float)
    # convert p-value columns to binary (1 if significant, 0 if not)
    df['class0_enriched'] = df['class0_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class0_depleted'] = df['class0_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class1_enriched'] = df['class1_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class1_depleted'] = df['class1_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class2_enriched'] = df['class2_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class2_depleted'] = df['class2_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class3_enriched'] = df['class3_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class3_depleted'] = df['class3_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    
    # flip_rows contains feature names that are reverse-coded in the enrichment/depletion columns
    flip_rows = ['q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 'q20_talk_friendly',
                'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly',
                'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention',
                'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
                'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    
    for row in flip_rows:
        if row in df['feature'].values:
            df.loc[df['feature'] == row, ['class0_enriched', 'class0_depleted']] = df.loc[df['feature'] == row, ['class0_depleted', 'class0_enriched']].values
            df.loc[df['feature'] == row, ['class1_enriched', 'class1_depleted']] = df.loc[df['feature'] == row, ['class1_depleted', 'class1_enriched']].values
            df.loc[df['feature'] == row, ['class2_enriched', 'class2_depleted']] = df.loc[df['feature'] == row, ['class2_depleted', 'class2_enriched']].values
            df.loc[df['feature'] == row, ['class3_enriched', 'class3_depleted']] = df.loc[df['feature'] == row, ['class3_depleted', 'class3_enriched']].values
        
    # create new dataframe with the proportions of significant features in each category
    prop_df = pd.DataFrame()
    prop_df['class0_enriched'] = df.groupby(['feature_category'])['class0_enriched'].sum()/df.groupby(['feature_category'])['class0_enriched'].count()
    prop_df['class0_depleted'] = df.groupby(['feature_category'])['class0_depleted'].sum()/df.groupby(['feature_category'])['class0_depleted'].count()
    prop_df['class1_enriched'] = df.groupby(['feature_category'])['class1_enriched'].sum()/df.groupby(['feature_category'])['class1_enriched'].count()
    prop_df['class1_depleted'] = df.groupby(['feature_category'])['class1_depleted'].sum()/df.groupby(['feature_category'])['class1_depleted'].count()
    prop_df['class2_enriched'] = df.groupby(['feature_category'])['class2_enriched'].sum()/df.groupby(['feature_category'])['class2_enriched'].count()
    prop_df['class2_depleted'] = df.groupby(['feature_category'])['class2_depleted'].sum()/df.groupby(['feature_category'])['class2_depleted'].count()
    prop_df['class3_enriched'] = df.groupby(['feature_category'])['class3_enriched'].sum()/df.groupby(['feature_category'])['class3_enriched'].count()
    prop_df['class3_depleted'] = df.groupby(['feature_category'])['class3_depleted'].sum()/df.groupby(['feature_category'])['class3_depleted'].count()
    
    # negate depleted columns
    prop_df['class0_depleted'] = -prop_df['class0_depleted']
    prop_df['class1_depleted'] = -prop_df['class1_depleted']
    prop_df['class2_depleted'] = -prop_df['class2_depleted']
    prop_df['class3_depleted'] = -prop_df['class3_depleted']

    # sum negative depleted columns with positive enriched columns
    prop_df['class0_max'] = prop_df[['class0_enriched', 'class0_depleted']].sum(axis=1)
    prop_df['class1_max'] = prop_df[['class1_enriched', 'class1_depleted']].sum(axis=1)
    prop_df['class2_max'] = prop_df[['class2_enriched', 'class2_depleted']].sum(axis=1)
    prop_df['class3_max'] = prop_df[['class3_enriched', 'class3_depleted']].sum(axis=1)
    
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['0', '1', '2', '3']
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'] 
    spark_prop_df = prop_df.loc[features_to_visualize]
        
    spark_prop_df.index = np.arange(len(spark_prop_df))

    # analyze SSC predictions
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()

    # rename ssc_pred to mixed_pred
    ssc_labels = ssc_labels.rename({'ssc_pred': 'mixed_pred'}, axis=1)

    ## extract values for classes
    class0 = ssc_labels[ssc_labels['mixed_pred'] == 0]
    class1 = ssc_labels[ssc_labels['mixed_pred'] == 1]
    class2 = ssc_labels[ssc_labels['mixed_pred'] == 2]
    class3 = ssc_labels[ssc_labels['mixed_pred'] == 3]

    ssc_pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, summary_df, fold_enrichments = get_feature_enrichments(ssc_labels, summarize=True)
    
    summary_df = summary_df.fillna('NaN')
    summary_df = summary_df.replace(np.nan, 1)
    summary_df = summary_df.loc[~summary_df['feature'].isin(features_to_exclude)] # remove non-contributory features from SPARK
    ssc_feature_subset = summary_df['feature'].to_list()
    
    summary_df['feature_category'] = summary_df['feature'].map(feature_to_category)
    summary_df = summary_df.dropna(subset=['feature_category'])

    summary_df['class0_enriched'] = summary_df['class0_enriched'].astype(float)
    summary_df['class0_depleted'] = summary_df['class0_depleted'].astype(float)
    summary_df['class1_enriched'] = summary_df['class1_enriched'].astype(float)
    summary_df['class1_depleted'] = summary_df['class1_depleted'].astype(float)
    summary_df['class2_enriched'] = summary_df['class2_enriched'].astype(float)
    summary_df['class2_depleted'] = summary_df['class2_depleted'].astype(float)
    summary_df['class3_enriched'] = summary_df['class3_enriched'].astype(float)
    summary_df['class3_depleted'] = summary_df['class3_depleted'].astype(float)
    
    # convert p value columns to binary (1 if significant, 0 if not)
    summary_df['class0_enriched'] = summary_df['class0_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class0_depleted'] = summary_df['class0_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_enriched'] = summary_df['class1_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_depleted'] = summary_df['class1_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_enriched'] = summary_df['class2_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_depleted'] = summary_df['class2_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class3_enriched'] = summary_df['class3_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class3_depleted'] = summary_df['class3_depleted'].apply(lambda x: 1 if x < 0.05 else 0)

    for row in flip_rows:
        if row in summary_df['feature'].values:
            summary_df.loc[summary_df['feature'] == row, ['class0_enriched', 'class0_depleted']] = summary_df.loc[summary_df['feature'] == row, ['class0_depleted', 'class0_enriched']].values
            summary_df.loc[summary_df['feature'] == row, ['class1_enriched', 'class1_depleted']] = summary_df.loc[summary_df['feature'] == row, ['class1_depleted', 'class1_enriched']].values
            summary_df.loc[summary_df['feature'] == row, ['class2_enriched', 'class2_depleted']] = summary_df.loc[summary_df['feature'] == row, ['class2_depleted', 'class2_enriched']].values
            summary_df.loc[summary_df['feature'] == row, ['class3_enriched', 'class3_depleted']] = summary_df.loc[summary_df['feature'] == row, ['class3_depleted', 'class3_enriched']].values
        
    # create new dataframe with the proportions of significant features in each category
    prop_df = pd.DataFrame()
    prop_df['class0_enriched'] = summary_df.groupby(['feature_category'])['class0_enriched'].sum()/summary_df.groupby(['feature_category'])['class0_enriched'].count()
    prop_df['class0_depleted'] = summary_df.groupby(['feature_category'])['class0_depleted'].sum()/summary_df.groupby(['feature_category'])['class0_depleted'].count()
    prop_df['class1_enriched'] = summary_df.groupby(['feature_category'])['class1_enriched'].sum()/summary_df.groupby(['feature_category'])['class1_enriched'].count()
    prop_df['class1_depleted'] = summary_df.groupby(['feature_category'])['class1_depleted'].sum()/summary_df.groupby(['feature_category'])['class1_depleted'].count()
    prop_df['class2_enriched'] = summary_df.groupby(['feature_category'])['class2_enriched'].sum()/summary_df.groupby(['feature_category'])['class2_enriched'].count()
    prop_df['class2_depleted'] = summary_df.groupby(['feature_category'])['class2_depleted'].sum()/summary_df.groupby(['feature_category'])['class2_depleted'].count()
    prop_df['class3_enriched'] = summary_df.groupby(['feature_category'])['class3_enriched'].sum()/summary_df.groupby(['feature_category'])['class3_enriched'].count()
    prop_df['class3_depleted'] = summary_df.groupby(['feature_category'])['class3_depleted'].sum()/summary_df.groupby(['feature_category'])['class3_depleted'].count()
    
    # negate depleted columns
    prop_df['class0_depleted'] = -prop_df['class0_depleted']
    prop_df['class1_depleted'] = -prop_df['class1_depleted']
    prop_df['class2_depleted'] = -prop_df['class2_depleted']
    prop_df['class3_depleted'] = -prop_df['class3_depleted']

    # sum negative depleted columns with positive enriched columns
    prop_df['class0_max'] = prop_df[['class0_enriched', 'class0_depleted']].sum(axis=1)
    prop_df['class1_max'] = prop_df[['class1_enriched', 'class1_depleted']].sum(axis=1)
    prop_df['class2_max'] = prop_df[['class2_enriched', 'class2_depleted']].sum(axis=1)
    prop_df['class3_max'] = prop_df[['class3_enriched', 'class3_depleted']].sum(axis=1)

    # drop the enriched and depleted columns
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'] 
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['0', '1', '2', '3']

    # HORIZONTAL LINE PLOT
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))
    
    # CORRELATION BETWEEN SPARK AND SSC GROUPS
    plot_df = prop_df.T
    plot_df['cluster'] = np.arange(4)
    polar = plot_df.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    polar.rename(columns={'value': 'ssc_value'}, inplace=True)
    
    # plot comparison between SPARK and SSC
    spark_df = spark_prop_df.T
    spark_df['cluster'] = np.arange(4)
    spark = spark_df.groupby('cluster').mean().reset_index()
    spark = pd.melt(spark, id_vars=['cluster'])
    spark.rename(columns={'value': 'spark_value'}, inplace=True)

    # merge spark and ssc dataframes
    polar = pd.merge(polar, spark, on=['cluster', 'variable'], how='inner')

    r, p = pearsonr(polar['ssc_value'], polar['spark_value'])

    category_correlations = []
    for i, feature in enumerate(features_to_visualize):
        category_correlations.append(pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], \
                                           polar.loc[polar["variable"] == i, "spark_value"])[0])

    return r, category_correlations


def get_feature_enrichments_3classes(mixed_data, summarize=False):
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary
            binary_features.append(feature)
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))

            sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
            sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue

            sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
            sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue

            sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
            sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue

            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])

            feature_to_pval[feature] = [sf0, sf1, sf2]
            feature_sig_df_high[feature] = [sf0, sf1, sf2]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2]
            mean_values[feature] = [mean0, mean1, mean2]

        elif len(unique) > 2:  ## continuous or categorical
            pval_class0 = ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue

            pval_class0_less = ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])

            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2]
            mean_values[feature] = [mean0, mean1, mean2]

        else:
            continue
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2]
    fold_enrichments['cluster'] = [0, 1, 2]
    mean_values['cluster'] = [0, 1, 2]
    pval_df = pd.DataFrame(columns=np.arange(3), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(3), index=mixed_data.columns)

    for tested_class in range(3):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1 # mark enrichment with 1

        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1 # mark depletion with -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(6), index=mixed_data.columns)
        for i in range(3):
            enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+3] = val
        df = df[[0,3,1,4,2,5]] # rearrange columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted']
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d)
        fold_enrichments = fold_enrichments.drop('cluster', axis=1).T.dropna(axis=0)
        fold_enrichments = pd.concat([fold_enrichments[0], fold_enrichments[0],
                                      fold_enrichments[1], fold_enrichments[1],
                                      fold_enrichments[2], fold_enrichments[2],
                                      ], axis=1)
        fold_enrichments.columns = df.columns[1:]
        df_binary = df.copy()
        df_binary = df_binary.set_index('feature')
        df_binary = df_binary.applymap(lambda x: 1 if x < 0.05 else 0)
        df_binary = df_binary * fold_enrichments
        df_binary = df_binary.groupby(lambda x:x.split('_')[0], axis=1).sum()
        df_binary = df_binary.drop([col for col in df_binary.columns if 'depleted' in col], axis=1)
        df_binary.columns = [col.split('_')[0] for col in df_binary.columns]
        df_binary = df_binary.replace(0, np.nan)
        df_binary = df_binary.drop('mixed_pred', axis=0)
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector


def get_feature_enrichments_5classes(mixed_data, ncomp=5, summarize=False):
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]
    class4 = mixed_data[mixed_data['mixed_pred'] == 4]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary
            binary_features.append(feature)
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            subset_class3 = int(np.sum(class3[feature]))
            total_in_class4 = len(class4[feature])
            subset_class4 = int(np.sum(class4[feature]))

            sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
            sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue

            sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
            sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue

            sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
            sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue

            sf3 = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='greater').pvalue
            sf3_less = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='less').pvalue

            sf4 = binomtest(subset_class4, n=total_in_class4, p=background_prob/total, alternative='greater').pvalue
            sf4_less = binomtest(subset_class4, n=total_in_class4, p=background_prob/total, alternative='less').pvalue

            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background
            fe3 = (subset_class3/total_in_class3)/background
            fe4 = (subset_class4/total_in_class4)/background

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])
            mean4 = np.mean(class4[feature])

            feature_to_pval[feature] = [sf0, sf1, sf2, sf3, sf4]
            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3, sf4]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less, sf3_less, sf4_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3, fe4]
            mean_values[feature] = [mean0, mean1, mean2, mean3, mean4]

        elif len(unique) > 2:  ## continuous or categorical
            pval_class0 = ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature], class3[feature], class4[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature], class3[feature], class4[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature], class3[feature], class4[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class3 = ttest_ind(class3[feature],
                                            pd.concat([class0[feature], class1[feature], class2[feature], class4[feature]],
                                                        ignore_index=True, sort=False), equal_var=False,
                                            alternative='greater').pvalue
            pval_class4 = ttest_ind(class4[feature],
                                            pd.concat([class0[feature], class1[feature], class2[feature], class3[feature]],
                                                        ignore_index=True, sort=False), equal_var=False,
                                            alternative='greater').pvalue

            pval_class0_less = ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature], class3[feature], class4[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature], class3[feature], class4[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature], class3[feature], class4[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class3_less = ttest_ind(class3[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature], class4[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class4_less = ttest_ind(class4[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)
            fe4 = cohens_d(class4[feature], total)

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])
            mean4 = np.mean(class4[feature])

            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3, pval_class4]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3, pval_class4]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, pval_class3_less, pval_class4_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3, fe4]
            mean_values[feature] = [mean0, mean1, mean2, mean3, mean4]

        else:
            continue
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = np.arange(ncomp)
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = np.arange(ncomp)
    fold_enrichments['cluster'] = np.arange(ncomp)
    mean_values['cluster'] = np.arange(ncomp)
    pval_df = pd.DataFrame(columns=np.arange(ncomp), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(ncomp), index=mixed_data.columns)

    for tested_class in range(ncomp):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1 # mark enrichment with 1

        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1 # mark depletion with -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(2*ncomp), index=mixed_data.columns)
        for i in range(ncomp):
            enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+ncomp] = val
        df = df[[0,5,1,6,2,7,3,8,4,9]] # rearrange columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 
                      'class3_enriched', 'class3_depleted', 'class4_enriched', 'class4_depleted']
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d)
        fold_enrichments = fold_enrichments.drop('cluster', axis=1).T.dropna(axis=0)
        fold_enrichments = pd.concat([fold_enrichments[0], fold_enrichments[0],
                                      fold_enrichments[1], fold_enrichments[1],
                                      fold_enrichments[2], fold_enrichments[2],
                                        fold_enrichments[3], fold_enrichments[3],
                                        fold_enrichments[4], fold_enrichments[4]
                                      ], axis=1)
        fold_enrichments.columns = df.columns[1:]
        df_binary = df.copy()
        df_binary = df_binary.set_index('feature')
        df_binary = df_binary.applymap(lambda x: 1 if x < 0.05 else 0)
        df_binary = df_binary * fold_enrichments
        df_binary = df_binary.groupby(lambda x:x.split('_')[0], axis=1).sum()
        df_binary = df_binary.drop([col for col in df_binary.columns if 'depleted' in col], axis=1)
        df_binary.columns = [col.split('_')[0] for col in df_binary.columns]
        df_binary = df_binary.replace(0, np.nan)
        df_binary = df_binary.drop('mixed_pred', axis=0)
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector


def get_feature_enrichments_6classes(mixed_data, ncomp=6, summarize=False):
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]
    class4 = mixed_data[mixed_data['mixed_pred'] == 4]
    class5 = mixed_data[mixed_data['mixed_pred'] == 5]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary
            binary_features.append(feature)
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            subset_class3 = int(np.sum(class3[feature]))
            total_in_class4 = len(class4[feature])
            subset_class4 = int(np.sum(class4[feature]))
            total_in_class5 = len(class5[feature])
            subset_class5 = int(np.sum(class5[feature]))

            sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
            sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue

            sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
            sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue

            sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
            sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue

            sf3 = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='greater').pvalue
            sf3_less = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='less').pvalue

            sf4 = binomtest(subset_class4, n=total_in_class4, p=background_prob/total, alternative='greater').pvalue
            sf4_less = binomtest(subset_class4, n=total_in_class4, p=background_prob/total, alternative='less').pvalue

            sf5 = binomtest(subset_class5, n=total_in_class5, p=background_prob/total, alternative='greater').pvalue
            sf5_less = binomtest(subset_class5, n=total_in_class5, p=background_prob/total, alternative='less').pvalue

            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background
            fe3 = (subset_class3/total_in_class3)/background
            fe4 = (subset_class4/total_in_class4)/background
            fe5 = (subset_class5/total_in_class5)/background

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])
            mean4 = np.mean(class4[feature])
            mean5 = np.mean(class5[feature])

            feature_to_pval[feature] = [sf0, sf1, sf2, sf3, sf4, sf5]
            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3, sf4, sf5]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less, sf3_less, sf4_less, sf5_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3, fe4, fe5]
            mean_values[feature] = [mean0, mean1, mean2, mean3, mean4, mean5]

        elif len(unique) > 2:  ## continuous or categorical
            pval_class0 = ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature], class3[feature], class4[feature], class5[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature], class3[feature], class4[feature], class5[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature], class3[feature], class4[feature], class5[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class3 = ttest_ind(class3[feature],
                                            pd.concat([class0[feature], class1[feature], class2[feature], class4[feature], class5[feature]],
                                                        ignore_index=True, sort=False), equal_var=False,
                                            alternative='greater').pvalue
            pval_class4 = ttest_ind(class4[feature],
                                            pd.concat([class0[feature], class1[feature], class2[feature], class3[feature], class5[feature]],
                                                        ignore_index=True, sort=False), equal_var=False,
                                            alternative='greater').pvalue
            pval_class5 = ttest_ind(class5[feature],
                                            pd.concat([class0[feature], class1[feature], class2[feature], class3[feature], class4[feature]],
                                                        ignore_index=True, sort=False), equal_var=False,
                                            alternative='greater').pvalue

            pval_class0_less = ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature], class3[feature], class4[feature], class5[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature], class3[feature], class4[feature], class5[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature], class3[feature], class4[feature], class5[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class3_less = ttest_ind(class3[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature], class4[feature], class5[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class4_less = ttest_ind(class4[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature], class3[feature], class5[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class5_less = ttest_ind(class5[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature], class3[feature], class4[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)
            fe4 = cohens_d(class4[feature], total)
            fe5 = cohens_d(class5[feature], total)

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])
            mean4 = np.mean(class4[feature])
            mean5 = np.mean(class5[feature])

            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3, pval_class4, pval_class5]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3, pval_class4, pval_class5]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, pval_class3_less, pval_class4_less, pval_class5_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3, fe4, fe5]
            mean_values[feature] = [mean0, mean1, mean2, mean3, mean4, mean5]

        else:
            continue
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = np.arange(ncomp)
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = np.arange(ncomp)
    fold_enrichments['cluster'] = np.arange(ncomp)
    mean_values['cluster'] = np.arange(ncomp)
    pval_df = pd.DataFrame(columns=np.arange(ncomp), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(ncomp), index=mixed_data.columns)

    for tested_class in range(ncomp):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1 # mark enrichment with 1

        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1 # mark depletion with -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(2*ncomp), index=mixed_data.columns)
        for i in range(ncomp):
            enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+ncomp] = val
        df = df[[0,6,1,7,2,8,3,9,4,10,5,11]]
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 
                      'class3_enriched', 'class3_depleted', 'class4_enriched', 'class4_depleted', 'class5_enriched', 'class5_depleted']
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d)
        fold_enrichments = fold_enrichments.drop('cluster', axis=1).T.dropna(axis=0)
        fold_enrichments = pd.concat([fold_enrichments[0], fold_enrichments[0],
                                      fold_enrichments[1], fold_enrichments[1],
                                      fold_enrichments[2], fold_enrichments[2],
                                        fold_enrichments[3], fold_enrichments[3],
                                        fold_enrichments[4], fold_enrichments[4],
                                        fold_enrichments[5], fold_enrichments[5]
                                      ], axis=1)
        fold_enrichments.columns = df.columns[1:]
        df_binary = df.copy()
        df_binary = df_binary.set_index('feature')
        df_binary = df_binary.applymap(lambda x: 1 if x < 0.05 else 0)
        df_binary = df_binary * fold_enrichments
        df_binary = df_binary.groupby(lambda x:x.split('_')[0], axis=1).sum()
        df_binary = df_binary.drop([col for col in df_binary.columns if 'depleted' in col], axis=1)
        df_binary.columns = [col.split('_')[0] for col in df_binary.columns]
        df_binary = df_binary.replace(0, np.nan)
        df_binary = df_binary.drop('mixed_pred', axis=0)
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector


def generate_summary_table(df_enriched_depleted, fold_enrichments, ncomp):
    """
    Generate figures to summarize phenotypes enriched and depleted in each class, 
    where the number of classes is determined by 'ncomp' (between 3 and 6).
    """
    if not (3 <= ncomp <= 6):
        raise ValueError("ncomp should be between 3 and 6")

    features_to_exclude = fold_enrichments.copy()

    # Compute absolute values for fold enrichments for all classes
    for cls in range(ncomp):
        features_to_exclude[f'class{cls}'] = features_to_exclude[f'class{cls}'].abs()

    binary_features = [
        'repeat_grade', 'q01_phrases', 'q02_conversation', 'q03_odd_phrase',
        'q04_inappropriate_question', 'q05_pronouns_mixed', 'q06_invented_words',
        'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate',
        'q10_hand_tool', 'q11_interest_preoccupy', 'q12_parts_object',
        'q13_interests_intensity', 'q14_senses', 'q15_odd_ways',
        'q16_complicated_movements', 'q17_injured_deliberately',
        'q18_objects_carry', 'q19_best_friend', 'q20_talk_friendly',
        'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted',
        'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 
        'q27_smile_back', 'q28_things_interested', 'q29_share', 
        'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention', 
        'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe',
        'q36_same_age', 'q37_respond_positively', 'q38_pay_attention', 
        'q39_imaginative_games', 'q40_cooperatively_games'
    ]

    # Select features to exclude based on their presence in all classes
    nan_condition = (features_to_exclude[f'class{cls}'].isna() for cls in range(ncomp))
    nan_features = features_to_exclude.loc[np.all(list(nan_condition), axis=0)]
    
    low_features_continuous = features_to_exclude.loc[
        ~features_to_exclude['feature'].isin(binary_features) &
        np.all([features_to_exclude[f'class{cls}'] < 0.2 for cls in range(ncomp)], axis=0)
    ]
    
    low_features_binary = features_to_exclude.loc[
        features_to_exclude['feature'].isin(binary_features) &
        np.all([features_to_exclude[f'class{cls}'] < 1.5 for cls in range(ncomp)], axis=0)
    ]
    
    features_to_exclude = pd.concat([nan_features, low_features_continuous, low_features_binary])
    features_to_exclude = features_to_exclude['feature'].unique()

    # Load feature-category mapping
    features_to_category = pd.read_csv(
        '../PhenotypeValidations/data/feature_to_category_mapping.csv',
        index_col=None
    )
    feature_to_category = dict(zip(
        features_to_category['feature'], features_to_category['category'])
    )
    
    df = df_enriched_depleted.copy().fillna('NaN')
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    
    df = df.loc[~df['feature'].isin(features_to_exclude)]
    df['feature_category'] = df['feature'].map(feature_to_category)
    df = df.dropna(subset=['feature_category']).replace('NaN', 1)

    # Mark features as enriched or depleted based on p-value threshold
    for cls in range(ncomp):
        df[f'class{cls}_enriched'] = df[f'class{cls}_enriched'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
        df[f'class{cls}_depleted'] = df[f'class{cls}_depleted'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )

    # Flip rows for specific features as per your requirement
    flip_rows = [
        'q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 
        'q20_talk_friendly', 'q21_copy_you', 'q22_point_things',
        'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 
        'q26_look_directly', 'q27_smile_back', 'q28_things_interested', 
        'q29_share', 'q30_join_enjoyment', 'q31_comfort', 
        'q32_help_attention', 'q33_range_expressions', 
        'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 
        'q37_respond_positively', 'q38_pay_attention', 
        'q39_imaginative_games', 'q40_cooperatively_games'
    ]
    
    for row in flip_rows: 
        for cls in range(ncomp):
            df.loc[df['feature'] == row, [f'class{cls}_enriched', f'class{cls}_depleted']] = \
                df.loc[df['feature'] == row, [f'class{cls}_depleted', f'class{cls}_enriched']].values

    prop_df = pd.DataFrame()

    # Calculate proportion of enriched and depleted features by category
    for cls in range(ncomp):
        prop_df[f'class{cls}_enriched'] = df.groupby(['feature_category'])[f'class{cls}_enriched'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_enriched'].count()
        prop_df[f'class{cls}_depleted'] = df.groupby(['feature_category'])[f'class{cls}_depleted'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_depleted'].count()
        prop_df[f'class{cls}_depleted'] = -prop_df[f'class{cls}_depleted']
        prop_df[f'class{cls}_max'] = prop_df[[f'class{cls}_enriched', f'class{cls}_depleted']].sum(axis=1)
    
    df = prop_df.drop([f'class{cls}_max' for cls in range(ncomp)], axis=1)
    df = df.loc[~df.index.isin(['somatic', 'other problems', 'thought problems'])]

    # Prepare the proportions data
    proportions = pd.DataFrame(index=df.index)
    for cls in range(ncomp):
        proportions[f'class{cls}_enriched'] = df[f'class{cls}_enriched']
        proportions[f'class{cls}_depleted'] = df[f'class{cls}_depleted']
    
    proportions = proportions.reset_index()
    proportions = proportions.set_index('feature_category').reindex([
        'anxiety/mood', 'attention', 'disruptive behavior', 
        'self-injury', 'social/communication', 'restricted/repetitive', 
        'developmental'
    ]).reset_index()

    proportions_melted = pd.melt(
        proportions, 
        id_vars=['feature_category'], 
        value_vars=[f'class{cls}_enriched' for cls in range(ncomp)] + [f'class{cls}_depleted' for cls in range(ncomp)],
        var_name='class', 
        value_name='proportion'
    )
    
    proportions_melted['type'] = proportions_melted['class'].apply(lambda x: x.split('_')[1])
    proportions_melted['class'] = proportions_melted['class'].apply(lambda x: x.split('_')[0])

    # plot variation figure (supplementary figure)
    fig, ax = plt.subplots(figsize=(12, 5))    
    feature_categories = proportions['feature_category'].unique()
    classes = [f'class{cls}' for cls in range(ncomp)]
    bar_width = 0.1
    n_classes = len(classes)
    spacing = 0.2
    group_width = n_classes * bar_width + spacing
    bar_positions = np.arange(len(feature_categories)) * group_width
    
    class_colors = {
        'class0': '#F85C50',
        'class1': '#6E1E76',
        'class2': '#1CA4B8',
        'class3': '#0073B7',
        'class4': '#F5A623',
        'class5': '#2C3E50'
    }
    
    for idx, cls in enumerate(classes):
        enriched = proportions_melted[
            (proportions_melted['class'] == cls) & 
            (proportions_melted['type'] == 'enriched')
        ]
        depleted = proportions_melted[
            (proportions_melted['class'] == cls) & 
            (proportions_melted['type'] == 'depleted')
        ]
        
        ax.bar(
            bar_positions + idx * bar_width, 
            depleted['proportion'], 
            width=bar_width, 
            label=f'{cls} depleted', 
            linewidth=0, 
            color=class_colors[cls]
        )
        
        ax.bar(
            bar_positions + idx * bar_width, 
            enriched['proportion'], 
            width=bar_width, 
            label=f'{cls} enriched', 
            linewidth=0, 
            color=class_colors[cls]
        )
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(bar_positions + (n_classes / 2 - 0.5) * bar_width)
    ax.set_xticklabels(feature_categories, rotation=35, ha='right', fontsize=14)
    ax.set_ylabel('Proportion of features', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/GFMM_{ncomp}classes_variation_figure.png', dpi=600)
    plt.close()

    # Prepare data for the main horizontal line plot
    prop_df = prop_df.drop(
        [
            f'class{cls}_enriched' for cls in range(ncomp)] +
            [f'class{cls}_depleted' for cls in range(ncomp)
        ], 
        axis=1
    )
    prop_df.columns = [str(i) for i in range(ncomp)]

    features_to_visualize = [
        'anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 
        'restricted/repetitive', 'social/communication', 'developmental'
    ]
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))

    # Create the main horizontal line plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    palette = class_colors.values()

    ax = sns.lineplot(
        data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3
    )
    ax.set(xlabel="Phenotype Category", ylabel="")

    plt.xticks(
        ha='right', rotation=30, fontsize=16,
        ticks=np.arange(len(features_to_visualize)),
        labels=[
            'anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 
            'restricted/repetitive', 'limited social/communication', 
            'developmental delay'
        ]
    )
    plt.ylim([-1.1, 1.1])

    for line in ax.lines:
        line.set_linewidth(5)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linewidth=1)

    ax.get_legend().remove()
    ax.tick_params(labelsize=20)

    plt.xlabel('')
    plt.title(f'Feature enrichment by category ({ncomp} classes)', fontsize=24)
    ax.set_ylabel('Proportion and direction', fontsize=18)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')

    plt.savefig(
        f'figures/{ncomp}classes_phenotype_categories_horizontal_lineplot.png', 
        bbox_inches='tight', dpi=300
    )
    plt.close()


def scq_and_developmental_milestones_validation(gfmm_labels, ncomp):
    # get sibling data for background history
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    bhdf = pd.read_csv(
        f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv'
        )
    bhdf = bhdf.loc[
        (bhdf['age_at_eval_years'] <= 18) & (bhdf['age_at_eval_years'] >= 4)
        ]
    dev_milestones = ['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 
                      'walked_age_mos', 'fed_self_spoon_age_mos', 'used_words_age_mos', 
                      'combined_words_age_mos', 'combined_phrases_age_mos',
                      'bladder_trained_age_mos', 'bowel_trained_age_mos']
    bhdf = bhdf.set_index('subject_sp_id',drop=True)[dev_milestones]

    # subset to paired sibs
    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sib_data = pd.merge(bhdf, paired_sibs, left_index=True, right_index=True)
    sib_bh_data = sib_data[dev_milestones].dropna().astype(float)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'axes.titlepad': 20})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 16.5))
    
    # plot first milestone
    milestone = 'used_words_age_mos'
    milestone_data = gfmm_labels[[milestone, 'mixed_pred']]
    all_proband_bh_data = milestone_data[milestone].astype(float).to_list()
    class0 = milestone_data[
        milestone_data['mixed_pred'] == 0][milestone].astype(float).to_list()
    class1 = milestone_data[
        milestone_data['mixed_pred'] == 1][milestone].astype(float).to_list()
    class2 = milestone_data[
        milestone_data['mixed_pred'] == 2][milestone].astype(float).to_list()
    class3 = milestone_data[
        milestone_data['mixed_pred'] == 3][milestone].astype(float).to_list()

    pvals = []
    pvals.append(ttest_ind(
        class0, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class1, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class2, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class3, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals = multipletests(pvals, method='fdr_bh')[1]
    print(milestone)
    print(pvals)
    
    sns.boxplot(data=[
        sib_bh_data[milestone].to_list(), class0, class1, class2, class3], 
        showfliers=True, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
        whiskerprops = dict(color = "black", linewidth=2), 
        capprops = dict(color = "black", linewidth=2),
        medianprops=dict(color='white', linewidth=2), 
        boxprops=dict(edgecolor='white', linewidth=0.5), ax=ax1)
    ax1.set_ylim([0, 90])
    yticks = ax1.get_yticks()
    yticks = [int(y) for y in yticks]
    ax1.set_yticks(yticks)
    ax1.set_xlabel('')
    ax1.set_ylabel('Months', fontsize=20)
    ax1.set_title('Age first used words', fontsize=22)
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=16)
    ax1.set_xticklabels([])
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color('black')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # plot second milestone
    milestone = 'walked_age_mos'
    milestone_data = gfmm_labels[[milestone, 'mixed_pred']]
    all_proband_bh_data = milestone_data[milestone].astype(float).to_list()
    class0 = milestone_data[
        milestone_data['mixed_pred'] == 0][milestone].astype(float).to_list()
    class1 = milestone_data[
        milestone_data['mixed_pred'] == 1][milestone].astype(float).to_list()
    class2 = milestone_data[
        milestone_data['mixed_pred'] == 2][milestone].astype(float).to_list()
    class3 = milestone_data[
        milestone_data['mixed_pred'] == 3][milestone].astype(float).to_list()

    pvals = []
    pvals.append(ttest_ind(
        class0, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class1, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class2, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals.append(ttest_ind(
        class3, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue)
    pvals = multipletests(pvals, method='fdr_bh')[1]
    print(milestone)
    print(pvals)
    
    sns.boxplot(data=[
        sib_bh_data[milestone].to_list(), class0, class1, class2, class3], 
        showfliers=True, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
        whiskerprops = dict(color = "black", linewidth=2), 
        capprops = dict(color = "black", linewidth=2),
        medianprops=dict(color='white', linewidth=2), 
        boxprops=dict(edgecolor='white', linewidth=0.5), ax=ax2)
    ax2.set_ylim([0, 45])
    ax2.set_xlabel('')
    ax2.set_ylabel('Months', fontsize=20)
    yticks = ax2.get_yticks()
    yticks = [int(y) for y in yticks]
    ax2.set_yticks(yticks)
    ax2.set_title('Age first walked', fontsize=22)
    plt.rcParams['axes.titlepad'] = 20
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=16)
    ax2.set_xticklabels([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_color('black')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # plot scq total score
    # get sibling data for SCQ
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv')
    scqdf = scqdf.loc[(
        scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) 
        & (scqdf['age_at_eval_years'] >= 4)]
    scqdf = scqdf.set_index('subject_sp_id',drop=True).drop([
        'respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id',
        'current_depend_adult','age_at_eval_months','scq_measure_validity_flag',
        'eval_year','missing_values','summary_score'],axis=1)
    scqdf = scqdf[scqdf['asd'] == 0]
    
    # intersect with paired sibs
    sib_data = pd.merge(scqdf, paired_sibs, left_index=True, right_index=True)
    sib_scq_data = sib_data['final_score'].dropna().astype(int).to_list()

    # get total scores for each class
    final_score = gfmm_labels[['final_score', 'mixed_pred']]
    all_proband_scq_data = final_score['final_score'].dropna().astype(int).to_list()
    class0 = final_score[final_score[
        'mixed_pred'] == 0]['final_score'].dropna().astype(int).to_list()
    class1 = final_score[final_score[
        'mixed_pred'] == 1]['final_score'].dropna().astype(int).to_list()
    class2 = final_score[final_score[
        'mixed_pred'] == 2]['final_score'].dropna().astype(int).to_list()
    class3 = final_score[final_score[
        'mixed_pred'] == 3]['final_score'].dropna().astype(int).to_list()

    # hypothesis testing vs. sibs
    p_vals = []
    p_vals.append(ttest_ind(
        class0, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    p_vals.append(ttest_ind(
        class1, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    p_vals.append(ttest_ind(
        class2, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    p_vals.append(ttest_ind(
        class3, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    # FDR correction
    p_vals = multipletests(p_vals, method='fdr_bh')[1]
    print('SCQ total score')
    print(p_vals)
    data = [sib_scq_data, class0, class1, class2, class3]
    sns.boxplot(
        data=data, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
        showfliers=True, whiskerprops = dict(color = "black", linewidth=2), 
        capprops = dict(color = "black", linewidth=2),
        medianprops=dict(color='white', linewidth=2), 
        boxprops=dict(edgecolor='white', linewidth=0.5), ax=ax3)
    ax3.set_xlabel('')
    ax3.set_ylabel('Total Score', fontsize=20)
    plt.rcParams['axes.titlepad'] = 20
    plt.title('Social Communication Questionnaire', fontsize=22)
    ax3.set_xticklabels([])
    yticks = ax3.get_yticks()
    yticks = [int(y) for y in yticks]
    ax3.set_yticks(yticks)
    ax3.set_ylim([0, 40])
    ax3.set_yticklabels(ax3.get_yticks(), fontsize=16)
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(1.5)
        ax3.spines[axis].set_color('black')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.savefig(f'figures/{ncomp}classes_pheno_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close()
