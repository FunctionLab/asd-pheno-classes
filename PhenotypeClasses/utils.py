import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, binomtest
from statsmodels.stats.multitest import multipletests


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
    """
    Pefrom multiple testing correction on p-values.
    """
    return multipletests(p_values, method=method)[1]


def split_columns(feature_subset):
    """
    Given list of features (strings), return stratified lists of 
    continuous, binary, and categorical lists containing the feature subset.
    """
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
    """
    Compute feature enrichments and depletions for each class.
    """
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    # isolate the four classes
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary feature
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

            # binomial hypothesis test in both directions
            sf0 = binomtest(
                subset_class0, n=total_in_class0, p=background_prob/total, 
                alternative='greater').pvalue
            sf0_less = binomtest(
                subset_class0, n=total_in_class0, p=background_prob/total, 
                alternative='less').pvalue

            sf1 = binomtest(
                subset_class1, n=total_in_class1, p=background_prob/total, 
                alternative='greater').pvalue
            sf1_less = binomtest(
                subset_class1, n=total_in_class1, p=background_prob/total, 
                alternative='less').pvalue

            sf2 = binomtest(
                subset_class2, n=total_in_class2, p=background_prob/total, 
                alternative='greater').pvalue
            sf2_less = binomtest(
                subset_class2, n=total_in_class2, p=background_prob/total, 
                alternative='less').pvalue

            sf3 = binomtest(
                subset_class3, n=total_in_class3, p=background_prob/total, 
                alternative='greater').pvalue
            sf3_less = binomtest(
                subset_class3, n=total_in_class3, p=background_prob/total, 
                alternative='less').pvalue

            # compute fold enrichment
            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background
            fe3 = (subset_class3/total_in_class3)/background

            # compute mean values
            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            # store results for feature
            feature_to_pval[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less, sf3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        elif len(unique) > 2:  ## continuous or categorical feature
            #  t-test hypothesis testing in both directions
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
            
            # compute Cohen's d
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)

            # compute mean values
            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            # store results for feature
            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, 
                                           pval_class3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        else:
            continue
    
    # format results
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2, 3]
    fold_enrichments['cluster'] = [0, 1, 2, 3]
    mean_values['cluster'] = [0, 1, 2, 3]
    pval_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)

    # perform multiple testing correction for each class and direction
    for tested_class in range(4):
        enriched_class_high = feature_sig_norm_high[
            feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                        axis=1).T.dropna(axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[
            enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1  

        enriched_class_low = feature_sig_norm_low[
            feature_sig_norm_low['cluster'] == tested_class].drop('cluster', axis=1).T.dropna(
                axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[
            enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(8), index=mixed_data.columns)
        for i in range(4):
            enriched_class_high = feature_sig_norm_high[
                feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[
                feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[
                enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(
                    ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[
                enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(
                    ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+4] = val
        
        df = df[[0,4,1,5,2,6,3,7]] # rearrange columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 
                      'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted']
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