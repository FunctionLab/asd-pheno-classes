import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import scipy.stats as stats
from stepmix.stepmix import StepMix
from sklearn.metrics import rand_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats
from scipy.stats import hypergeom, pearsonr, spearmanr
import pickle as rick
import plotly.express as px
from statsmodels.stats.multitest import multipletests
from sklearn.impute import KNNImputer

from latent_class_analysis import get_feature_enrichments, plot_line_polar, get_feature_enrichments_3classes
from GFMM_model_validation import plot_posterior_probs, scramble_column


def plot_line_polar_ssc_version(data, output):
    y = list(data['mixed_pred'])
    X = data.drop(['mixed_pred'], axis=1)
    
    # normalize:
    scaler = StandardScaler()
    mixed_data_norm = scaler.fit_transform(X)
    mixed_data_norm = pd.DataFrame(mixed_data_norm, columns=X.columns)
    mixed_data_norm['cluster'] = y
    
    # 'final_score', 'dev_lang_dis'
    features_to_visualize = ['social_problems_t_score', 'anxiety_problems_t_score', 'i_stereotyped_behavior_score', 'vi_restricted_behavior_score', 'iii_compulsive_behavior_score', 'dev_id', 'cluster'] # for cbcl model
    #features_to_visualize = ['i_stereotyped_behavior_score', 'vi_restricted_behavior_score', 'v_sameness_behavior_score', 'final_score', 'dev_id', 'dev_lang_dis', 'cluster'] # for WGS model

    selected_features = mixed_data_norm[[x for x in features_to_visualize if x in mixed_data_norm.columns]]
    print(selected_features.columns)
    
    ### FINAL PLOT SUMMARIZING THE 4 CLASSES
    polar = selected_features.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    #fig = px.line_polar(polar, r="value", theta="variable", color="cluster", line_close=True, height=800, width=1400)
    #fig.write_image("GFMM_all_figures/nclasses4_final_summary_line_polar.png")
    #print(polar)

    ### option to CUSTOMIZE the plot to only include one class at a time
    polar1 = polar[polar['cluster'] == 0] # only cluster 1
    polar2 = polar[(polar['cluster'] == 0) | (polar['cluster'] == 1)] # only 1 and 2
    polar3 = polar[(polar['cluster'] == 0) | (polar['cluster'] == 1) | (polar['cluster'] == 2)]
    polar4 = polar # all classes
    
    # rename variables for plotting
    polar = polar4
    polar['variable'] = polar['variable'].replace('i_stereotyped_behavior_score', 'Stereotyped Behavior')
    polar['variable'] = polar['variable'].replace('social_problems_t_score', 'Social Problems')
    polar['variable'] = polar['variable'].replace('anxiety_problems_t_score', 'Anxiety Problems')
    polar['variable'] = polar['variable'].replace('iii_compulsive_behavior_score', 'Compulsive Behavior')
    polar['variable'] = polar['variable'].replace('vi_restricted_behavior_score', 'Restricted Behavior')
    polar['variable'] = polar['variable'].replace('v_sameness_behavior_score', 'Sameness Behavior')
    polar['variable'] = polar['variable'].replace('dev_id', 'Intellectual Disability')
    polar['variable'] = polar['variable'].replace('derived_cog_impair', 'Intellectual Disability')
    polar['variable'] = polar['variable'].replace('final_score', 'SCQ Score')
    #polar4['variable'] = polar4['variable'].replace('ml_predicted_cog_impair', 'Intellectual Disability (ML-Predicted)')
    polar['variable'] = polar['variable'].replace('dev_lang_dis', 'Language Delay')

    #colors = ['violet','green','blue','red'] # CBCL COHORT
    colors = ['violet','red','green','blue'] # WGS COHORT
    fig = px.line_polar(polar, r="value", theta="variable", color="cluster", 
                        color_discrete_sequence=colors, line_close=True, height=800, width=1400) # template="plotly_dark",
    fig.update_layout(
        font_size=26
    )
    fig.write_image(f"{output}_polar.png")

def split_columns(feature_subset):
    '''given list of features (strings), return stratified lists of continuous, binary, and categorical lists containing the feature subset.'''
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
                          'iv_ritualistic_behavior_score', 'v_sameness_behavior_score', 'vi_restricted_behavior_score', 'final_score', 'add_adhd_t_score', 'affective_problems_t_score',
                            'aggressive_behavior_t_score', 'anxiety_problems_t_score',
                            'anxious_depressed_t_score', 'attention_problems_t_score',
                            'conduct_problems_t_score', 'externalizing_problems_t_score',
                            'internalizing_problems_t_score', 'oppositional_defiant_t_score',
                            'rule_breaking_t_score', 'social_problems_t_score', 'social_t_score',
                            'somatic_complaints_t_score', 'somatic_prob_t_score',
                            'thought_problems_t_score', 'total_problems_t_score',
                            'withdrawn_t_score', 'age_bladder_trained_day', 'age_bowel_trained',
                            'age_combined_words_short_sen', 'age_crawled', 'age_fed_self_w_spoon',
                            'age_gave_up_bottle', 'age_sat_wo_support', 'age_smiled',
                            'age_started_solid_foods', 'age_used_words', 'age_walked_alone', 
                            'ssc_diagnosis_full_scale_iq', 'ssc_diagnosis_nonverbal_iq', 'ssc_diagnosis_verbal_iq']

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

def get_SSC_data(ncomp=4, bootstrap=False, impute=False):
    """
    finaldf = generate_ssc_data(impute=impute)
    print(finaldf.shape)

    if bootstrap:
        num_bootstraps = 5000
        indices = np.arange(len(finaldf))
        # pick random indices with replacement
        selected_indices = np.random.choice(indices, size=num_bootstraps, replace=True)
        finaldf_boot = finaldf.iloc[selected_indices]
        finaldf = finaldf_boot # TURN BOOTSTRAPPING ON/OFF

    ### 2. RUN GFMM ON SSC DATA
    # split data into continuous, binary, and categorical
    ncomp = 4
    Z_p = finaldf[['sex', 'age_at_eval_years']]

    X = finaldf.drop(['sex', 'age_at_eval_years'],
                    axis=1)  # drop asd label and convert to np array

    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200) #random_state=100) 

    # Grid-Search CV for SSC data
    '''
    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
    ### val log likelihood n_class parameter search
    gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
    gs.fit(mixed_data)
    results = pd.DataFrame(gs.cv_results_)
    results["Val_Log_Likelihood"] = results['mean_test_score']
    sns.set_style("darkgrid")
    sns.lineplot(data=results, x='param_n_components', y='Val_Log_Likelihood',
                 palette='Dark2')
    plt.ylabel('Val Log Likelihood')
    plt.title('SSC parameter search')
    plt.savefig('GFMM_all_figures/SSC_val_likelihood_class_search.png')
    exit()
    
    ### AIC/BIC analysis for n_class parameter search
    results = dict(param_n_components=[], aic=[], bic=[])
    for g in ParameterGrid(grid):
        model.set_params(**g)
        model.fit(mixed_data)
        results['param_n_components'].append(g['n_components'])
        results['aic'].append(model.aic(mixed_data))
        results['bic'].append(model.bic(mixed_data))

    results = pd.DataFrame(results)
    sns.lineplot(data=results, x='param_n_components', y='aic',
                 palette='Dark2')
    plt.savefig('GFMM_all_figures/aic_grid_search_SSC.png')
    plt.clf()
    sns.lineplot(data=results, x='param_n_components', y='bic',
                 palette='Dark2')
    plt.savefig('GFMM_all_figures/bic_grid_search_SSC.png')
    exit()
    '''
    model.fit(mixed_data)
    pred_categorical = model.predict(mixed_data)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final_noimpute.csv')
    """
    #mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final.csv', index_col=0)
    mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final_noimpute.csv', index_col=0)
    #mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final_10.csv', index_col=0)

    ### HYPERGEOMETRIC TEST
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()

    ## extract values for classes
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    print(f'class0 size: {len(class0)}')
    print(f'class1 size: {len(class1)}')
    print(f'class2 size: {len(class2)}')
    print(f'class3 size: {len(class3)}')

    ssc_pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, summary_df, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)

    features_to_exclude = fold_enrichments.copy()
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # numerically get features to exclude:
    # (1) features with no significant enrichments in any class
    # (2) features with all cohen's d values < 0.2 or FE < 1.5
    # get features where all classes is nan
    binary_features = ['repeat_grade', 'q01_phrases', 'q02_conversation', 'q03_odd_phrase', 'q04_inappropriate_question', 'q05_pronouns_mixed', 'q06_invented_words', 'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate', 'q10_hand_tool', 'q11_interest_preoccupy', 'q12_parts_object', 'q13_interests_intensity', 'q14_senses', 'q15_odd_ways', 'q16_complicated_movements', 'q17_injured_deliberately', 'q18_objects_carry', 'q19_best_friend', 'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention', 'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively', 'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
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

    features_to_category = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))
    
    summary_df = summary_df.fillna('NaN')
    summary_df = summary_df.replace(np.nan, 1)

    #features_to_exclude = [x for x in features_to_exclude if x != 'walked_age_mos']
    summary_df = summary_df.loc[~summary_df['feature'].isin(features_to_exclude)] # remove non-contributory features with no significant enrichments
    ssc_feature_subset = summary_df['feature'].to_list()
    # print number of features after filtering
    #print(f'Number of features: {len(ssc_feature_subset)}'); exit()
    
    summary_df['feature_category'] = summary_df['feature'].map(feature_to_category)
    summary_df = summary_df.dropna(subset=['feature_category'])
    print(summary_df['feature_category'].value_counts())

    ssc_category_to_features = defaultdict(list)
    for index, row in summary_df.iterrows():
        ssc_category_to_features[row['feature_category']].append(row['feature'])
    
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
    prop_df.columns = ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay']
    df = prop_df.loc[features_to_visualize]
    fig, ax = plt.subplots(1,1,figsize=(9.5,7))
    ax = sns.heatmap(df, cmap='coolwarm', annot=False, yticklabels=features_to_visualize)
    plt.ylabel('')
    ax.tick_params(labelsize=20)
    plt.title('SSC', fontsize=26)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    plt.savefig('GFMM_all_figures/SSC_7_pheno_categories_heatmap.png', bbox_inches='tight')
    plt.close()

    # HORIZONTAL LINE PLOT
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    palette = ['violet','red','limegreen','blue']
    ax = sns.lineplot(data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3)    
    ax.set(xlabel="Phenotype Category", ylabel="")
    plt.xticks(rotation=45)
    ax.set_xticklabels(features_to_visualize)
    plt.title('SSC', fontsize=26)
    plt.xticks(ha='right')
    plt.xticks(np.arange(len(features_to_visualize)))
    plt.ylim([-1.1,1.1])
    for line in ax.lines:
        line.set_linewidth(5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.get_legend().remove()
    ax.tick_params(labelsize=20)
    plt.xlabel('')
    ax.set_ylabel('Proportion+direction of sig. features', fontsize=18)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    plt.savefig('GFMM_all_figures/SSC_4_pheno_categories_lineplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(mixed_data['mixed_pred'].value_counts())
    
    # CORRELATION BETWEEN SPARK AND SSC GROUPS
    plot_df = prop_df.T
    plot_df['cluster'] = np.arange(4)
    polar = plot_df.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    polar.rename(columns={'value': 'ssc_value'}, inplace=True)
    
    # plot comparison between SPARK and SSC
    spark, spark_category_to_features = run_spark_model(ssc_feature_subset, subset=True)
    spark.rename(columns={'value': 'spark_value'}, inplace=True)
    spark_df = spark.T
    spark_df['cluster'] = np.arange(4)
    # replace 3 with 0, 0 with 1, 1 with 3 in cluster with mapping to be consistent with SSC classes
    spark_df['cluster'] = spark_df['cluster'].map({1: 0, 0: 1, 2: 2, 3: 3}) # nonimputed data
    #spark_df['cluster'] = spark_df['cluster'].map({3: 0, 0: 1, 2: 2, 1: 3}) # knn7 imputed data
    spark = spark_df.groupby('cluster').mean().reset_index()
    spark = pd.melt(spark, id_vars=['cluster'])
    spark.rename(columns={'value': 'spark_value'}, inplace=True)

    # merge spark and ssc dataframes on [cluster, feature_category]
    polar = pd.merge(polar, spark, on=['cluster', 'variable'], how='inner')
    # plot comparison between SPARK and SSC 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    # plot ssc value, shift the bars to the left
    # create a pos for each row
    polar['pos'] = np.arange(len(polar))
    #polar['variable_shifted_left'] = polar['variable'] - 0.2
    #polar['variable_shifted_right'] = polar['variable'] + 0.2
    for i, row in polar.iterrows():
        ax.bar(row['pos'], row['ssc_value'], color='red', alpha=0.5, width=0.45)
        ax.bar(row['pos'], row['spark_value'], color='blue', alpha=0.5, width=0.45)
    #sns.barplot(x='variable_shifted_left', y='ssc_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, color='red')#, palette=['violet','red','limegreen','blue'])
    #sns.barplot(x='variable_shifted_right', y='spark_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, color='blue')#palette=['violet','red','limegreen','blue'])
    plt.ylabel('Proportion+direction of sig. features', fontsize=18)
    plt.xlabel('')
    # xtick labels 
    ax.set_xticks(polar['pos'])
    print(features_to_visualize)
    ax.set_xticklabels(['anxiety/mood_0', 'anxiety/mood_1', 'anxiety/mood_2', 'anxiety/mood_3', 'attention_0', 'attention_1', 'attention_2', 'attention_3', 'disruptive behavior_0', 'disruptive behavior_1', 'disruptive behavior_2', 'disruptive behavior_3', 'self-injury_0', 'self-injury_1', 'self-injury_2', 'self-injury_3',
                        'limited social/communication_0', 'limited social/communication_1', 'limited social/communication_2', 'limited social/communication_3', 'restricted/repetitive_0', 'restricted/repetitive_1', 'restricted/repetitive_2', 'restricted/repetitive_3', 'developmental delay_0', 'developmental delay_1', 'developmental delay_2', 'developmental delay_3'])
    # create legend of SPARK, SSC with red,
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='SSC', alpha=0.6)
    blue_patch = mpatches.Patch(color='blue', label='SPARK', alpha=0.6)
    plt.legend(handles=[red_patch, blue_patch], title='Dataset', fontsize=16, title_fontsize=16, loc='upper left', bbox_to_anchor=(1.03, 1))
    # remove legend
    #ax.get_legend().remove()
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.savefig(f'GFMM_all_figures/SSC_SPARK_replication_7classes_barplot.png', bbox_inches='tight')
    plt.close()

    '''
    # plot comparison between SPARK and SSC
    # barplot for each category, plot the 4 classes with the two datasets
    # so there should be a total of 8 bars per category
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    # plot ssc value, shift the bars to the left
    polar['variable_shifted_left'] = polar['variable'] - 0.2
    polar['variable_shifted_right'] = polar['variable'] + 0.2
    sns.barplot(x='variable_shifted_left', y='ssc_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, palette=['violet','red','limegreen','blue'])
    # plot spark value
    sns.barplot(x='variable_shifted_right', y='spark_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, palette=['violet','red','limegreen','blue'])
    plt.ylabel('Proportion+direction of sig. features', fontsize=18)
    plt.xlabel('')
    # remove legend
    ax.get_legend().remove()
    plt.xticks(rotation=35, ha='right', fontsize=16)
    plt.savefig(f'GFMM_all_figures/SSC_SPARK_replication_7classes_barplot.png', bbox_inches='tight')
    plt.close()
    '''

    # SCATTER PLOT 
    shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shapes = shapes[:len(features_to_visualize)]
    polar['color'] = polar['cluster'].map({0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
    polar['shape'] = polar['variable'].map({0: 'o', 1: 'v', 2: 'p', 3: '*', 4: 'd', 5: "P", 6: 's', 7: 'X', 8: '>', 9: 'D', 10: '^'})
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i, row in polar.iterrows():
        ax.scatter(row['spark_value'], row['ssc_value'], color=row['color'], marker=row['shape'], s=165, alpha=0.7)
    # fit a line
    x = np.linspace(-1, 1, 100)
    y = x
    ax.plot(x, y, color='gray', linestyle='--')
    r, p = pearsonr(polar['ssc_value'], polar['spark_value'])
    r2 = r**2
    print(p, r, r2); exit()
    # add text to plot with r2 and p value
    ax.text(0.3, 0.11, f'R^2: {r2:.2f}', fontsize=16)
    ax.text(0.3, 0.01, f'p < 1e-5', fontsize=16)
    plt.xlabel('SPARK', fontsize=20)
    plt.ylabel('SSC', fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # make legend for shapes
    for i, shape in enumerate(shapes):
        if features_to_visualize[i] == 'social/communication':
            features_to_visualize[i] = 'limited social/communication'
        elif features_to_visualize[i] == 'developmental':
            features_to_visualize[i] = 'developmental delay'
        ax.scatter([], [], color='black', marker=shape, s=100, label=f'{features_to_visualize[i]}')
    ax.legend(title='', fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.savefig(f'GFMM_all_figures/SSC_SPARK_replication_7classes_scatterplot.png', bbox_inches='tight')
    plt.close()

    # for each feature category, correlate the enrichment patterns between SPARK and SSC
    corr_matrix = np.zeros((len(features_to_visualize), len(features_to_visualize)))
    vals_to_plot = []
    for i, feature in enumerate(features_to_visualize):
        #print(f'{feature} correlation: {pearsonr(polar.loc[polar["feature_category"] == feature, "ssc_value"], polar.loc[polar["feature_category"] == feature, "spark_value"])}')
        corr_matrix[i, i] = pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[0] # get the correlation value
        print(f'{feature} correlation: {pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[0]}')
        print(f'pval: {pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[1]}')
        vals_to_plot.append(pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[0])

    exit()
    
    # plot the correlation matrix
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    sns.heatmap(corr_matrix, cmap="PuRd", annot=False) # BuPu, YlGnBu, PuRd
    ax.set_xticklabels(features_to_visualize, rotation=35, fontsize=16, ha='right')
    ax.set_yticklabels(features_to_visualize, rotation=0, fontsize=16)
    plt.ylabel('SSC', fontsize=24)
    plt.xlabel('SPARK', fontsize=24)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    plt.title(f'SSC Replication: Correlation of enrichment patterns', fontsize=20)
    plt.savefig(f'GFMM_all_figures/SSC_SPARK_replication_7classes_correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    # compute % overlap in each category between spark and ssc
    overlaps = []
    for category in features_to_visualize:
        spark_features = spark_category_to_features[category]
        ssc_features = ssc_category_to_features[category]
        overlap = len(set(spark_features) & set(ssc_features))/len(set(spark_features) | set(ssc_features))
        overlaps.append(overlap)
    print(overlaps)
    
    # bar plot of correlation values
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(x=features_to_visualize, y=vals_to_plot, palette='husl', width=0.45, linewidth=0.8, edgecolor='black')
    plt.ylabel('Pearson r(SPARK, SSC)', fontsize=16)
    plt.xlabel('Category', fontsize=15.5)
    #plt.title(f'SSC Replication: Correlation of enrichment patterns', fontsize=20)
    # legend for colors of bars
    palette = sns.color_palette('husl', n_colors=7)
    for i, color in enumerate(palette):
        ax.bar(0, 0, color=color, label=features_to_visualize[i])
    #ax.legend(title='Category', fontsize=16, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.xticks(np.arange(len(features_to_visualize)), rotation=25, ha='right', fontsize=21)
    ax.set_xticklabels(features_to_visualize, rotation=20, fontsize=14)
    plt.yticks(fontsize=18)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
    plt.savefig(f'GFMM_all_figures/SSC_SPARK_replication_7classes_correlation_barplot.png', bbox_inches='tight')
    plt.close()

    print(f'Correlation values: {vals_to_plot}')


def posterior_prob_validation():
    
    datadf = generate_ssc_data(impute=False)

    age = datadf['age_at_eval_years']
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'age_at_eval_years'], 
                    axis=1)  
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=4, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200)

    model.fit(mixed_data, Z_p)
    posterior_probs = model.predict_proba(mixed_data)
    # take max posterior probability for each sample
    posterior_probs = np.max(posterior_probs, axis=1)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    mixed_data['age'] = age
    labels.to_csv('GFMM_validation_pickles/SSC_labels.csv')

    # scramble each feature in the unlabeled data, retrain new model, and get posterior probabilities for each class
    copydf = datadf.copy()
    scrambled_data = copydf.apply(scramble_column)
    age = scrambled_data['age_at_eval_years']

    # get covariate data
    Z_p = scrambled_data[['sex', 'age_at_eval_years']]
    X = scrambled_data.drop(['sex', 'age_at_eval_years'],
                    axis=1)  # drop asd label and convert to np array
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    scrambled, scrambled_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )
    model_scram = StepMix(n_components=4, measurement=scrambled_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200)
    model_scram.fit(scrambled, Z_p)
    posterior_probs_scram = model_scram.predict_proba(scrambled)
    # take max posterior probability for each sample
    posterior_probs_scram = np.max(posterior_probs_scram, axis=1)
    scrambled['mixed_pred'] = model_scram.predict(scrambled)
    labels_scram = scrambled['mixed_pred']
    labels_scram.to_csv('GFMM_validation_pickles/SSC_labels_scram.csv') 

    # save posterior probabilities 
    with open(f'GFMM_validation_pickles/SSC_posterior_probs.pkl', 'wb') as f:
        rick.dump(posterior_probs, f)
    with open(f'GFMM_validation_pickles/SSC_posterior_probs_scram.pkl', 'wb') as f:
        rick.dump(posterior_probs_scram, f) 
    
    with open(f'GFMM_validation_pickles/SSC_posterior_probs.pkl', 'rb') as f:
        posterior_probs = rick.load(f)
    with open(f'GFMM_validation_pickles/SSC_posterior_probs_scram.pkl', 'rb') as f:
        posterior_probs_scram = rick.load(f)
    labels = pd.read_csv('GFMM_validation_pickles/SSC_labels.csv', index_col=0)
    labels = labels['mixed_pred']
    labels_scram = pd.read_csv('GFMM_validation_pickles/SSC_labels_scram.csv', index_col=0)
    labels_scram = labels_scram['mixed_pred']
    
    #labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0)
    #labels = labels['mixed_pred']
    
    # plot posterior probabilities for each class for scrambled and unscrambled data
    plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, labels_scram, ncomp=4, cohort='SSC')


def run_spark_model(ssc_feature_subset, subset=False):
    mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0)

    # OPTIONAL: subset to ssc features
    if subset:
        ssc_feature_subset = [x for x in ssc_feature_subset if x in mixed_data.columns]
        mixed_data = mixed_data[ssc_feature_subset]

    # get feature enrichments
    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)
    
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
    features_to_category = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))

    df = df_enriched_depleted.copy()
    df = df.fillna('NaN')
    # drop feature category column
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    
    df = df.loc[~df['feature'].isin(features_to_exclude)] # remove non-contributory features with no significant enrichments
    
    # annotate each feature with its category
    df['feature_category'] = df['feature'].map(feature_to_category)
    # drop features with no category
    df = df.dropna(subset=['feature_category'])
    df = df.replace('NaN', 1)
    print(df['feature_category'].value_counts())

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
    prop_df.columns = ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay']
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'] 
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))

    return prop_df, feature_category_to_features
    

def adjust_pvalues(p_values, method):
    return multipletests(p_values, method=method)[1]


def validate_SSC_cohort():
    file = '/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_mixed_pred_final.csv'
    data = pd.read_csv(file, index_col=0)
    print(data.shape)

    # relabel class 3 to class 1 (mixed_pred)
    data['mixed_pred'] = data['mixed_pred'].replace(2, 0)
    data['mixed_pred'] = data['mixed_pred'].replace(3, 2)
    print(data['mixed_pred'].value_counts())
    data.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_replication_3_classes.csv')

    _, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments_3classes(data)

    # get summary_df
    df = pd.DataFrame(columns=np.arange(data['mixed_pred'].nunique()*2), index=data.columns)
    # each column is a class and either enriched or depleted 
    for i in range(len(data['mixed_pred'].unique())):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
        # multiple hypothesis correction - FDR bh
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{i}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
        # multiple hypothesis correction
        for key, val in enriched_class_high_dict.items():
            df.loc[key, i] = val
        
        # multiple hypothesis correction - FDR bh
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(ascending=True).to_dict()
        # multiple hypothesis correction
        for key, val in enriched_class_low_dict.items():
            df.loc[key, i+data['mixed_pred'].nunique()] = val
    
    # rearrange columns
    df = df[[0,3,1,4,2,5]]
    # rename columns
    df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted']
    # move index to column
    df.reset_index(inplace=True)
    df.rename(columns={'index':'feature'}, inplace=True)
    summary_df = df.copy()
    print(summary_df)

    # NOW PLOT FOR 3 CLASSES
    features_to_exclude = pd.read_csv('../ceph/GFMM_Labeled_Data/4classes_fold_enrichments_filtered.csv').rename(columns={'Unnamed: 0': 'feature'}) # Fold enrichments + Cohen's d values filtered by significance
    # take abs value of cohen's d
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    binary_features = ['repeat_grade', 'q01_phrases', 'q02_conversation', 'q03_odd_phrase', 'q04_inappropriate_question', 'q05_pronouns_mixed', 'q06_invented_words', 'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate', 'q10_hand_tool', 'q11_interest_preoccupy', 'q12_parts_object', 'q13_interests_intensity', 'q14_senses', 'q15_odd_ways', 'q16_complicated_movements', 'q17_injured_deliberately', 'q18_objects_carry', 'q19_best_friend', 'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention', 'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively', 'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    nan_features = features_to_exclude.loc[(features_to_exclude['class0'].isna()) & 
                                            (features_to_exclude['class1'].isna()) & 
                                            (features_to_exclude['class2'].isna()) & 
                                            (features_to_exclude['class3'].isna())] # features that are null in all classes
    low_features_continuous = features_to_exclude.loc[~features_to_exclude['feature'].isin(binary_features)] # exclude binary features for continuous filter
    low_features_continuous = features_to_exclude.loc[(features_to_exclude['class0'] < 0.2) & (features_to_exclude['class1'] < 0.2) 
                                            & (features_to_exclude['class2'] < 0.2) & (features_to_exclude['class3'] < 0.2)] 
    low_features_binary = features_to_exclude.loc[features_to_exclude['feature'].isin(binary_features)]
    low_features_binary = low_features_binary.loc[(low_features_binary['class0'] < 1.5) & (low_features_binary['class1'] < 1.5)
                                            & (low_features_binary['class2'] < 1.5) & (low_features_binary['class3'] < 1.5)]
    features_to_exclude = pd.concat([nan_features, low_features_continuous, low_features_binary])
    features_to_exclude = features_to_exclude['feature'].unique()

    # GENERATE SUMMARY FIGURE for SSC rep
    # read in feature_to_category mapping
    features_to_category = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))

    # get mapping of feature to category
    df = pd.read_csv('/mnt/home/alitman/ceph/Cross_Cohort_Results/SPARK_enrich_deplete_4classes.csv', index_col=None)
    df = df.fillna('NaN')
    # drop feature category column
    df = df.drop('feature category', axis=1)
    df = df.loc[~df['feature'].isin(features_to_exclude)] # remove non-contributory features with no significant enrichments
    # annotate each feature with its category
    df['feature_category'] = df['feature'].map(feature_to_category)
    # drop features with no category
    df = df.dropna(subset=['feature_category'])
    # get new feature to category mapping
    feature_to_category = dict(zip(df['feature'], df['feature_category']))
    
    # replace 'NaN' with 1 - insignificant features
    summary_df = summary_df.replace('NaN', 1)
    summary_df = summary_df.replace(np.nan, 1)
    # annotate each feature with its category
    summary_df['feature_category'] = summary_df['feature'].map(feature_to_category)
    # drop features with no category
    summary_df = summary_df.dropna(subset=['feature_category'])
    print(summary_df['feature_category'].value_counts())
    
    flip_rows = ['q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 'q20_talk_friendly',
                'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly',
                'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention',
                'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
                'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    
    # convert to float
    summary_df['class0_enriched'] = summary_df['class0_enriched'].astype(float)
    summary_df['class0_depleted'] = summary_df['class0_depleted'].astype(float)
    summary_df['class1_enriched'] = summary_df['class1_enriched'].astype(float)
    summary_df['class1_depleted'] = summary_df['class1_depleted'].astype(float)
    summary_df['class2_enriched'] = summary_df['class2_enriched'].astype(float)
    summary_df['class2_depleted'] = summary_df['class2_depleted'].astype(float)
    #summary_df['class3_enriched'] = summary_df['class3_enriched'].astype(float)
    #summary_df['class3_depleted'] = summary_df['class3_depleted'].astype(float)
    
    # convert p value columns to binary (1 if significant, 0 if not)
    summary_df['class0_enriched'] = summary_df['class0_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class0_depleted'] = summary_df['class0_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_enriched'] = summary_df['class1_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_depleted'] = summary_df['class1_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_enriched'] = summary_df['class2_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_depleted'] = summary_df['class2_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    #summary_df['class3_enriched'] = summary_df['class3_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    #summary_df['class3_depleted'] = summary_df['class3_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    
    for row in flip_rows:
        df.loc[df['feature'] == row, ['class0_enriched', 'class0_depleted']] = df.loc[df['feature'] == row, ['class0_depleted', 'class0_enriched']].values
        df.loc[df['feature'] == row, ['class1_enriched', 'class1_depleted']] = df.loc[df['feature'] == row, ['class1_depleted', 'class1_enriched']].values
        df.loc[df['feature'] == row, ['class2_enriched', 'class2_depleted']] = df.loc[df['feature'] == row, ['class2_depleted', 'class2_enriched']].values
        #df.loc[df['feature'] == row, ['class3_enriched', 'class3_depleted']] = df.loc[df['feature'] == row, ['class3_depleted', 'class3_enriched']].values
    
    # create new dataframe with the proportions of significant features in each category
    prop_df = pd.DataFrame()
    prop_df['class0_enriched'] = summary_df.groupby(['feature_category'])['class0_enriched'].sum()/summary_df.groupby(['feature_category'])['class0_enriched'].count()
    prop_df['class0_depleted'] = summary_df.groupby(['feature_category'])['class0_depleted'].sum()/summary_df.groupby(['feature_category'])['class0_depleted'].count()
    prop_df['class1_enriched'] = summary_df.groupby(['feature_category'])['class1_enriched'].sum()/summary_df.groupby(['feature_category'])['class1_enriched'].count()
    prop_df['class1_depleted'] = summary_df.groupby(['feature_category'])['class1_depleted'].sum()/summary_df.groupby(['feature_category'])['class1_depleted'].count()
    prop_df['class2_enriched'] = summary_df.groupby(['feature_category'])['class2_enriched'].sum()/summary_df.groupby(['feature_category'])['class2_enriched'].count()
    prop_df['class2_depleted'] = summary_df.groupby(['feature_category'])['class2_depleted'].sum()/summary_df.groupby(['feature_category'])['class2_depleted'].count()
    #prop_df['class3_enriched'] = summary_df.groupby(['feature_category'])['class3_enriched'].sum()/summary_df.groupby(['feature_category'])['class3_enriched'].count()
    #prop_df['class3_depleted'] = summary_df.groupby(['feature_category'])['class3_depleted'].sum()/summary_df.groupby(['feature_category'])['class3_depleted'].count()
    
    # take the max of the absolute values of the enriched and depleted columns
    prop_df['class0_max'] = prop_df[['class0_enriched', 'class0_depleted']].abs().max(axis=1)
    prop_df['class1_max'] = prop_df[['class1_enriched', 'class1_depleted']].abs().max(axis=1)
    prop_df['class2_max'] = prop_df[['class2_enriched', 'class2_depleted']].abs().max(axis=1)
    #prop_df['class3_max'] = prop_df[['class3_enriched', 'class3_depleted']].abs().max(axis=1)
    
    # if the depleted value is larger, negate the max
    prop_df.loc[prop_df['class0_depleted'] > prop_df['class0_enriched'], 'class0_max'] = -prop_df['class0_max']
    prop_df.loc[prop_df['class1_depleted'] > prop_df['class1_enriched'], 'class1_max'] = -prop_df['class1_max']
    prop_df.loc[prop_df['class2_depleted'] > prop_df['class2_enriched'], 'class2_max'] = -prop_df['class2_max']
    #prop_df.loc[prop_df['class3_depleted'] > prop_df['class3_enriched'], 'class3_max'] = -prop_df['class3_max']

    # drop the enriched and depleted columns
    features_to_visualize = ['social/communication', 'anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'restricted/repetitive', 'developmental']
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted'], axis=1)
    prop_df.columns = ['highASD-lowDelays', 'lowASD-lowDelays', 'lowASD-highDelays']
    df = prop_df.loc[features_to_visualize]
    fig, ax = plt.subplots(1,1,figsize=(9.5,7))
    ax = sns.heatmap(df, cmap='coolwarm', annot=False, yticklabels=features_to_visualize)
    plt.ylabel('')
    ax.tick_params(labelsize=20)
    plt.title('SSC', fontsize=26)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    # rotation for xlabels
    plt.xticks(rotation=20, ha='right')
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    plt.savefig('GFMM_all_figures/SSC_3_groups_7_pheno_categories_heatmap.png', bbox_inches='tight')
    plt.close()

    # line plot for 3 classes with seaborn
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))
    fig, ax = plt.subplots(1,1,figsize=(10.5,6))
    palette = ['green','violet','blue']
    ax = sns.lineplot(data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3)    
    ax.set(xlabel="Phenotype Category", ylabel="")
    plt.xticks(rotation=45)
    plt.xticks(ha='right')
    plt.xticks(np.arange(len(features_to_visualize)), features_to_visualize)
    handles = [plt.Line2D([0], [0], color=palette[i], linewidth=3, linestyle='-', marker='o') for i in range(len(palette))]
    plt.legend(handles, ['highASD-lowDelays', 'lowASD-lowDelays', 'lowASD-highDelays'], title='Class', loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=18)
    ax.tick_params(labelsize=20)
    plt.xlabel('')
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    plt.savefig('GFMM_all_figures/SSC_3_groups_4_pheno_categories_lineplot.png', bbox_inches='tight')
    plt.close()


def developmental_milestones_validation():
    pass

def generate_ssc_data(impute=False):
    ### Preprocess, clean, and integrate SSC data
    ssc_data_dir = '/mnt/home/alitman/ceph/SSC_Phenotype_Dataset/Proband_Data'
    
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
    
    # try SCQs with both life and current scores
    scq_raw = pd.read_csv(f'{ssc_data_dir}/scq_life_recode.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq_raw.replace('yes', 1, inplace=True)
    scq_raw.replace('no', 0, inplace=True)
    scq_raw.rename(columns={'q08_hits_self_object': 'q08_hits_self_against_object', 'q09_hits_self_object': 'q09_hits_self_with_object',
                            'q28_communicatiion': 'q28_communication'}, inplace=True)
    
    # take this for the summary score
    scq = pd.read_csv(f'{ssc_data_dir}/scq_life.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq.rename(columns={'summary_score': 'final_score'}, inplace=True)
    
    # sex, verbal and nonverbal IQ:   
    core_descriptive = pd.read_csv(f'{ssc_data_dir}/ssc_core_descriptive.csv').set_index('individual', drop=True).drop(['ssc_diagnosis_full_scale_iq', 'ssc_diagnosis_nonverbal_iq', 'ssc_diagnosis_verbal_iq', 'measure', 'abc_total_score', 'adi_r_b_comm_verbal_total', 'adi_r_comm_b_non_verbal_total', 'adi_r_cpea_dx', 'adi_r_evidence_onset', 'adi_r_rrb_c_total', 'adi_r_soc_a_total', 'ados_communication_social', 'ados_css', 'ados_module', 'ados_restricted_repetitive', 'ados_social_affect', 'cbcl_2_5_externalizing_t_score', 'cbcl_2_5_internalizing_t_score', 'cbcl_6_18_externalizing_t_score', 'cbcl_6_18_internalizing_t_score', 'cpea_dx', 'diagnosis_ados', 'ethnicity', 'family_type', 'febrile_seizures', 'non_febrile_seizures', 'pregnancy_optimality_code', 'pregnancy_optimality_code_intrapartal', 'pregnancy_optimality_code_neonatal', 'pregnancy_optimality_code_prenatal', 'pregnancy_optimality_intrapartal', 'pregnancy_optimality_neonatal', 'pregnancy_optimality_prenatal', 'pregnancy_optimality_total', 'puberty_ds_progress', 'puberty_ds_total', 'race', 'rbs_r_overall_score', 'regression', 'regression_loss', 'regression_no_insert', 'srs_parent_raw_total', 'srs_parent_t_score', 'srs_teacher_raw_total', 'srs_teacher_t_score', 'ssc_diagnosis_full_scale_iq_type', 'ssc_diagnosis_nonverbal_iq_type', 'ssc_diagnosis_nvma', 'ssc_diagnosis_verbal_iq_type', 'ssc_diagnosis_vma', 'vineland_ii_composite_standard_score'], axis=1)
    core_descriptive.replace('male', 1, inplace=True)
    core_descriptive.replace('female', 0, inplace=True)
    core_descriptive.rename(columns={'age_at_ados': 'age_at_eval_years'}, inplace=True)
    # convert age from months to years
    core_descriptive['age_at_eval_years'] = core_descriptive['age_at_eval_years']/12
    
    # read in processed Background hx
    bh = pd.read_csv('/mnt/home/alitman/ceph/SSC_Phenotype_Dataset/SSC_background_hx_clean.csv').set_index('individual', drop=True)
    bh = bh.drop(['age_gave_up_bottle', 'age_started_solid_foods'], axis=1) # not in SPARK model
    
    # merge data on subject ID
    finaldf = pd.concat([core_descriptive, bh, scq_raw, scq, rbsr_raw, rbsr_scores, cbcl], axis=1, join='inner') # bh
    print(finaldf.isnull().sum().sum())
    
    finaldf = finaldf.loc[:, finaldf.isna().sum()/finaldf.shape[0] < 0.1] # drop columns with more than 10% missing values
    print(list(finaldf.columns))

    if impute:
        imputer = KNNImputer(n_neighbors=7) # tune this parameter
        finaldf = pd.DataFrame(imputer.fit_transform(finaldf), columns=finaldf.columns, index=finaldf.index)
    else:
        finaldf = finaldf.dropna(axis=0)
        finaldf = finaldf.astype('float32')
    print(finaldf.shape)
    
    return finaldf


def cross_cohort_classifier(ncomp=4):
    ssc_data = generate_ssc_data(bootstrap=False, impute=False)
    print(ssc_data.shape)

    # load spark data
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)
    flip_rows = ['q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 'q20_talk_friendly',
            'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly',
            'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention',
            'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
            'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    flip_rows = [x for x in flip_rows if x in datadf.columns]
    # reverse those columns (if 1 -> 0, if 0 -> 1)
    datadf[flip_rows] = datadf[flip_rows].apply(lambda x: 1-x)
    datadf = datadf.round()

    # subset columns to only those in ssc_data
    cols_to_match = list(set(ssc_data.columns).intersection(set(datadf.columns)))
    ssc_data = ssc_data[cols_to_match]
    datadf = datadf[cols_to_match]
    print(cols_to_match)
    print(ssc_data.shape)
    print(datadf.shape)

    # concat ssc and spark data
    datadf = pd.concat([datadf, ssc_data], axis=0)
    print(datadf.shape)
    print(list(datadf.columns))

    # 1. train model on ssc+spark data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    ) 

    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200) #n_init=200) , random_state=42

    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    mixed_data.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_SPARK_mixed_pred_final.csv')
    
    mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SSC_SPARK_mixed_pred_final.csv', index_col=0)
    '''
    # now predict labels for ssc_data
    X_ssc = ssc_data.drop(['age_at_eval_years', 'sex'], axis=1)
    # make sure the columns are in the same order
    X_ssc = X_ssc[X.columns]
    # predict labels
    mixed_data_ssc = X_ssc.copy()
    mixed_data_ssc['mixed_pred'] = model.predict(X_ssc)
    labels_ssc = mixed_data_ssc['mixed_pred']
    '''

    # plot distribution of labels
    pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, summary_df = get_feature_enrichments(mixed_data, summarize=True)
    features_to_exclude = pd.read_csv('../ceph/GFMM_Labeled_Data/4classes_fold_enrichments_filtered.csv').rename(columns={'Unnamed: 0': 'feature'}) # Fold enrichments + Cohen's d values filtered by significance
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

    # GENERATE SUMMARY FIGURE for SSC rep
    # read in feature_to_category mapping
    features_to_category = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))

    summary_df = summary_df.fillna('NaN')
    summary_df = summary_df.replace(np.nan, 1)
    summary_df = summary_df.replace('NaN', 1)
    summary_df = summary_df.loc[~summary_df['feature'].isin(features_to_exclude)] # remove non-contributory features with no significant enrichments
    summary_df['feature_category'] = summary_df['feature'].map(feature_to_category)
    summary_df = summary_df.dropna(subset=['feature_category'])
    feature_to_category = dict(zip(summary_df['feature'], summary_df['feature_category']))
    print(summary_df['feature_category'].value_counts())
    
    # convert to float
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
    '''
    prop_df['class0_max'] = prop_df[['class0_enriched', 'class0_depleted']].abs().max(axis=1)
    prop_df['class1_max'] = prop_df[['class1_enriched', 'class1_depleted']].abs().max(axis=1)
    prop_df['class2_max'] = prop_df[['class2_enriched', 'class2_depleted']].abs().max(axis=1)
    prop_df['class3_max'] = prop_df[['class3_enriched', 'class3_depleted']].abs().max(axis=1)
    
    # if the depleted value is larger, negate the max
    prop_df.loc[prop_df['class0_depleted'] > prop_df['class0_enriched'], 'class0_max'] = -prop_df['class0_max']
    prop_df.loc[prop_df['class1_depleted'] > prop_df['class1_enriched'], 'class1_max'] = -prop_df['class1_max']
    prop_df.loc[prop_df['class2_depleted'] > prop_df['class2_enriched'], 'class2_max'] = -prop_df['class2_max']
    prop_df.loc[prop_df['class3_depleted'] > prop_df['class3_enriched'], 'class3_max'] = -prop_df['class3_max']
    '''
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
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'social/communication', 'self-injury', 'restricted/repetitive', 'developmental']  #,
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay']
    print(prop_df)
    df = prop_df.loc[features_to_visualize]
    fig, ax = plt.subplots(1,1,figsize=(9.5,7))
    ax = sns.heatmap(df, cmap='coolwarm', annot=False, yticklabels=features_to_visualize)
    plt.ylabel('')
    ax.tick_params(labelsize=20)
    plt.title('SSC', fontsize=26)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    plt.savefig('GFMM_all_figures/SSC_CROSS_COHORT_7_pheno_categories_heatmap.png', bbox_inches='tight')
    plt.close()

    # HORIZONTAL LINE PLOT
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    palette = ['violet','red','limegreen','blue']
    ax = sns.lineplot(data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3)    
    ax.set(xlabel="Phenotype Category", ylabel="")
    plt.xticks(rotation=45)
    ax.set_xticklabels(features_to_visualize)
    plt.title('SSC', fontsize=26)
    plt.xticks(ha='right')
    plt.xticks(np.arange(len(features_to_visualize)))
    plt.ylim([-1.1,1.1])
    for line in ax.lines:
        line.set_linewidth(5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.get_legend().remove()
    ax.tick_params(labelsize=20)
    plt.xlabel('')
    ax.set_ylabel('Proportion+direction of sig. features', fontsize=18)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    plt.savefig('GFMM_all_figures/SSC_CROSS_COHORT_4_pheno_categories_lineplot.png', bbox_inches='tight', dpi=300)
    plt.close()


def bootstrap_samples(data_matrix, num_samples):
    """
    Generate bootstrap samples from a data matrix and return as a single concatenated DataFrame.

    Parameters:
    - data_matrix (numpy array): Input data matrix where each row is a sample and each column is a feature.
    - num_samples (int): Number of bootstrap samples to generate.

    Returns:
    - bootstrap_samples_df (pandas DataFrame): Concatenated DataFrame containing all bootstrap samples.
    """
    num_samples_orig = data_matrix.shape[0]
    bootstrap_samples_dfs = []

    # reset index of data_matrix
    data_matrix = data_matrix.reset_index(drop=True)

    for _ in range(num_samples):
        # Randomly sample with replacement from the original data indices
        sample_indices = np.random.choice(num_samples_orig, num_samples_orig, replace=True)

        # Use the sampled indices to create a bootstrap sample as a DataFrame
        bootstrap_sample_df = pd.DataFrame(data_matrix[sample_indices, :], columns=data_matrix.columns)
        bootstrap_samples_dfs.append(bootstrap_sample_df)

    # Concatenate all bootstrap sample DataFrames into a single DataFrame
    bootstrap_samples_df = pd.concat(bootstrap_samples_dfs, ignore_index=True)
    return bootstrap_samples_df

    
if __name__ == '__main__':
    #posterior_prob_validation(); exit()
    #cross_cohort_classifier(ncomp=4); exit()
    get_SSC_data(ncomp=4); exit()
    validate_SSC_cohort()