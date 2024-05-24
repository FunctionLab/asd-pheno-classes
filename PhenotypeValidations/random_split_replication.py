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
from scipy.stats import hypergeom
import pickle
import plotly.express as px
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from latent_class_analysis import run_lca_no_bms

def get_class_memberships(class0_df, class1_df, class2_df, class3_df, output_name):
    ## extract values for classes for class membership dictionaries
    class0 = list(class0_df.index)
    class1 = list(class1_df.index)
    class2 = list(class2_df.index)
    class3 = list(class3_df.index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    class_membership = dict()
    class_membership['class0'] = class0
    class_membership['class1'] = class1
    class_membership['class2'] = class2
    class_membership['class3'] = class3
    
    return class_membership

if __name__ == "__main__":

    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland

    X = datadf.drop(['asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)

    ### REPLICATION: RANDOMLY SPLIT SAMPLE
    X_sample1, X_sample2 = train_test_split(X, test_size=0.5)

    # OPTIONAL: SAVE TO FILE (incase you want to replicate these)
    #X_sample1.to_csv('replication_split/X_sample1.csv')
    #X_sample2.to_csv('replication_split/X_sample2.csv')

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
                      'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
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
                          'iv_ritualistic_behavior_score', 'v_sameness_behavior_score', 'vi_restricted_behavior_score']

    # Run a mixed model for each split separately:
    mixed_data1, mixed_descriptor1 = get_mixed_descriptor(
        dataframe=X_sample1,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    mixed_data2, mixed_descriptor2 = get_mixed_descriptor(
        dataframe=X_sample2,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # LCA using StepMix
    model1 = StepMix(n_components=4, measurement=mixed_descriptor1, verbose=1, random_state=123)
    model2 = StepMix(n_components=4, measurement=mixed_descriptor2, verbose=1, random_state=123)

    # fit models to data
    model1.fit(mixed_data1)
    model2.fit(mixed_data2)

    mixed_data1['mixed_pred'] = model1.predict(mixed_data1)
    mixed_data2['mixed_pred'] = model2.predict(mixed_data2)

    ### STATISTICAL ENRICHMENT TESTS
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()

    ## extract dataframes for each class
    class0_split1 = mixed_data1[mixed_data1['mixed_pred'] == 0]
    class1_split1 = mixed_data1[mixed_data1['mixed_pred'] == 1]
    class2_split1 = mixed_data1[mixed_data1['mixed_pred'] == 2]
    class3_split1 = mixed_data1[mixed_data1['mixed_pred'] == 3]

    class0_split2 = mixed_data2[mixed_data2['mixed_pred'] == 0]
    class1_split2 = mixed_data2[mixed_data2['mixed_pred'] == 1]
    class2_split2 = mixed_data2[mixed_data2['mixed_pred'] == 2]
    class3_split2 = mixed_data2[mixed_data2['mixed_pred'] == 3]

    # save class membership dictionaries
    split1_4classes = get_class_memberships(class0_split1, class1_split1, class2_split1, class3_split1, 'split1_4classes_membership')
    split2_4classes = get_class_memberships(class0_split2, class1_split2, class2_split2, class3_split2, 'split2_4classes_membership')

    # generate jaccard index heatmaps comparing full LCA membership to split membership
    # define Jaccard Similarity function
    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union
    
    mixed_data = run_lca_no_bms(ncomp=4)
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]
    # get class membership
    lca_4classes = get_class_memberships(class0, class1, class2, class3, 'lca_4classes_membership')

    first_split_comp = pd.DataFrame(columns=split1_4classes.keys(), index=lca_4classes.keys(), dtype=float)
    second_split_comp = pd.DataFrame(columns=split2_4classes.keys(), index=lca_4classes.keys(), dtype=float)

    for class_key in lca_4classes:
        for class_key2 in split1_4classes:
            first_split_comp.loc[class_key, class_key2] = 2*jaccard(lca_4classes[class_key], split1_4classes[class_key2]) # multiply by 2 to get total prop out of 100

    for class_key in lca_4classes:
        for class_key2 in split2_4classes:
            second_split_comp.loc[class_key, class_key2] = 2*jaccard(lca_4classes[class_key], split2_4classes[class_key2])

    # plot jaccard index heatmaps for each split
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    sns.heatmap(first_split_comp, cmap="BuPu", annot=True, annot_kws={"size": 14})
    plt.ylabel('Full SPARK dataset', fontsize=18)
    # increase size of x-axis and y-axis labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('SPARK Subsample', fontsize=18)
    plt.savefig('lca_plots/split1_4classes_jaccard_heatmap.png')

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 7))
    sns.heatmap(second_split_comp, cmap="BuPu", annot=True, annot_kws={"size": 14})
    plt.ylabel('Full SPARK dataset', fontsize=18)
    plt.xlabel('SPARK Subsample', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.title('Class membership intersection between full set and subset', fontsize=18)
    plt.savefig('lca_plots/split2_4classes_jaccard_heatmap.png')