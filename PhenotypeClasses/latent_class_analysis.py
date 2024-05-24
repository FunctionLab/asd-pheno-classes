"""Code to analyze the SPARK phenotype data with a GFMM clustering model."""

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
import pickle as rick
import plotly.express as px
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go
from numpy.linalg import norm
from scipy.stats import hypergeom, pearsonr
from sklearn.decomposition import PCA
from scipy.stats import binomtest


def run_main_LCA_model(ncomp):
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI
    # y = datadf['asd'] # try outcome as ASD
    age = datadf['age_at_eval_years']

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    
    #scaler = MinMaxScaler()
    #features = X.columns
    #X = pd.DataFrame(scaler.fit_transform(X), columns=features)
    binary_columns, categorical_columns, continuous_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )
    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=42)
                    #structural_params=covariate_params)
    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data['age'] = age
    
    # get feature enrichments - optional
    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments(mixed_data)
    #plot_line_polar(mixed_data, 'GFMM_all_figures/LCA_line_polar_plot_noadi_noid')
    
    return mixed_data

def run_lca_no_bms(ncomp=4, summarize=False):
    """
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5391_unimputed_cohort.txt', index_col=0)  # 5391 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5570_unimputed_cohort.txt', sep='\t', index_col=0)  # 5391 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  # 5392 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5341_unimputed_cohort.txt', sep='\t', index_col=0)  # 5570 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5280_unimputed_cohort.txt', sep='\t', index_col=0)  # 5279 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_7032_imputed_cohort.txt', sep='\t', index_col=0)  # 7032 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms.txt', sep='\t', index_col=0)  # ~4700 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_imputed.txt', sep='\t', index_col=0)  # 6406 individuals, RBSR+CBCL, no vineland, no ADI, no BMS, imputed
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_imputed_genetic_diagnosis.txt', sep='\t', index_col=0)  # 5837 individuals, CBCL scores only
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_new.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI, no BMS, imputed, genetic diagnosis
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_impute_validation.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI, no BMS, imputed, genetic diagnosis
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_cbcl_scores_5760.txt', sep='\t', index_col=0)  # 5760 probands, BH+SCQ+RBSR+CBCL, only CBCL scores.
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_5721_imputed.txt', sep='\t', index_col=0)  # 5721 probands, BH+SCQ+RBSR+CBCL, no vineland, no ADI, no BMS, imputed
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_6395_imputed.txt', sep='\t', index_col=0)  # 6395 probands, only CBCL scores, imputed
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_5727_imputed.txt', sep='\t', index_col=0)  # 5727 probands, only CBCL scores, imputed
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms_5714_cbcl_scores.txt', sep='\t', index_col=0)  # 5714 probands, only CBCL scores, no BMS
    
    flip_rows = ['q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 'q20_talk_friendly',
                'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 'q24_nod_head', 'q25_shake_head', 'q26_look_directly',
                'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 'q32_help_attention',
                'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
                'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    flip_rows = [x for x in flip_rows if x in datadf.columns]
    # reverse those columns (if 1 -> 0, if 0 -> 1)
    datadf[flip_rows] = datadf[flip_rows].apply(lambda x: 1-x)
    print(datadf.loc[:,flip_rows].head())

    # round all features to nearest integer
    datadf = datadf.round()

    age = datadf['age_at_eval_years']

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'age_at_eval_years'], # , 'not_able', 'left', 'right', 'ambi'],
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
                    n_steps=1, n_init=200)#, random_state=42)
                    #structural_params=covariate_params)

    model.fit(mixed_data, Z_p)
    posterior_probs = model.predict_proba(mixed_data)
    # take max posterior probability for each sample
    posterior_probs = np.max(posterior_probs, axis=1)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    mixed_data['age'] = age

    # get derived_cog_impair in dataframe post prediction
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
    prediq = pd.read_csv(f'{BASE_PHENO_DIR}/predicted_iq_experimental_2022-12-12.csv') ## predicted iq
    prediq = prediq.set_index('subject_sp_id',drop=True).drop(['family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'asd', 'ml_predicted_cog_impair'], axis=1)
    prediq = prediq[~prediq.index.duplicated(keep=False)]
    # combine with mixed_data
    #mixed_data = mixed_data.merge(prediq[['derived_cog_impair']], left_index=True, right_index=True)
    print(mixed_data.shape)
    mixed_data.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_recode4_ninit_cohort_GFMM_labeled.csv')
    """
    mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0)

    #mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_recode_ninit_cohort_GFMM_labeled.csv', index_col=0)
    #mixed_data = mixed_data.apply(pd.to_numeric, errors='ignore')
    # subset to ssc features
    features_subset = ['walked_age_mos', 'final_score', 'iii_compulsive_behavior_score', 'ii_self_injurious_score', 'i_stereotyped_behavior_score', 'iv_ritualistic_behavior_score', 'vi_restricted_behavior_score', 'v_sameness_behavior_score', 'dsm5_attention_deficit_hyperactivity_t_score', 'dsm5_depressive_problems_t_score', 'aggressive_behavior_t_score', 'dsm5_anxiety_problems_t_score', 'anxious_depressed_t_score', 'attention_problems_t_score', 'dsm5_conduct_problems_t_score', 'externalizing_problems_t_score', 'internalizing_problems_t_score', 'dsm5_oppositional_defiant_t_score', 'rule_breaking_behavior_t_score', 'social_problems_t_score', 'somatic_complaints_t_score', 'dsm5_somatic_problems_t_score', 'thought_problems_t_score', 'total_problems_t_score', 'withdrawn_depressed_t_score', 'q03_odd_phrase', 'q05_pronouns_mixed', 'q06_invented_words', 'q07_same_over', 'q08_particular_way', 'q09_expressions_appropriate', 'q13_interests_intensity', 'q17_injured_deliberately', 'q20_talk_friendly', 'q23_gestures_wanted', 'q26_look_directly', 'q01_whole_body', 'q02_head', 'q03_hand_finger', 'q04_locomotion', 'q05_object_usage', 'q06_sensory', 'q07_hits_self_body', 'q10_bites_self', 'q11_pulls', 'q12_rubs', 'q13_inserts_finger', 'q14_skin_picking', 'q15_arranging', 'q16_complete', 'q17_washing', 'q18_checking', 'q19_counting', 'q20_hoarding', 'q21_repeating', 'q22_touch_tap', 'q23_eating', 'q24_sleep', 'q25_self_care', 'q26_travel', 'q27_play', 'q29_things_same_place', 'q30_objects', 'q31_becomes_upset', 'q32_insists_walking', 'q33_insists_sitting', 'q34_dislikes_changes', 'q35_insists_door', 'q36_likes_piece_music', 'q37_resists_change', 'q38_insists_routine', 'q39_insists_time', 'q40_fascination_subject', 'q41_strongly_attached', 'q42_preoccupation', 'q43_fascination_movement', 'mixed_pred']
    features_subset = [x for x in features_subset if x in mixed_data.columns]
    #mixed_data = mixed_data[features_subset]

    # get sex
    sex_data = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)
    mixed_data = pd.merge(mixed_data, sex_data[['sex']], left_index=True, right_index=True)

    # plot age and sex distributions for each class
    #get_age_distributions_for_classes(mixed_data)
    
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
    prediq = pd.read_csv(f'{BASE_PHENO_DIR}/predicted_iq_experimental_2022-12-12.csv') ## predicted iq
    prediq = prediq.set_index('subject_sp_id',drop=True).drop(['family_sf_id', 'biomother_sp_id', 'biofather_sp_id', 'sex', 'asd', 'ml_predicted_cog_impair'], axis=1)
    prediq = prediq[~prediq.index.duplicated(keep=False)]
    # combine with mixed_data
    mixed_data = mixed_data.merge(prediq[['derived_cog_impair']], left_index=True, right_index=True)
    print(mixed_data.groupby('mixed_pred')['derived_cog_impair'].mean())

    # get feature enrichments - optional
    if summarize:
        classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)
    else:
        classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments(mixed_data)
    #classification_df.to_csv('/mnt/home/alitman/ceph/SPARK_replication/LCA_feature_enrichments_nobms.csv')
    plot_line_polar(mixed_data, f'GFMM_all_figures/GFMM_line_polar_plot_5391_{ncomp}comp')
    
    print(mixed_data['mixed_pred'].value_counts())
    # pie chart of class proportions
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax = mixed_data['mixed_pred'].value_counts().plot.pie(autopct='%1.0f%%', startangle=90, colors=['violet','limegreen','blue','red'], labels=None)
    # increase transparency of pie chart
    for patch in ax.patches:
        patch.set_alpha(0.75)
    plt.title('SPARK class proportions', fontsize=24)
    # increase font size of pie chart
    plt.rcParams.update({'font.size': 24})
    # increase font size of percentages
    plt.setp(ax.texts, size=22)
    plt.ylabel('')
    plt.savefig(f'GFMM_all_figures/GFMM_pie_chart_5392_{ncomp}comp.png', bbox_inches='tight')
    plt.close()
    return df_enriched_depleted, fold_enrichments
    
    # scramble each feature in the unlabeled data, retrain new model, and get posterior probabilities for each class
    copydf = datadf.copy()
    scrambled_data = copydf.apply(scramble_column)
    age = scrambled_data['age_at_eval_years']

    # get covariate data
    Z_p = scrambled_data[['sex', 'age_at_eval_years']]
    X = scrambled_data.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    scrambled, scrambled_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )
    model_scram = StepMix(n_components=ncomp, measurement=scrambled_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=42)
    model_scram.fit(scrambled, Z_p)
    posterior_probs_scram = model_scram.predict_proba(scrambled)
    # take max posterior probability for each sample
    posterior_probs_scram = np.max(posterior_probs_scram, axis=1)  
    # OPTIONAL: plot posterior probabilities for each class for scrambled and unscrambled data
    plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, ncomp)

    return mixed_data


def plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, ncomp):
    # for each class, plot posterior probabilities for scrambled and unscrambled data
    posterior_probs_df = pd.DataFrame()
    posterior_probs_df['posterior_prob'] = posterior_probs
    posterior_probs_df['scrambled'] = False
    posterior_probs_df['class'] = labels.values
    posterior_probs_df_scram = pd.DataFrame()
    posterior_probs_df_scram['posterior_prob'] = posterior_probs_scram
    posterior_probs_df_scram['scrambled'] = True
    posterior_probs_df_scram['class'] = labels.values
    posterior_probs_df = pd.concat([posterior_probs_df, posterior_probs_df_scram])
    # plot
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=posterior_probs_df, x='class', y='posterior_prob', hue='scrambled', palette='Dark2')
    plt.savefig(f'GFMM_all_figures/GFMM_posterior_probs_boxplot_{ncomp}comp.png')
    plt.clf()
    print("done.")


def scramble_column(column):
    return np.random.permutation(column)

def generate_summary_table(df_enriched_depleted, fold_enrichments):
    '''GENERATE TABLE SUMMARIZING THE RESULTS OF THE LCA ANALYSIS WITH 8 PHENOTYPE CLASSES.'''
    features_to_exclude = fold_enrichments.copy() # Fold enrichments + Cohen's d values filtered by significance
    #features_to_exclude = pd.read_csv('../ceph/GFMM_Labeled_Data/4classes_fold_enrichments_filtered.csv').rename(columns={'Unnamed: 0': 'feature'}) # Fold enrichments + Cohen's d values filtered by significance
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

    #df = pd.read_csv('../ceph/GFMM_Labeled_Data/SPARK_4classes_enriched_depleted.csv', index_col=None)
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
    #df.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_4classes_enriched_depleted_filtered.csv', index=False)
    
    # replace 'NaN' with 1 - insignificant features
    df = df.replace('NaN', 1)
    print(df.shape)
    print(df['feature_category'].value_counts())
    
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
    '''
    print(prop_df)
    
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
    # for each variable, plot class0_depleted, class0_enriched, class1_depleted, class1_enriched, class2_depleted, class2_enriched, class3_depleted, class3_enriched in a bar plot
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    # drop 'somatic' and 'other problems' from feature category
    df = prop_df.drop(['class0_max', 'class1_max', 'class2_max', 'class3_max'], axis=1)
    df = df[df.index != 'somatic']
    df = df[df.index != 'other problems']
    df = df[df.index != 'thought problems']

    proportions = pd.DataFrame(index=df.index)
    for i in range(4):
        proportions[f'class{i}_enriched'] = df[f'class{i}_enriched']
        proportions[f'class{i}_depleted'] = df[f'class{i}_depleted']
    
    proportions = proportions.reset_index()
    # set the following order for feature category: 
    #'anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'
    proportions = proportions.set_index('feature_category').reindex(['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental']).reset_index()
    proportions_melted = pd.melt(proportions, id_vars=['feature_category'], value_vars=[f'class{i}_enriched' for i in range(4)] + [f'class{i}_depleted' for i in range(4)],
                                var_name='class', value_name='proportion')

    # Extract class and type information
    proportions_melted['type'] = proportions_melted['class'].apply(lambda x: x.split('_')[1])
    proportions_melted['class'] = proportions_melted['class'].apply(lambda x: x.split('_')[0])

    fig, ax = plt.subplots(figsize=(12, 5))

    # Define colors for enriched and depleted
    colors = {'enriched': '#1f77b4', 'depleted': '#ff7f0e'}

    # Get unique feature categories and classes
    feature_categories = proportions['feature_category'].unique()
    classes = ['class0', 'class1', 'class2', 'class3']

    # Define bar width and positions
    bar_width = 0.1
    n_classes = len(classes)
    spacing = 0.2
    group_width = n_classes * bar_width + spacing

    # Calculate positions for each group
    bar_positions = np.arange(len(feature_categories)) * group_width

    # Custom colors for each class
    class_colors = {
        'class0': ['pink', 'violet'],
        'class1': ['lightcoral', 'red'],
        'class2': ['palegreen', 'limegreen'],
        'class3': ['lightblue', 'blue']
    }

    # Plot each class's enriched and depleted proportions
    for idx, cls in enumerate(classes):
        enriched = proportions_melted[(proportions_melted['class'] == cls) & (proportions_melted['type'] == 'enriched')]
        depleted = proportions_melted[(proportions_melted['class'] == cls) & (proportions_melted['type'] == 'depleted')]
        ax.bar(bar_positions + idx * bar_width, depleted['proportion'], width=bar_width, label=f'{cls} depleted', linewidth=0.5, edgecolor='black', color=class_colors[cls][0])
        ax.bar(bar_positions + idx * bar_width, enriched['proportion'], width=bar_width, label=f'{cls} enriched', linewidth=0.5, edgecolor='black', color=class_colors[cls][1])

    # Set the x-ticks and labels
    ax.set_xticks(bar_positions + bar_width * 1.5)
    ax.set_xticklabels(['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'limited social/communication', 'restricted/repetitive', 'developmental delay'], rotation=30, ha='right')
    # make figure borders thicker and black
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        #ax.spines[axis].set_color('black')
    # line at y = 0
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    # Add labels and title
    plt.xlabel('')
    plt.ylabel('')
    #plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=16)    

    plt.tight_layout()
    plt.savefig('GFMM_all_figures/GFMM_summary_figure.png')
    plt.show()
    exit()


    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay']
    plot_df = prop_df.T
    plot_df['cluster'] = np.arange(4)
    print(plot_df)
    
    # plot line polar plot
    fig, ax = plt.subplots(1,1,figsize=(9.5,7))
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental', 'cluster'] 
    plot_df = plot_df[features_to_visualize]
    polar = plot_df.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    #polar.to_csv('GFMM_all_figures/SPARK_summary_figure_data.csv'); exit()
    colors = ['violet','red','limegreen','blue']
    fig = px.line_polar(polar, r="value", theta="feature_category", color="cluster", 
                        color_discrete_sequence=colors, line_close=True, height=800, width=1400) # template="plotly_dark",
    fig.update_layout(
        font_size=36
    )
    fig.update_traces(line=dict(width=5))
    fig.update_polars(radialaxis=dict(visible=True, linewidth=2, tickwidth=2, ticklen=10))
    fig.update_polars(angularaxis=dict(visible=True, linewidth=2, tickwidth=2, ticklen=10))
    fig.update_polars(angularaxis_showticklabels=True)
    fig.update_layout(polar_radialaxis_range=[-1.2, 1.2])
    fig.update_polars(radialaxis=dict(gridcolor='black', gridwidth=0.5))
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("cluster=", "Class ")))
    fig.update_layout(legend_title_text='Class')
    fig.write_image(f"GFMM_all_figures/GFMM_summary_table_polar_plot_5392.png", width=1500, height=800)
    
    # Horizontally plot the phenotype categories, have one line per class, y-axis = proportion of significant features
    # make a line plot
    features_to_visualize = features_to_visualize[:-1]
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))

    #prop_df.to_csv('GFMM_all_figures/SPARK_phenotypes_summary_table.csv')
    
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    palette = ['violet','red','limegreen','blue']
    #prop_df.drop(['lowASD/highDelay'], axis=1, inplace=True) # for build up
    ax = sns.lineplot(data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3)    
    ax.set(xlabel="Phenotype Category", ylabel="")
    plt.xticks(ha='right', rotation=30, fontsize=16)
    plt.xticks(np.arange(len(features_to_visualize)), ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'limited social/communication', 'restricted/repetitive', 'developmental delay'])
    plt.ylim([-1.1,1.1])
    # make lines thicker
    for line in ax.lines:
        line.set_linewidth(5)

    # add colors to legend and change font size
    # make legend handles with palette
    # make legend with no title
    # make figure borders thicker and black
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')

    #handles = [plt.Line2D([0], [0], color=palette[i], linewidth=3, linestyle='-', marker='o') for i in range(4)]
    #plt.legend(handles, ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay'], title='Class', loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=18)
    # remove legend
    ax.get_legend().remove()
    ax.tick_params(labelsize=20)
    plt.xlabel('')
    ax.set_ylabel('Proportion+direction of sig. features', fontsize=18)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    plt.savefig('GFMM_all_figures/GFMM_4_pheno_categories_lineplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Heatmap of the max values
    fig, ax = plt.subplots(1,1,figsize=(9.5,7))
    ax = sns.heatmap(prop_df, annot=False, cmap='coolwarm', yticklabels=features_to_visualize)
    #ax.hlines([0,1,2,3], *ax.get_xlim(), linewidth=2)
    plt.ylabel('')
    ax.tick_params(labelsize=20)
    plt.title('SPARK', fontsize=26)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    plt.savefig('GFMM_all_figures/GFMM_7_pheno_categories_heatmap.png', bbox_inches='tight')
    plt.close()
    

def run_lca_no_cbcl(main_data, drop_bh=False):
    '''run LCA on data without CBCL features to see if groupings are maintained'''
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_w_finalscore.txt', sep='\t', index_col=0)
    
    X = datadf.drop(['repeat_grade', 'not_able', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi', 'close_friends',
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
                           'q110_wishes_to_be_opp_sex', 'q111_withdrawn', 'q112_worries'], axis=1)  # drop asd label and convert to np array
    
    if drop_bh:
        X = X.drop(['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
                          'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos',
                          'combined_phrases_age_mos', 'bladder_trained_age_mos', 'bowel_trained_age_mos'], axis=1)
        continuous_columns = ['i_stereotyped_behavior_score', 'ii_self_injurious_score', 'iii_compulsive_behavior_score',
                          'iv_ritualistic_behavior_score', 'v_sameness_behavior_score', 'vi_restricted_behavior_score', 'final_score']
    else:
        continuous_columns = ['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
                          'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos',
                          'combined_phrases_age_mos', 'bladder_trained_age_mos', 'bowel_trained_age_mos',
                          'i_stereotyped_behavior_score', 'ii_self_injurious_score', 'iii_compulsive_behavior_score',
                          'iv_ritualistic_behavior_score', 'v_sameness_behavior_score', 'vi_restricted_behavior_score', 'final_score', 'adi_national_rank_percentile', 'adi_state_rank_decile',
                          'anxious_depressed_t_score', 'withdrawn_depressed_t_score', 'somatic_complaints_t_score',
                          'social_problems_t_score', 'thought_problems_t_score', 'attention_problems_t_score',
                          'rule_breaking_behavior_t_score', 'aggressive_behavior_t_score',
                          'internalizing_problems_t_score', 'externalizing_problems_t_score', 'total_problems_t_score',
                          'obsessive_compulsive_problems_t_score', 'sluggish_cognitive_tempo_t_score',
                          'stress_problems_t_score', 'dsm5_conduct_problems_t_score', 'dsm5_somatic_problems_t_score',
                          'dsm5_oppositional_defiant_t_score', 'dsm5_attention_deficit_hyperactivity_t_score',
                          'dsm5_anxiety_problems_t_score', 'dsm5_depressive_problems_t_score']

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
                           'q41_strongly_attached', 'q42_preoccupation', 'q43_fascination_movement']
    
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # LCA using StepMix
    model = StepMix(n_components=4, measurement=mixed_descriptor, verbose=1, random_state=123)

    model.fit(mixed_data)
    mixed_data['mixed_pred'] = model.predict(mixed_data)

    # get heatmap between main model and no_cbcl_model
    ## extract values for classes for class membership dictionaries
    class0subset = list(mixed_data[mixed_data['mixed_pred'] == 0].index)
    class1subset = list(mixed_data[mixed_data['mixed_pred'] == 1].index)
    class2subset = list(mixed_data[mixed_data['mixed_pred'] == 2].index)
    class3subset = list(mixed_data[mixed_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    subset_class_membership = dict()
    subset_class_membership[0] = class0subset
    subset_class_membership[1] = class1subset
    subset_class_membership[2] = class2subset
    subset_class_membership[3] = class3subset

    class0 = list(main_data[main_data['mixed_pred'] == 0].index)
    class1 = list(main_data[main_data['mixed_pred'] == 1].index)
    class2 = list(main_data[main_data['mixed_pred'] == 2].index)
    class3 = list(main_data[main_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    class_membership = dict()
    class_membership[0] = class0
    class_membership[1] = class1
    class_membership[2] = class2
    class_membership[3] = class3

    #return subset_class_membership

    # get Jaccard Index heatmap
    jaccard_index = np.zeros((4,4))
    for i, list1 in subset_class_membership.items():
        for j, list2 in class_membership.items():
            jac = jaccard(list1, list2)
            inter = len(list(set(list1) & set(list2)))
            # assign j,i element of jaccard_index to jac
            #jaccard_index[j,i] = jac
            jaccard_index[j,i] = inter/len(list2)
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax = sns.heatmap(jaccard_index, cmap='BuPu', annot=True)
    ax.set(xlabel="No CBCL Class Membership", ylabel="w/ CBCL Class Membership")
    plt.savefig('GFMM_all_figures/LCA_no_cbcl_jaccard_heatmap.png', bbox_inches='tight')
    plt.close()

    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments(mixed_data)
    #classification_df.to_csv('/mnt/home/alitman/ceph/SPARK_replication/SPARK_4classes_no_cbcl_classification.csv')
    #plot_line_polar(mixed_data, 'GFMM_all_figures/LCA_no_cbcl_line_polar_plot'); exit()

    # compare direction of enrichment for each feature between main model and WGS model
    ### COMPUTE CORRELATIONS BETWEEN ENRICHMENT PATTERNS
    spark_pval_classification = pd.read_csv('/mnt/home/alitman/ceph/SPARK_replication/SPARK_4classes_pval_replication_classification.csv', index_col=0)
    #nocbcl_classification = pd.read_csv('/mnt/home/alitman/ceph/LCA_4classes_no_cbcl_training_data.csv', index_col=0)
    #print(nocbcl_classification)
    # intersect index to find common features
    idx_intersect = np.intersect1d(classification_df.index, spark_pval_classification.index)
    print(idx_intersect)

    spark_pval_classification = spark_pval_classification.loc[idx_intersect]
    classification_df = classification_df.loc[idx_intersect]
    
    corr_array = pd.DataFrame(columns=np.arange(3), index=np.arange(4), dtype=float) # change to 3 if using 3 classes
    for i in range(4):
        for j in range(3): # change to 3 if using 3 classes
            if pearsonr(list(spark_pval_classification.iloc[:,i]), list(classification_df.iloc[:,j])).statistic >= 0:
                corr_array.loc[i, j] = pearsonr(list(spark_pval_classification.iloc[:,i]), list(classification_df.iloc[:,j])).statistic
            else:
                corr_array.loc[i, j] = 0 # cutoff at 0 so we don't see - correlations

    print(corr_array)

    # heatmap for correlation of enrichment patterns
    #print(pearsonr(spark_pval_classification.iloc[:,0], ssc_pval_classification.iloc[:,0]).statistic)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    sns.heatmap(corr_array, cmap="PuRd", annot=True) # BuPu, YlGnBu
    plt.ylabel('SPARK classes', fontsize=18)
    plt.xlabel('WGS classes', fontsize=18)
    plt.title('Correlation of enrichment patterns (n_classes=4)', fontsize=18)
    plt.savefig('GFMM_all_figures/spark_4classes_correlation_heatmap.png')


def get_age_distributions_for_classes(mixed_data):
    # visualize age distributions for classes
    colors = ['violet','red','green','blue']
    fig, ax = plt.subplots(2,2,figsize=(10,6))
    
    sns.distplot(list(mixed_data[mixed_data['mixed_pred'] == 0]['age']), hist=True, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label=f'class {0}', color=colors[0], ax=ax[0,0])
    ax[0, 0].set_xlabel('Age', fontsize=14)
    ax[0, 0].set_title('Class 0', fontsize=14)
    ax[0, 0].set_ylabel('Density', fontsize=14)
    sns.distplot(list(mixed_data[mixed_data['mixed_pred'] == 1]['age']), hist=True, kde=True,
                        kde_kws={'shade': True, 'linewidth': 3}, label=f'class {1}', color=colors[1], ax=ax[0,1])
    ax[0, 1].set_xlabel('Age', fontsize=14)
    ax[0, 1].set_title('Class 1', fontsize=14)
    ax[0, 1].set_ylabel('Density', fontsize=14)
    sns.distplot(list(mixed_data[mixed_data['mixed_pred'] == 2]['age']), hist=True, kde=True,
                        kde_kws={'shade': True, 'linewidth': 3}, label=f'class {2}', color=colors[2], ax=ax[1,0])
    ax[1, 0].set_xlabel('Age', fontsize=14)
    ax[1, 0].set_title('Class 2', fontsize=14)
    ax[1, 0].set_ylabel('Density', fontsize=14)
    sns.distplot(list(mixed_data[mixed_data['mixed_pred'] == 3]['age']), hist=True, kde=True,
                        kde_kws={'shade': True, 'linewidth': 3}, label=f'class {3}', color=colors[3], ax=ax[1,1])
    ax[1, 1].set_xlabel('Age', fontsize=14)
    ax[1, 1].set_title('Class 3', fontsize=14) 
    ax[1, 1].set_ylabel('Density', fontsize=14)
    # tight layout
    fig.tight_layout()
    plt.savefig('GFMM_all_figures/GFMM_4class_age_density.png', bbox_inches='tight')
    plt.close()

    # visualize sex breakdown by class
    fig, ax = plt.subplots(1,1,figsize=(7,4))
    mixed_data['sex'].replace({0: 'Female', 1: 'Male'}, inplace=True)
    grouped = mixed_data.groupby('mixed_pred')['sex'].value_counts(normalize=True).reset_index(name='proportion')
    # Plot the proportions
    sns.barplot(data=grouped, x='mixed_pred', y='proportion', hue='sex')
    plt.ylabel('Proportion', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.title('Sex distribution by class', fontsize=14)
    # put legend outside of plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14)
    plt.savefig('GFMM_all_figures/GFMM_4class_sex_distribution_normalized.png', bbox_inches='tight')
    plt.close()
    


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

def get_feature_enrichments(mixed_data, summarize=False):
    '''get feature enrichments for each class'''
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    ## extract dataframes for each class
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    binary_features = []

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        # determine if feature is continuous, binary, categorical based on # of options
        unique = mixed_data[feature].unique()

        if len(unique) in [1,2]:  ## binary
            binary_features.append(feature)
            # perform a hypergeometric-test
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
            # total_in_class4 = len(class4[feature])
            # subset_class4 = np.sum(class4[feature])

            if feature == 'sex':
                # TEST FOR FEMALE ENRICHMENT (0), NOT MALE (1)
                rv0 = hypergeom(total, total - background_prob,
                                total_in_class0)  # initiate random variable distribution
                # sf0 = -np.log10(rv0.sf(total_in_class0-subset_class0 - 1))
                #sf0 = rv0.sf(total_in_class0 - subset_class0 - 1)
                # use binomtest to get p-value
                sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
                
                rv1 = hypergeom(total, total - background_prob,
                                total_in_class1)  # initiate random variable distribution
                # sf1 = -np.log10(rv1.sf(total_in_class1-subset_class1 - 1))
                #sf1 = rv1.sf(total_in_class1 - subset_class1 - 1)
                sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue

                rv2 = hypergeom(total, total - background_prob,
                                total_in_class2)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(total_in_class2-subset_class2 - 1))
                #sf2 = rv2.sf(total_in_class2 - subset_class2 - 1)
                sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue

                rv3 = hypergeom(total, total - background_prob,
                                total_in_class3)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(total_in_class2-subset_class2 - 1))
                #sf3 = rv3.sf(total_in_class3 - subset_class3 - 1)
                sf3 = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='greater').pvalue

                # rv4 = hypergeom(total, total - background_prob, total_in_class4)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(total_in_class2-subset_class2 - 1))
                # sf4 = rv4.sf(total_in_class4 - subset_class4 - 1)
                
                # calculate fold enrichment
                background = background_prob/total
                fe0 = (subset_class0/total_in_class0)/background
                fe1 = (subset_class1/total_in_class1)/background
                fe2 = (subset_class2/total_in_class2)/background
                fe3 = (subset_class3/total_in_class3)/background

                mean0 = np.mean(class0[feature])
                mean1 = np.mean(class1[feature])
                mean2 = np.mean(class2[feature])
                mean3 = np.mean(class3[feature])
                
            else:
                rv0 = hypergeom(total, background_prob, total_in_class0)  # initiate random variable distribution
                # sf0 = -np.log10(rv0.sf(subset_class0 - 1))
                #sf0 = rv0.sf(subset_class0 - 1)
                sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
                sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue

                rv1 = hypergeom(total, background_prob, total_in_class1)  # initiate random variable distribution
                # sf1 = -np.log10(rv1.sf(subset_class1 - 1))
                #sf1 = rv1.sf(subset_class1 - 1)
                sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
                sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue

                rv2 = hypergeom(total, background_prob, total_in_class2)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(subset_class2 - 1))
                #sf2 = rv2.sf(subset_class2 - 1)
                sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
                sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue

                rv3 = hypergeom(total, background_prob, total_in_class3)  # initiate random variable distribution
                # sf3 = -np.log10(rv3.sf(subset_class3 - 1))
                #sf3 = rv3.sf(subset_class3 - 1)
                sf3 = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='greater').pvalue
                sf3_less = binomtest(subset_class3, n=total_in_class3, p=background_prob/total, alternative='less').pvalue

                # rv4 = hypergeom(total, background_prob, total_in_class4)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(subset_class2 - 1))
                # sf4 = rv4.sf(subset_class4 - 1)
                
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
            # one-sided greater: do a t-test to compare the two groups for each class
            pval_class0 = stats.ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = stats.ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = stats.ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature], class3[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class3 = stats.ttest_ind(class3[feature],
                                          pd.concat([class0[feature], class1[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue

            # one-sided less: do a t-test to compare the two groups for each class
            pval_class0_less = stats.ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = stats.ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = stats.ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class3_less = stats.ttest_ind(class3[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            # concat lists to make total list
            #total = pd.concat([class0[feature], class1[feature], class2[feature], class3[feature]], ignore_index=True, sort=False)
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)

            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            # feature_to_pval[feature] = [-np.log10(pval_class0), -np.log10(pval_class1), -np.log10(pval_class2), -np.log10(pval_class3)]
            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, pval_class3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        else:  ## shouldn't happen
            continue
    
    #print(feature_to_pval['derived_cog_impair']); exit()

    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2, 3]
    fold_enrichments['cluster'] = [0, 1, 2, 3]
    mean_values['cluster'] = [0, 1, 2, 3]

    def adjust_pvalues(p_values, method):
        return multipletests(p_values, method=method)[1]

    ### extract the top enriched features for each class (useful for replication analysis)
    pval_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)

    for tested_class in range(4):
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        # multiple hypothesis correction - FDR bh
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  # populate dataframe
            pval_classification_df.loc[key, tested_class] = 1  # populate dataframe

        # class top downregulated:
        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        
        # multiple hypothesis correction - FDR bh
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        # enriched_class_low_dict = enriched_class_low.sort_values(ascending=True).to_dict()
        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  # populate dataframe
            pval_classification_df.loc[key, tested_class] = -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(8), index=mixed_data.columns)
        # each column is a class and either enriched or depleted 
        for i in range(4):
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
                df.loc[key, i+4] = val
        
        # rearrange columns: 0,4,1,5,2,6,3,7
        df = df[[0,4,1,5,2,6,3,7]]
        # rename columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted']
        # move index to column
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        #df.to_csv('../ceph/GFMM_Labeled_Data/SPARK_4classes_enriched_depleted.csv')
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d) and save to file
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
        #df_binary.to_csv('../ceph/GFMM_Labeled_Data/4classes_fold_enrichments_filtered.csv')
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        # process mean values and save to file
        #mean_values = mean_values.drop('cluster', axis=1).T
        #print(mean_values)
        # get 'q01_phrases'
        #mean_values = mean_values.loc['q01_phrases']; exit()
        #mean_values.to_csv('../ceph/GFMM_Labeled_Data/4classes_mean_values.csv')
        
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector

def test_severity_hypothesis():
    invert_features = ['q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted',
                       'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 'q28_things_interested', 'q29_share', 'q30_join_enjoyment',
                       'q31_comfort', 'q32_help_attention', 'q33_range_expressions', 'q34_copy_actions', 'q35_make_believe', 'q36_same_age', 'q37_respond_positively',
                       'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']

    # TEST SEVERITY HYPOTHESIS
    # remove features in invert_features from pval_classification_df
    severity_test = pval_classification_df.drop(invert_features, axis=0)
    invert_feature_df = pval_classification_df.loc[invert_features]
    # for each class, count number of features that are enriched (severity test df) and add it to number of features in invert_feature_df that are == -1 (depleted)
    severity_scores = []
    # in severity test: replace all -1 with 0
    severity_test = severity_test.replace(-1, 0)
    # in invert_feature_df, replace all 1 with 0
    invert_feature_df = invert_feature_df.replace(1, 0)
    # replace all -1 with 1
    invert_feature_df = invert_feature_df.replace(-1, 1)
    for i in range(4):
        # sum up ith column of severity test df and invert_feature_df and add them. append to severity_scores
        severity_scores.append(np.sum(severity_test.iloc[:, i]) + np.sum(invert_feature_df.iloc[:, i]))
    print(severity_scores)

    # plot in a bar plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.bar(np.arange(4), severity_scores, color=['red', 'violet', 'green', 'blue'])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    ax.set_ylabel('Severity Score Count')
    ax.set_xlabel('Class')
    plt.savefig('GFMM_all_figures/severity_score_count_binary.png', bbox_inches='tight')
    plt.close()


def get_feature_enrichments_3classes(mixed_data):
    '''specific for n_components=3'''
    ### HYPERGEOMETRIC TEST
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()

    ## extract dataframes for each class
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    # print(np.mean(class0['age']))
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    # print(np.mean(class1['age']))
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    # print(np.mean(class2['age']))

    print(f'class0 size: {len(class0)}')
    print(f'class1 size: {len(class1)}')
    print(f'class2 size: {len(class2)}')

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        # determine if feature is continuous, binary, categorical based on # of options
        unique = mixed_data[feature].unique()

        if len(unique) == 2:  ## binary
            # perform a hypergeometric-test
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))

            if feature == 'sex':
                # TEST FOR FEMALE ENRICHMENT (0), NOT MALE (1)
                #rv0 = hypergeom(total, total - background_prob,
                #                total_in_class0)  # initiate random variable distribution
                # sf0 = -np.log10(rv0.sf(total_in_class0-subset_class0 - 1))
                #sf0 = rv0.sf(total_in_class0 - subset_class0 - 1)
                sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue


                #rv1 = hypergeom(total, total - background_prob,
                #                total_in_class1)  # initiate random variable distribution
                # sf1 = -np.log10(rv1.sf(total_in_class1-subset_class1 - 1))
                #sf1 = rv1.sf(total_in_class1 - subset_class1 - 1)
                sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue


                #rv2 = hypergeom(total, total - background_prob,
                #                total_in_class2)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(total_in_class2-subset_class2 - 1))
                #sf2 = rv2.sf(total_in_class2 - subset_class2 - 1)
                sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue


            else:
                #rv0 = hypergeom(total, background_prob, total_in_class0)  # initiate random variable distribution
                # sf0 = -np.log10(rv0.sf(subset_class0 - 1))
                #sf0 = rv0.sf(subset_class0 - 1)
                sf0 = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='greater').pvalue
                sf0_less = binomtest(subset_class0, n=total_in_class0, p=background_prob/total, alternative='less').pvalue


                #rv1 = hypergeom(total, background_prob, total_in_class1)  # initiate random variable distribution
                # sf1 = -np.log10(rv1.sf(subset_class1 - 1))
                #sf1 = rv1.sf(subset_class1 - 1)
                sf1 = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='greater').pvalue
                sf1_less = binomtest(subset_class1, n=total_in_class1, p=background_prob/total, alternative='less').pvalue


                #rv2 = hypergeom(total, background_prob, total_in_class2)  # initiate random variable distribution
                # sf2 = -np.log10(rv2.sf(subset_class2 - 1))
                #sf2 = rv2.sf(subset_class2 - 1)
                sf2 = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='greater').pvalue
                sf2_less = binomtest(subset_class2, n=total_in_class2, p=background_prob/total, alternative='less').pvalue


            feature_to_pval[feature] = [sf0, sf1, sf2]
            feature_sig_df_high[feature] = [sf0, sf1, sf2]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less]
            feature_vector.append(feature)

        elif len(unique) > 2:  ## continuous or categorical
            # one-sided greater: do a t-test to compare the two groups for each class
            pval_class0 = stats.ttest_ind(class0[feature],
                                          pd.concat([class1[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class1 = stats.ttest_ind(class1[feature],
                                          pd.concat([class0[feature], class2[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue
            pval_class2 = stats.ttest_ind(class2[feature],
                                          pd.concat([class0[feature], class1[feature]],
                                                    ignore_index=True, sort=False), equal_var=False,
                                          alternative='greater').pvalue

            # one-sided less: do a t-test to compare the two groups for each class
            pval_class0_less = stats.ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = stats.ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = stats.ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            # feature_to_pval[feature] = [-np.log10(pval_class0), -np.log10(pval_class1), -np.log10(pval_class2), -np.log10(pval_class3)]
            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less]
            feature_vector.append(feature)

        else:  ## shouldn't happen
            continue

    #print(feature_to_pval['sex']); exit()
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2]

    #return feature_sig_norm_high, feature_vector

    def adjust_pvalues(p_values, method):
        return multipletests(p_values, method=method)[1]

    ### extract the top enriched features for each class (useful for replication analysis)
    pval_df = pd.DataFrame(columns=np.arange(3), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(3), index=mixed_data.columns)

    for tested_class in range(3):
        # tested_class = 3
        # class top upregulated:
        # print(f'enriched class {tested_class} high:')
        enriched_class_high = feature_sig_norm_high[feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                                                                                                           axis=1).T.dropna(
            axis=0)
        # multiple hypothesis correction - FDR bh
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        # multiple hypothesis correction
        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  # populate dataframe
            pval_classification_df.loc[key, tested_class] = 1  # populate dataframe

        # class top downregulated:
        # print(f'enriched class {tested_class} low:')
        enriched_class_low = feature_sig_norm_low[feature_sig_norm_low['cluster'] == tested_class].drop('cluster',
                                                                                                        axis=1).T.dropna(
            axis=0)
        # multiple hypothesis correction - FDR bh
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        # enriched_class_low_dict = enriched_class_low.sort_values(ascending=True).to_dict()
        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  # populate dataframe
            pval_classification_df.loc[key, tested_class] = -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)
    #pval_classification_df.to_csv('/mnt/home/alitman/ceph/SPARK_replication/SPARK_3classes_pval_replication_classification.csv')

    return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector

def plot_line_polar(data, output):
    '''create line polar plots to summarize class identifying features'''
    
    y = list(data['mixed_pred'])
    X = data.drop(['mixed_pred'], axis=1)
    
    # normalize:
    scaler = StandardScaler()
    mixed_data_norm = scaler.fit_transform(X)
    mixed_data_norm = pd.DataFrame(mixed_data_norm, columns=X.columns)
    mixed_data_norm['cluster'] = y
    
    # 'final_score', 'dev_lang_dis'
    features_to_visualize = ['social_problems_t_score', 'dsm5_anxiety_problems_t_score', 'i_stereotyped_behavior_score', 'vi_restricted_behavior_score', 'iii_compulsive_behavior_score', 'derived_cog_impair', 'cluster'] # for cbcl model
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
    polar['variable'] = polar['variable'].replace('dsm5_anxiety_problems_t_score', 'Anxiety Problems')
    polar['variable'] = polar['variable'].replace('iii_compulsive_behavior_score', 'Compulsive Behavior')
    polar['variable'] = polar['variable'].replace('vi_restricted_behavior_score', 'Restricted Behavior')
    polar['variable'] = polar['variable'].replace('v_sameness_behavior_score', 'Sameness Behavior')
    polar['variable'] = polar['variable'].replace('dev_id', 'Intellectual Disability')
    polar['variable'] = polar['variable'].replace('derived_cog_impair', 'Intellectual Disability')
    polar['variable'] = polar['variable'].replace('final_score', 'SCQ Score')
    #polar4['variable'] = polar4['variable'].replace('ml_predicted_cog_impair', 'Intellectual Disability (ML-Predicted)')
    polar['variable'] = polar['variable'].replace('dev_lang_dis', 'Language Delay')

    colors = ['red','violet','green','blue'] # 4700 COHORT
    #colors = ['violet','red','green','blue'] # 6400 COHORT
    fig = px.line_polar(polar, r="value", theta="variable", color="cluster", 
                        color_discrete_sequence=colors, line_close=True, height=800, width=1400) # template="plotly_dark",
    fig.update_layout(
        font_size=36
    )
    # make lines thicker
    fig.update_traces(line=dict(width=4))
    # bold radial line at 0
    fig.update_polars(radialaxis=dict(visible=True, linewidth=2, tickwidth=2, ticklen=10))
    fig.update_polars(angularaxis=dict(visible=True, linewidth=2, tickwidth=2, ticklen=10))
    #fig.update_polars(radialaxis_showticklabels=False)
    fig.update_polars(angularaxis_showticklabels=True)
    # make circles black and bold
    fig.update_polars(radialaxis=dict(gridcolor='black', gridwidth=0.5))
    # change legend labels
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("cluster=", "Class ")))
    fig.update_layout(legend_title_text='Class')
    fig.write_image(f"{output}_polar.png")

def run_continuous_LCA_model_with_feature_subset(subset_data, main_data, covariates, ncomp):
    '''run LCA model with continuous variables (no binary or categorical variables).'''
    idx = list(subset_data.index)
    model = StepMix(n_components=ncomp, measurement='continuous',
                    structural='covariate',
                    n_steps=1, random_state=123, verbose=1)
    model.fit(subset_data.reset_index(drop=True), covariates)
    subset_data['mixed_pred'] = model.predict(subset_data)
    subset_data.index = idx

    ## extract values for classes for class membership dictionaries
    class0subset = list(subset_data[subset_data['mixed_pred'] == 0].index)
    class1subset = list(subset_data[subset_data['mixed_pred'] == 1].index)
    class2subset = list(subset_data[subset_data['mixed_pred'] == 2].index)
    class3subset = list(subset_data[subset_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    subset_class_membership = dict()
    subset_class_membership[0] = class0subset
    subset_class_membership[1] = class1subset
    subset_class_membership[2] = class2subset
    subset_class_membership[3] = class3subset

    class0 = list(main_data[main_data['mixed_pred'] == 0].index)
    class1 = list(main_data[main_data['mixed_pred'] == 1].index)
    class2 = list(main_data[main_data['mixed_pred'] == 2].index)
    class3 = list(main_data[main_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    class_membership = dict()
    class_membership[0] = class0
    class_membership[1] = class1
    class_membership[2] = class2
    class_membership[3] = class3

    return subset_data, subset_class_membership, class_membership


def run_continuous_categorical_LCA_model_with_feature_subset(subset_data, main_data, continuous_columns, categorical_columns, covariates, ncomp):
    '''run LCA model with continuous and categorical variables (mixed), but no binary variables.'''
    idx = list(subset_data.index)
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=subset_data.reset_index(drop=True),
        continuous=continuous_columns,
        categorical=categorical_columns
    )

    # LCA using StepMix
    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=123, verbose=1)
    model.fit(mixed_data.reset_index(drop=True), covariates)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data.index = idx

    ## extract values for classes for class membership dictionaries
    class0subset = list(mixed_data[mixed_data['mixed_pred'] == 0].index)
    class1subset = list(mixed_data[mixed_data['mixed_pred'] == 1].index)
    class2subset = list(mixed_data[mixed_data['mixed_pred'] == 2].index)
    class3subset = list(mixed_data[mixed_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    subset_class_membership = dict()
    subset_class_membership[0] = class0subset
    subset_class_membership[1] = class1subset
    subset_class_membership[2] = class2subset
    subset_class_membership[3] = class3subset

    class0 = list(main_data[main_data['mixed_pred'] == 0].index)
    class1 = list(main_data[main_data['mixed_pred'] == 1].index)
    class2 = list(main_data[main_data['mixed_pred'] == 2].index)
    class3 = list(main_data[main_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    class_membership = dict()
    class_membership[0] = class0
    class_membership[1] = class1
    class_membership[2] = class2
    class_membership[3] = class3

    return mixed_data, subset_class_membership, class_membership


def run_LCA_model_with_feature_subset(subset_data, main_data, continuous_columns, binary_columns, categorical_columns, covariates, ncomp):
    '''run LCA model with continuous, binary, and categorical variables (mixed).'''
    idx = list(subset_data.index)
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=subset_data.reset_index(drop=True),
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # LCA using StepMix
    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=123, verbose=1)
    model.fit(mixed_data.reset_index(drop=True), covariates)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data.index = idx

    ## extract values for classes for class membership dictionaries
    class0subset = list(mixed_data[mixed_data['mixed_pred'] == 0].index)
    class1subset = list(mixed_data[mixed_data['mixed_pred'] == 1].index)
    class2subset = list(mixed_data[mixed_data['mixed_pred'] == 2].index)
    class3subset = list(mixed_data[mixed_data['mixed_pred'] == 3].index)
    print(f'class0 size: {len(class0subset)}')
    print(f'class1 size: {len(class1subset)}')
    print(f'class2 size: {len(class2subset)}')
    print(f'class3 size: {len(class3subset)}')
    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    subset_class_membership = dict()
    subset_class_membership[0] = class0subset
    subset_class_membership[1] = class1subset
    subset_class_membership[2] = class2subset
    subset_class_membership[3] = class3subset

    class0 = list(main_data[main_data['mixed_pred'] == 0].index)
    class1 = list(main_data[main_data['mixed_pred'] == 1].index)
    class2 = list(main_data[main_data['mixed_pred'] == 2].index)
    class3 = list(main_data[main_data['mixed_pred'] == 3].index)

    # generate class membership dictionary to see overlap with Jaccard Index heatmap
    class_membership = dict()
    class_membership[0] = class0
    class_membership[1] = class1
    class_membership[2] = class2
    class_membership[3] = class3

    return mixed_data, subset_class_membership, class_membership

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

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def run_LCA_model_for_f_classif(subset_file, main_data):
    
    subset_data = pd.read_csv(subset_file, index_col=0)

    continuous_columns, binary_columns, categorical_columns = split_columns(list(subset_data.columns))
    if (len(binary_columns) == 0) and (len(categorical_columns) == 0):
        # run LCA model with continuous variables only (not mixed)
        subset_data, subset_class_memberships, class_memberships = run_continuous_LCA_model_with_feature_subset(subset_data, main_data)
    elif (len(binary_columns) == 0) and (len(categorical_columns) != 0):
        # run mixed model with only continous and categorical variables
        subset_data, subset_class_memberships, class_memberships = run_continuous_categorical_LCA_model_with_feature_subset(subset_data, main_data, continuous_columns, categorical_columns)
    else:
        # run mixed model with continuous, binary, categorical variables
        subset_data, subset_class_memberships, class_memberships = run_LCA_model_with_feature_subset(subset_data, main_data, continuous_columns, binary_columns, categorical_columns)
    
    jaccard_index = np.zeros((4,4))
    for i, list1 in subset_class_memberships.items():
        for j, list2 in class_memberships.items():
            jac = jaccard(list1, list2)
            # assign j,i element of jaccard_index to jac
            jaccard_index[j,i] = jac
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax = sns.heatmap(jaccard_index, cmap='BuPu', annot=True)
    ax.set(xlabel="Subset Data (20%) Class Membership", ylabel="Main Data Class Membership")
    plt.savefig('GFMM_all_figures/f_classif_66perc_nocbcl_jaccard_index_heatmap.png', bbox_inches='tight')

    subset_data['sex'] = main_data['sex']

    return subset_data

def run_LCA_for_WGS_data(main_data, nocbcl_class_membership):
    wgs = '/mnt/home/alitman/ceph/WGS_probands_LCA_training_data_iterative_simple.csv'
    wgs_rf = '/mnt/home/alitman/ceph/WGS_probands_LCA_training_data_iterative_rf.csv' # RandomForestRegression imputation
    wgs_complete = '/mnt/home/alitman/ceph/WGS_probands_only_complete.csv'
    wgs_rm1 = '/mnt/home/alitman/ceph/WGS_probands_LCA_training_data_rm_1perc_iterative.csv' # removed features with >1% missing data, imputed rest
    wgs_data = pd.read_csv(wgs, index_col=0)

    continuous_columns, binary_columns, categorical_columns = split_columns(list(wgs_data.columns))
    if (len(binary_columns) == 0) and (len(categorical_columns) == 0):
        # run LCA model with continuous variables only (not mixed)
        subset_data, subset_class_memberships, class_memberships = run_continuous_LCA_model_with_feature_subset(wgs_data, main_data, ncomp=4)
    elif (len(binary_columns) == 0) and (len(categorical_columns) != 0):
        # run mixed model with only continous and categorical variables
        subset_data, subset_class_memberships, class_memberships = run_continuous_categorical_LCA_model_with_feature_subset(wgs_data, main_data, continuous_columns, categorical_columns, ncomp=4)
    else:
        # run mixed model with continuous, binary, categorical variables
        subset_data, subset_class_memberships, class_memberships = run_LCA_model_with_feature_subset(wgs_data, main_data, continuous_columns, binary_columns, categorical_columns, ncomp=4)
    
    #print(subset_data.shape)
    #subset_data.to_csv('/mnt/home/alitman/ceph/WGS_probands_LCA_4classes_labeled_iterative_simple.csv')
    jaccard_index = np.zeros((4,4)) # change to 3 if using 3 classes
    for i, list1 in subset_class_memberships.items():
        for j, list2 in nocbcl_class_membership.items():
            jac = jaccard(list1, list2)
            # assign j,i element of jaccard_index to jac
            jaccard_index[j,i] = jac
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax = sns.heatmap(jaccard_index, cmap='BuPu', annot=True)
    ax.set(xlabel="WGS Data Class Membership", ylabel="Main Data Class Membership")
    plt.savefig('GFMM_all_figures/WGS_4classes_jaccard_heatmap_iterative_simple.png', bbox_inches='tight')

    wgs_pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments(subset_data)
    plot_line_polar(subset_data, 'WGS_LCA_4classes_line_polar_plot_rm1_iterative')
    
    # compare direction of enrichment for each feature between main model and WGS model
    spark_pval_classification = pd.read_csv('/mnt/home/alitman/ceph/SPARK_replication/SPARK_4classes_pval_replication_classification.csv', index_col=0)
    nocbcl_classification = pd.read_csv('/mnt/home/alitman/ceph/LCA_4classes_no_cbcl_training_data.csv', index_col=0)
    # intersect index to find common features
    idx_intersect = np.intersect1d(wgs_pval_classification_df.index, spark_pval_classification.index)

    spark_pval_classification = spark_pval_classification.loc[idx_intersect]
    wgs_pval_classification_df = wgs_pval_classification_df.loc[idx_intersect]
    
    corr_array = pd.DataFrame(columns=np.arange(4), index=np.arange(4), dtype=float) # change to 3 if using 3 classes
    for i in range(4):
        for j in range(4): # change to 3 if using 3 classes
            if pearsonr(list(spark_pval_classification.iloc[:,i]), list(wgs_pval_classification_df.iloc[:,j])).statistic >= 0:
                corr_array.loc[i, j] = pearsonr(list(spark_pval_classification.iloc[:,i]), list(wgs_pval_classification_df.iloc[:,j])).statistic
            else:
                corr_array.loc[i, j] = 0 # cutoff at 0 so we don't see - correlations

    # heatmap for correlation of enrichment patterns
    #print(pearsonr(spark_pval_classification.iloc[:,0], ssc_pval_classification.iloc[:,0]).statistic)
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    sns.heatmap(corr_array, cmap="PuRd", annot=True) # BuPu, YlGnBu
    plt.ylabel('SPARK classes', fontsize=18)
    plt.xlabel('WGS classes', fontsize=18)
    plt.title('Correlation of enrichment patterns (n_classes=4)', fontsize=18)
    plt.savefig('GFMM_all_figures/WGS_spark_4classes_correlation_heatmap_iterative_simple.png')


if __name__ == '__main__':
    df_enriched_depleted, fold_enrichments = run_lca_no_bms(ncomp=4, summarize=True)
    generate_summary_table(df_enriched_depleted, fold_enrichments); exit()

    main_data = run_main_LCA_model(ncomp=4) # returns data with labels for each row
    nocbcl_class_membership = run_lca_no_cbcl(main_data); exit()
    #run_LCA_for_WGS_data(main_data, nocbcl_class_membership); exit()

    #logreg_file = '/mnt/home/alitman/ceph/RFE_LogReg_selected_features_for_LCA.csv'
    #selectfrommodel_logreg_file = '/mnt/home/alitman/ceph/select_from_model_l1_logreg_for_LCA.csv'
    #sfs_file = '/mnt/home/alitman/ceph/LCA_4classes_training_data_SFS_selected_features.csv'
    svc_file = '/mnt/home/alitman/ceph/feature_selection/RFE_SVC_selected_features_for_LCA.csv'
    percent20_file = '/mnt/home/alitman/ceph/feature_selection/SVC-Anova_selected_features_for_LCA_20percent.csv'
    subset_data_nocbcl = '/mnt/home/alitman/ceph/feature_selection/SVC-Anova_selected_features_for_LCA_66percent_no_cbcl.csv'
    subset_data = run_LCA_model_for_f_classif(subset_data_nocbcl, main_data); exit()
    #subset_data = run_LCA_model_for_RFE_SVC(svc_file, main_data)
    #get_enrichment_plot_line_polar(subset_data, 'f_classif_20perc_feature_selection_polar_plot'); exit()

    feature_importances_file = 'feature_importances_random_forest.csv'
    feature_importances_w_perm_file = 'feature_importances_w_permutation_random_forest.csv'
    feature_importances_file_all_data = 'feature_importances_random_forest_all_data.csv'
    # iteratively feed more features to LCA model; write function to split input into continuous, categorical, and binary variables
    #iterate_through_features(feature_importances_w_perm_file, main_data)
    #test_top_feature_model(feature_importances_w_perm_file, main_data)
    