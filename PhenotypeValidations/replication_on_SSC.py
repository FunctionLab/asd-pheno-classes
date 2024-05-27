import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats

from ../PhenotypeClasses/GFMM import get_feature_enrichments
from ../PhenotypeClasses/utils import split_columns


def generate_ssc_data():
    ### Preprocess, clean, and integrate SSC data across cohort
    ssc_data_dir = '../SSC_Phenotype_Dataset/Proband_Data'
    
    ## CBCL
    cbcl = pd.read_csv(f'{ssc_data_dir}/cbcl_6_18.csv').set_index('individual',drop=True).drop(['measure', 'activities_total', 'add_adhd_total',
                                                                                                'affective_problems_total', 'aggressive_behavior_total', 'anxiety_problems_total', 'anxious_depressed_total',
                                                                                                'attention_problems_total', 'conduct_problems_total', 'externalizing_problems_total', 'internalizing_problems_total',
                                                                                                'oppositional_defiant_total', 'rule_breaking_total', 'school_total', 'social_problems_total', 'social_total',
                                                                                                'somatic_complaints_total', 'somatic_prob_total', 'thought_problems_total', 'total_competence_total',
                                                                                                'total_problems_total', 'withdrawn_total', 'activities_t_score', 'school_t_score', 'total_competence_t_score',
                                                                                                'social_t_score'], axis=1)
    # rename CBCL scores to match spark feature names
    cbcl.rename(columns={'add_adhd_t_score': 'dsm5_attention_deficit_hyperactivity_t_score', 'affective_problems_t_score': 'dsm5_depressive_problems_t_score', 'aggressive_behavior_t_score': 'aggressive_behavior_t_score',
                        'anxiety_problems_t_score': 'dsm5_anxiety_problems_t_score', 'anxious_depressed_t_score': 'anxious_depressed_t_score', 'attention_problems_t_score': 'attention_problems_t_score',
                        'conduct_problems_t_score': 'dsm5_conduct_problems_t_score', 'externalizing_problems_t_score': 'externalizing_problems_t_score', 'internalizing_problems_t_score': 'internalizing_problems_t_score',
                        'oppositional_defiant_t_score': 'dsm5_oppositional_defiant_t_score', 'rule_breaking_t_score': 'rule_breaking_behavior_t_score', 'social_problems_t_score': 'social_problems_t_score',
                        'somatic_complaints_t_score': 'somatic_complaints_t_score', 'thought_problems_t_score': 'thought_problems_t_score',
                        'total_problems_t_score': 'total_problems_t_score', 'withdrawn_t_score': 'withdrawn_depressed_t_score', 'somatic_prob_t_score': 'dsm5_somatic_problems_t_score'}, inplace=True)
    
    ## RBS-R
    rbsr_scores = pd.read_csv(f'{ssc_data_dir}/rbs_r.csv').set_index('individual',drop=True).drop(['measure', 'overall_number_items', 'overall_score', 'status', 'iii_compulsive_behavior_items', 'ii_self_injurious_items',
                                                                                                    'i_stereotyped_behavior_items', 'iv_ritualistic_behavior_items',
                                                                                                    'vi_restricted_behavior_items', 'v_sameness_behavior_items'], axis=1) 
    rbsr_raw = pd.read_csv(f'{ssc_data_dir}/rbs_r_raw.csv').set_index('individual',drop=True).drop(['measure'], axis=1) 
    rbsr_raw.rename(columns={'q39_insists_palce': 'q39_insists_time'}, inplace=True)
    
    ## SCQ
    scq_raw = pd.read_csv(f'{ssc_data_dir}/scq_life_recode.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq_raw.replace('yes', 1, inplace=True)
    scq_raw.replace('no', 0, inplace=True)
    scq_raw.rename(columns={'q08_hits_self_object': 'q08_hits_self_against_object', 'q09_hits_self_object': 'q09_hits_self_with_object',
                            'q28_communicatiion': 'q28_communication'}, inplace=True)
    scq = pd.read_csv(f'{ssc_data_dir}/scq_life.csv').set_index('individual', drop=True).drop(['measure'], axis=1)
    scq.rename(columns={'summary_score': 'final_score'}, inplace=True)
    
    ## CORE DESCRIPTIVE  
    core_descriptive = pd.read_csv(f'{ssc_data_dir}/ssc_core_descriptive.csv').set_index('individual', drop=True).drop(['ssc_diagnosis_full_scale_iq', 'ssc_diagnosis_nonverbal_iq', 'ssc_diagnosis_verbal_iq', 'measure', 'abc_total_score', 'adi_r_b_comm_verbal_total', 'adi_r_comm_b_non_verbal_total', 'adi_r_cpea_dx', 'adi_r_evidence_onset', 'adi_r_rrb_c_total', 'adi_r_soc_a_total', 'ados_communication_social', 'ados_css', 'ados_module', 'ados_restricted_repetitive', 'ados_social_affect', 'cbcl_2_5_externalizing_t_score', 'cbcl_2_5_internalizing_t_score', 'cbcl_6_18_externalizing_t_score', 'cbcl_6_18_internalizing_t_score', 'cpea_dx', 'diagnosis_ados', 'ethnicity', 'family_type', 'febrile_seizures', 'non_febrile_seizures', 'pregnancy_optimality_code', 'pregnancy_optimality_code_intrapartal', 'pregnancy_optimality_code_neonatal', 'pregnancy_optimality_code_prenatal', 'pregnancy_optimality_intrapartal', 'pregnancy_optimality_neonatal', 'pregnancy_optimality_prenatal', 'pregnancy_optimality_total', 'puberty_ds_progress', 'puberty_ds_total', 'race', 'rbs_r_overall_score', 'regression', 'regression_loss', 'regression_no_insert', 'srs_parent_raw_total', 'srs_parent_t_score', 'srs_teacher_raw_total', 'srs_teacher_t_score', 'ssc_diagnosis_full_scale_iq_type', 'ssc_diagnosis_nonverbal_iq_type', 'ssc_diagnosis_nvma', 'ssc_diagnosis_verbal_iq_type', 'ssc_diagnosis_vma', 'vineland_ii_composite_standard_score'], axis=1)
    core_descriptive.replace('male', 1, inplace=True)
    core_descriptive.replace('female', 0, inplace=True)
    core_descriptive.rename(columns={'age_at_ados': 'age_at_eval_years'}, inplace=True)
    core_descriptive['age_at_eval_years'] = core_descriptive['age_at_eval_years']/12
    
    ## BACKGROUND HISTORY
    bh = pd.read_csv(f'{ssc_data_dir}/SSC_background_hx_clean.csv').set_index('individual', drop=True)
    bh = bh.drop(['age_gave_up_bottle', 'age_started_solid_foods'], axis=1)
    
    finaldf = pd.concat([core_descriptive, bh, scq_raw, scq, rbsr_raw, rbsr_scores, cbcl], axis=1, join='inner')  
    finaldf = finaldf.loc[:, finaldf.isna().sum()/finaldf.shape[0] < 0.1]
    finaldf = finaldf.dropna(axis=0)
    finaldf = finaldf.astype('float32')
    print(finaldf.shape)
    
    return finaldf


def get_SSC_data(ncomp=4):
    finaldf = generate_ssc_data()
    Z_p = finaldf[['sex', 'age_at_eval_years']] # covariates

    X = finaldf.drop(['sex', 'age_at_eval_years'], axis=1)

    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate', n_steps=1, n_init=200) 
    model.fit(mixed_data)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data.to_csv('data/SSC_replication_mixed_pred_final.csv')
    
    # compute feature enrichments
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()

    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    ssc_pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, summary_df, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)

    features_to_exclude = fold_enrichments.copy()
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # exclude features based on criteria:
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

    features_to_category = pd.read_csv('data/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))
    summary_df = summary_df.fillna('NaN')
    summary_df = summary_df.replace(np.nan, 1)
    summary_df = summary_df.loc[~summary_df['feature'].isin(features_to_exclude)] # remove non-contributory features with no significant enrichments
    ssc_feature_subset = summary_df['feature'].to_list()
    summary_df['feature_category'] = summary_df['feature'].map(feature_to_category)
    summary_df = summary_df.dropna(subset=['feature_category'])

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
    
    summary_df['class0_enriched'] = summary_df['class0_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class0_depleted'] = summary_df['class0_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_enriched'] = summary_df['class1_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class1_depleted'] = summary_df['class1_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_enriched'] = summary_df['class2_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class2_depleted'] = summary_df['class2_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class3_enriched'] = summary_df['class3_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    summary_df['class3_depleted'] = summary_df['class3_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    
    prop_df = pd.DataFrame()
    prop_df['class0_enriched'] = summary_df.groupby(['feature_category'])['class0_enriched'].sum()/summary_df.groupby(['feature_category'])['class0_enriched'].count()
    prop_df['class0_depleted'] = summary_df.groupby(['feature_category'])['class0_depleted'].sum()/summary_df.groupby(['feature_category'])['class0_depleted'].count()
    prop_df['class1_enriched'] = summary_df.groupby(['feature_category'])['class1_enriched'].sum()/summary_df.groupby(['feature_category'])['class1_enriched'].count()
    prop_df['class1_depleted'] = summary_df.groupby(['feature_category'])['class1_depleted'].sum()/summary_df.groupby(['feature_category'])['class1_depleted'].count()
    prop_df['class2_enriched'] = summary_df.groupby(['feature_category'])['class2_enriched'].sum()/summary_df.groupby(['feature_category'])['class2_enriched'].count()
    prop_df['class2_depleted'] = summary_df.groupby(['feature_category'])['class2_depleted'].sum()/summary_df.groupby(['feature_category'])['class2_depleted'].count()
    prop_df['class3_enriched'] = summary_df.groupby(['feature_category'])['class3_enriched'].sum()/summary_df.groupby(['feature_category'])['class3_enriched'].count()
    prop_df['class3_depleted'] = summary_df.groupby(['feature_category'])['class3_depleted'].sum()/summary_df.groupby(['feature_category'])['class3_depleted'].count()
    
    prop_df['class0_depleted'] = -prop_df['class0_depleted']
    prop_df['class1_depleted'] = -prop_df['class1_depleted']
    prop_df['class2_depleted'] = -prop_df['class2_depleted']
    prop_df['class3_depleted'] = -prop_df['class3_depleted']

    prop_df['class0_max'] = prop_df[['class0_enriched', 'class0_depleted']].sum(axis=1)
    prop_df['class1_max'] = prop_df[['class1_enriched', 'class1_depleted']].sum(axis=1)
    prop_df['class2_max'] = prop_df[['class2_enriched', 'class2_depleted']].sum(axis=1)
    prop_df['class3_max'] = prop_df[['class3_enriched', 'class3_depleted']].sum(axis=1)

    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'] 
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = [0, 1, 2, 3]

    # compute correlations between SSC and SPARK
    plot_df = prop_df.T
    plot_df['cluster'] = np.arange(4)
    polar = plot_df.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    polar.rename(columns={'value': 'ssc_value'}, inplace=True)
    
    spark, spark_category_to_features = run_spark_model(ssc_feature_subset, subset=True)
    spark.rename(columns={'value': 'spark_value'}, inplace=True)
    spark_df = spark.T
    spark_df['cluster'] = np.arange(4)
    # qualitative mapping of SPARK classes to SSC classes
    spark_df['cluster'] = spark_df['cluster'].map({1: 0, 0: 1, 2: 2, 3: 3}) 
    spark = spark_df.groupby('cluster').mean().reset_index()
    spark = pd.melt(spark, id_vars=['cluster'])
    spark.rename(columns={'value': 'spark_value'}, inplace=True)

    polar = pd.merge(polar, spark, on=['cluster', 'variable'], how='inner')
    
    # 1. barplot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    polar['variable_shifted_left'] = polar['variable'] - 0.2
    polar['variable_shifted_right'] = polar['variable'] + 0.2
    sns.barplot(x='variable_shifted_left', y='ssc_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, palette=['violet','red','limegreen','blue'])
    sns.barplot(x='variable_shifted_right', y='spark_value', hue='cluster', data=polar, ax=ax, alpha=0.8, width=0.45, palette=['violet','red','limegreen','blue'])
    plt.ylabel('Proportion+direction of sig. features', fontsize=18)
    plt.xlabel('')
    ax.get_legend().remove()
    plt.xticks(rotation=35, ha='right', fontsize=16)
    plt.savefig(f'figures/SSC_SPARK_replication_7classes_barplot.png', bbox_inches='tight')
    plt.close()
    
    # 2. scatter plot
    shapes = ['o', 'v', 'p', '^', 'd', "P", 's', '>', '*', 'X', 'D']
    shapes = shapes[:len(features_to_visualize)]
    polar['color'] = polar['cluster'].map({0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
    polar['shape'] = polar['variable'].map({0: 'o', 1: 'v', 2: 'p', 3: '*', 4: 'd', 5: "P", 6: 's', 7: 'X', 8: '>', 9: 'D', 10: '^'})
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i, row in polar.iterrows():
        ax.scatter(row['spark_value'], row['ssc_value'], color=row['color'], marker=row['shape'], s=165, alpha=0.7)
    x = np.linspace(-1, 1, 100)
    y = x
    ax.plot(x, y, color='gray', linestyle='--')
    r, p = stats.pearsonr(polar['ssc_value'], polar['spark_value'])
    r2 = r**2
    ax.text(0.3, 0.11, f'R^2: {r2:.2f}', fontsize=16)
    ax.text(0.3, 0.01, f'p < 1e-5', fontsize=16)
    plt.xlabel('SPARK', fontsize=20)
    plt.ylabel('SSC', fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
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
    plt.savefig(f'figures/SSC_SPARK_replication_7classes_scatterplot.png', bbox_inches='tight')
    plt.close()


def run_spark_model(ssc_feature_subset, subset=False):
    mixed_data = pd.read_csv('../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0)

    # subset to features used in ssc model
    if subset:
        ssc_feature_subset = [x for x in ssc_feature_subset if x in mixed_data.columns]
        mixed_data = mixed_data[ssc_feature_subset]

    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)
    
    features_to_exclude = fold_enrichments.copy()
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # exclude features based on criteria:
    # (1) features with no significant enrichments in any class
    # (2) features with all cohen's d values < 0.2 or FE < 1.5
    # (3) features where all classes is nan
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

    features_to_category = pd.read_csv('data/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))
    df = df_enriched_depleted.copy()
    df = df.fillna('NaN')
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    df = df.loc[~df['feature'].isin(features_to_exclude)]
    df['feature_category'] = df['feature'].map(feature_to_category)
    df = df.dropna(subset=['feature_category'])
    df = df.replace('NaN', 1)

    feature_category_to_features = dict()
    for category in df['feature_category'].unique():
        feature_category_to_features[category] = df.loc[df['feature_category'] == category, 'feature'].to_list()
    
    df['class0_enriched'] = df['class0_enriched'].astype(float)
    df['class0_depleted'] = df['class0_depleted'].astype(float)
    df['class1_enriched'] = df['class1_enriched'].astype(float)
    df['class1_depleted'] = df['class1_depleted'].astype(float)
    df['class2_enriched'] = df['class2_enriched'].astype(float)
    df['class2_depleted'] = df['class2_depleted'].astype(float)
    df['class3_enriched'] = df['class3_enriched'].astype(float)
    df['class3_depleted'] = df['class3_depleted'].astype(float)
    # convert p-value columns to binary values (1 if significant, 0 if not)
    df['class0_enriched'] = df['class0_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class0_depleted'] = df['class0_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class1_enriched'] = df['class1_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class1_depleted'] = df['class1_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class2_enriched'] = df['class2_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class2_depleted'] = df['class2_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class3_enriched'] = df['class3_enriched'].apply(lambda x: 1 if x < 0.05 else 0)
    df['class3_depleted'] = df['class3_depleted'].apply(lambda x: 1 if x < 0.05 else 0)
    
    # flip_rows contains features which are reverse-coded in the original data
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
    
    prop_df = pd.DataFrame()
    prop_df['class0_enriched'] = df.groupby(['feature_category'])['class0_enriched'].sum()/df.groupby(['feature_category'])['class0_enriched'].count()
    prop_df['class0_depleted'] = df.groupby(['feature_category'])['class0_depleted'].sum()/df.groupby(['feature_category'])['class0_depleted'].count()
    prop_df['class1_enriched'] = df.groupby(['feature_category'])['class1_enriched'].sum()/df.groupby(['feature_category'])['class1_enriched'].count()
    prop_df['class1_depleted'] = df.groupby(['feature_category'])['class1_depleted'].sum()/df.groupby(['feature_category'])['class1_depleted'].count()
    prop_df['class2_enriched'] = df.groupby(['feature_category'])['class2_enriched'].sum()/df.groupby(['feature_category'])['class2_enriched'].count()
    prop_df['class2_depleted'] = df.groupby(['feature_category'])['class2_depleted'].sum()/df.groupby(['feature_category'])['class2_depleted'].count()
    prop_df['class3_enriched'] = df.groupby(['feature_category'])['class3_enriched'].sum()/df.groupby(['feature_category'])['class3_enriched'].count()
    prop_df['class3_depleted'] = df.groupby(['feature_category'])['class3_depleted'].sum()/df.groupby(['feature_category'])['class3_depleted'].count()
    
    prop_df['class0_depleted'] = -prop_df['class0_depleted']
    prop_df['class1_depleted'] = -prop_df['class1_depleted']
    prop_df['class2_depleted'] = -prop_df['class2_depleted']
    prop_df['class3_depleted'] = -prop_df['class3_depleted']

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
    
    
if __name__ == '__main__':
    get_SSC_data(ncomp=4)
