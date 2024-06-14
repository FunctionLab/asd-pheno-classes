import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats

from ../PhenotypeClasses/GFMM import get_feature_enrichments
from ../PhenotypeClasses/utils import split_columns, get_cross_cohort_SPARK_data


def cross_cohort_replication(ncomp):
    #spark_labels, ssc_labels = run_spark_model(ncomp)

    # load labels
    spark_labels = pd.read_csv('data/spark_cross_cohort_labels.csv', index_col=0)
    ssc_labels = pd.read_csv('data/ssc_cross_cohort_labels.csv', index_col=0)

    # get feature enrichments for SPARK
    classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(spark_labels, summarize=True)
    
    features_to_exclude = fold_enrichments.copy() # Fold enrichments + Cohen's d values filtered by significance
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # exclude features based on 3 criteria:
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

    df = df_enriched_depleted.copy()
    df = df.fillna('NaN')
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    
    df = df.loc[~df['feature'].isin(features_to_exclude)] # remove non-contributory features for spark
    
    # annotate each feature with its category
    df['feature_category'] = df['feature'].map(feature_to_category)
    # drop features with no category
    df = df.dropna(subset=['feature_category'])
    df = df.replace('NaN', 1)

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
    
    # flip_rows contains feature names that are reverse-coded and need to be flipped
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

    # get feature enrichments for SSC
    ssc_pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, summary_df, fold_enrichments = get_feature_enrichments(ssc_labels, summarize=True)
    
    summary_df = summary_df.fillna('NaN')
    summary_df = summary_df.replace(np.nan, 1)
    summary_df = summary_df.loc[~summary_df['feature'].isin(features_to_exclude)] # remove non-contributory features in ssc
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

    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))
    plot_df = prop_df.T
    plot_df['cluster'] = np.arange(4)
    polar = plot_df.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    polar.rename(columns={'value': 'ssc_value'}, inplace=True)
    
    spark_df = spark_prop_df.T
    spark_df['cluster'] = np.arange(4)
    spark = spark_df.groupby('cluster').mean().reset_index()
    spark = pd.melt(spark, id_vars=['cluster'])
    spark.rename(columns={'value': 'spark_value'}, inplace=True)

    # merge spark and ssc dataframes
    polar = pd.merge(polar, spark, on=['cluster', 'variable'], how='inner')
    polar.to_csv('data/SSC_replication_table.csv')

    corr_matrix = np.zeros((len(features_to_visualize), len(features_to_visualize)))
    correlations = []
    pvals = []
    for i, feature in enumerate(features_to_visualize):
        corr_matrix[i, i] = pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[0] # get the correlation value
        correlations.append(pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[0])
        pvals.append(pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], polar.loc[polar["variable"] == i, "spark_value"])[1])

    # correction of pvals
    pvals = multipletests(pvals, method='fdr_bh')[1]

    # Set the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    sns.barplot(y=features_to_visualize, x=correlations, width=0.45, color='rosybrown', ax=ax)
    plt.xlabel('Pearson r(SPARK, SSC)', fontsize=20)
    plt.ylabel('')
    plt.title(f'SSC Replication', fontsize=21)
    plt.xlim([0,1])
    ax.set_yticklabels(features_to_visualize, fontsize=15)
    plt.xticks(fontsize=15)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=18)
    plt.savefig('figures/SSC_SPARK_replication_7classes_correlation_barplot.png', bbox_inches='tight', dpi=600)
    plt.close()

    print(f'Correlation values: {correlations}')


def run_spark_model(ncomp):
    # get SPARK data (composite only for CBCL)
    spark_data = get_cross_cohort_SPARK_data()

    # look for common features
    spark_features = set(spark_data.columns)
    ssc_data = generate_ssc_data(impute=False)
    ssc_features = set(ssc_data.columns)
    common_features = spark_features & ssc_features
    print(f'Number of total SPARK features: {len(spark_features)}')
    print(f'Number of total SSC features: {len(ssc_features)}')
    print(f'Number of common features: {len(common_features)}')
    print(f'Common features: {common_features}')
    print(f'dropped features: {spark_features - common_features}')

    # order ssc_data in the same order as spark_data
    common_features = list(common_features)
    spark_data = spark_data[common_features]
    ssc_data = ssc_data[common_features]

    # train model on SPARK data only
    age = spark_data['age_at_eval_years']
    Z_p = spark_data[['sex', 'age_at_eval_years']]

    X = spark_data.drop(['sex', 'age_at_eval_years'], axis=1)
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=ncomp, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200)

    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data, Z_p)
    print('SPARK class breakdown:')
    print(mixed_data['mixed_pred'].value_counts())
    mixed_data.to_csv('ssc_replication_data/spark_cross_cohort_labels.csv')

    # predict on SSC test dataset
    Z_p_ssc = ssc_data[['sex', 'age_at_eval_years']]
    ssc_data = ssc_data.drop(['sex', 'age_at_eval_years'], axis=1)
    continuous_columns_ssc, binary_columns_ssc, categorical_columns_ssc = split_columns(list(ssc_data.columns))

    mixed_data_ssc, mixed_descriptor_ssc = get_mixed_descriptor(
        dataframe=ssc_data,
        continuous=continuous_columns_ssc,
        binary=binary_columns_ssc,
        categorical=categorical_columns_ssc
    )

    mixed_data_ssc['ssc_pred'] = model.predict(mixed_data_ssc, Z_p_ssc)
    print('SSC class breakdown:')
    print(mixed_data_ssc['ssc_pred'].value_counts())
    mixed_data_ssc.to_csv('ssc_replication_data/ssc_cross_cohort_labels.csv')
    
    return mixed_data, mixed_data_ssc


if __name__ == '__main__':
    cross_cohort_replication(ncomp=4)
