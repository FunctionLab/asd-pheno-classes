import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils import \
    get_cross_cohort_SPARK_data, \
    generate_ssc_data, \
    split_columns, \
    get_feature_enrichments


def cross_cohort_replication(ncomp):
    spark_labels, ssc_labels = run_spark_model(ncomp)

    # load labels
    spark_labels = pd.read_csv(
        'data/spark_cross_cohort_labels.csv', index_col=0
        )
    ssc_labels = pd.read_csv(
        'data/ssc_cross_cohort_labels.csv', index_col=0
        )

    # get feature enrichments for SPARK
    _, _, _, _, df_enriched_depleted, fold_enrichments = get_feature_enrichments(
        spark_labels, summarize=True)
    
    # fetch fold enrichments + Cohen's d values filtered by significance
    features_to_exclude = fold_enrichments.copy() 
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    # exclude features based on 3 criteria:
    # (1) features with no significant enrichments in any class
    # (2) features with all cohen's d values < 0.2 or FE < 1.5
    # get features where all classes is nan
    with open('../PhenotypeClasses/data/binary_columns.pkl', 'rb') as f:
        binary_features = pickle.load(f)
        
    nan_features = features_to_exclude.loc[(features_to_exclude['class0'].isna()) & 
                                            (features_to_exclude['class1'].isna()) & 
                                            (features_to_exclude['class2'].isna()) & 
                                            (features_to_exclude['class3'].isna())]
    low_features_continuous = features_to_exclude.loc[
        ~features_to_exclude['feature'].isin(binary_features)]
    low_features_continuous = features_to_exclude.loc[
        (features_to_exclude['class0'] < 0.2) 
        & (features_to_exclude['class1'] < 0.2) 
        & (features_to_exclude['class2'] < 0.2) 
        & (features_to_exclude['class3'] < 0.2)] 
    low_features_binary = features_to_exclude.loc[
        features_to_exclude['feature'].isin(binary_features)]
    low_features_binary = low_features_binary.loc[
        (low_features_binary['class0'] < 1.5) 
        & (low_features_binary['class1'] < 1.5)
        & (low_features_binary['class2'] < 1.5) 
        & (low_features_binary['class3'] < 1.5)]
    features_to_exclude = pd.concat(
        [nan_features, low_features_continuous, low_features_binary]
        )
    features_to_exclude = features_to_exclude['feature'].unique()

    features_to_category = pd.read_csv(
        'data/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(
        features_to_category['feature'], 
        features_to_category['category']))

    df = df_enriched_depleted.copy()
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    
    # remove non-contributory features for spark
    df = df.loc[~df['feature'].isin(features_to_exclude)]
    
    # annotate each feature with its category
    df['feature_category'] = df['feature'].map(feature_to_category)
    df = df.dropna(subset=['feature_category'])
    df = df.replace(np.nan, 1)

    for cls in range(4):
        df[f'class{cls}_enriched'] = df[f'class{cls}_enriched'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
        df[f'class{cls}_depleted'] = df[f'class{cls}_depleted'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
    
    # flip_rows contains feature names that are reverse-coded
    flip_rows = [
        'q02_conversation', 'q09_expressions_appropriate', 'q19_best_friend', 
        'q20_talk_friendly', 'q21_copy_you', 'q22_point_things', 'q23_gestures_wanted', 
        'q24_nod_head', 'q25_shake_head', 'q26_look_directly', 'q27_smile_back', 
        'q28_things_interested', 'q29_share', 'q30_join_enjoyment', 'q31_comfort', 
        'q32_help_attention', 'q33_range_expressions', 'q34_copy_actions', 
        'q35_make_believe', 'q36_same_age', 'q37_respond_positively', 
        'q38_pay_attention', 'q39_imaginative_games', 'q40_cooperatively_games']
    
    for row in flip_rows:
        for cls in range(4):
            df.loc[df['feature'] == row, [f'class{cls}_enriched', f'class{cls}_depleted']] = \
            df.loc[df['feature'] == row, [f'class{cls}_depleted', f'class{cls}_enriched']].values
    
    # create new dataframe with the proportions of significant features in each category
    prop_df = pd.DataFrame()
    for cls in range(4):
        prop_df[f'class{cls}_enriched'] = df.groupby(['feature_category'])[f'class{cls}_enriched'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_enriched'].count()
        prop_df[f'class{cls}_depleted'] = df.groupby(['feature_category'])[f'class{cls}_depleted'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_depleted'].count()
        prop_df[f'class{cls}_depleted'] = -prop_df[f'class{cls}_depleted']
        prop_df[f'class{cls}_sum'] = prop_df[[f'class{cls}_enriched', f'class{cls}_depleted']].sum(axis=1)
    
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 
                            'class1_depleted', 'class2_enriched', 'class2_depleted', 
                            'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['0', '1', '2', '3']
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 
                             'self-injury', 'social/communication', 
                             'restricted/repetitive', 'developmental'] 
    spark_prop_df = prop_df.loc[features_to_visualize]
    spark_prop_df.index = np.arange(len(spark_prop_df))

    ssc_labels = ssc_labels.rename({'ssc_pred': 'mixed_pred'}, axis=1)
    # get feature enrichments for SSC
    _, _, _, _, summary_df, fold_enrichments = get_feature_enrichments(
        ssc_labels, summarize=True)
    
    summary_df = summary_df.replace(np.nan, 1)
    summary_df = summary_df.loc[~summary_df['feature'].isin(
        features_to_exclude)] # remove non-contributory features
    
    summary_df['feature_category'] = summary_df['feature'].map(
        feature_to_category)
    summary_df = summary_df.dropna(subset=['feature_category'])

    for cls in range(4):
        summary_df[f'class{cls}_enriched'] = summary_df[f'class{cls}_enriched'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
        summary_df[f'class{cls}_depleted'] = summary_df[f'class{cls}_depleted'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
    
    # create new dataframe with the proportions of significant features 
    # in each category
    prop_df = pd.DataFrame()
    for cls in range(4):
        prop_df[f'class{cls}_enriched'] = summary_df.groupby(['feature_category'])[f'class{cls}_enriched'].sum() / \
                                          summary_df.groupby(['feature_category'])[f'class{cls}_enriched'].count()
        prop_df[f'class{cls}_depleted'] = summary_df.groupby(['feature_category'])[f'class{cls}_depleted'].sum() / \
                                          summary_df.groupby(['feature_category'])[f'class{cls}_depleted'].count()
        prop_df[f'class{cls}_depleted'] = -prop_df[f'class{cls}_depleted']
        prop_df[f'class{cls}_sum'] = prop_df[[f'class{cls}_enriched', f'class{cls}_depleted']].sum(axis=1)
    
    # drop the enriched and depleted columns
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 
                            'class1_depleted', 'class2_enriched', 'class2_depleted', 
                            'class3_enriched', 'class3_depleted'], axis=1)
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

    correlations = []
    pvals = []
    for i, feature in enumerate(features_to_visualize):
        correlations.append(stats.pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], \
                                           polar.loc[polar["variable"] == i, "spark_value"])[0])
        pvals.append(stats.pearsonr(polar.loc[polar["variable"] == i, "ssc_value"], \
                                    polar.loc[polar["variable"] == i, "spark_value"])[1])

    # correction of pvals
    pvals = multipletests(pvals, method='fdr_bh')[1]

    # Set the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    sns.barplot(y=features_to_visualize, x=correlations, 
                width=0.45, color='rosybrown', ax=ax)
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
    plt.savefig(
        'figures/SSC_SPARK_replication_7classes_correlation_barplot.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()

    print(f'Correlation values: {correlations}')


def run_spark_model(ncomp):
    # get SPARK data (composite-only CBCL)
    spark_data = get_cross_cohort_SPARK_data()

    # look for common features
    spark_features = set(spark_data.columns)
    ssc_data = generate_ssc_data()
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

    # train model on SPARK data
    Z_p = spark_data[['sex', 'age_at_eval_years']]
    X = spark_data.drop(['sex', 'age_at_eval_years'], axis=1)
    
    continuous_columns, binary_columns, categorical_columns = split_columns(
        list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=ncomp, 
                    measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, 
                    n_init=200
                   )

    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data, Z_p)
    print('SPARK class breakdown:')
    print(mixed_data['mixed_pred'].value_counts())
    mixed_data.to_csv('data/spark_cross_cohort_labels.csv')

    # predict on SSC test dataset
    Z_p_ssc = ssc_data[['sex', 'age_at_eval_years']]
    ssc_data = ssc_data.drop(['sex', 'age_at_eval_years'], axis=1)
    continuous_columns_ssc, binary_columns_ssc, categorical_columns_ssc = split_columns(
        list(ssc_data.columns))

    mixed_data_ssc, mixed_descriptor_ssc = get_mixed_descriptor(
        dataframe=ssc_data,
        continuous=continuous_columns_ssc,
        binary=binary_columns_ssc,
        categorical=categorical_columns_ssc
    )

    mixed_data_ssc['ssc_pred'] = model.predict(mixed_data_ssc, Z_p_ssc)
    print('SSC class breakdown:')
    print(mixed_data_ssc['ssc_pred'].value_counts())
    mixed_data_ssc.to_csv('data/ssc_cross_cohort_labels.csv')
    
    return mixed_data, mixed_data_ssc


if __name__ == '__main__':
    cross_cohort_replication(ncomp=4)
