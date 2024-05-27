import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from scipy import stats
from scipy.stats import binomtest, pearsonr

from utils import split_columns, get_feature_enrichments, cohens_d, adjust_pvalues


def run_mixture_model_on_phenotypes(ncomp=4, summarize=False):
    datadf = pd.read_csv('data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)
    datadf = datadf.round()
    age = datadf['age_at_eval_years']

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
                    n_steps=1, n_init=200)

    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    mixed_data['age'] = age
    mixed_data.to_csv('data/SPARK_5392_ninit_cohort_GFMM_labeled.csv')
    
    # get feature enrichments
    if summarize:
        classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments = get_feature_enrichments(mixed_data, summarize=True)
    else:
        classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector = get_feature_enrichments(mixed_data)
    
    # pie chart of class proportions
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax = mixed_data['mixed_pred'].value_counts().plot.pie(autopct='%1.0f%%', startangle=90, colors=['violet','limegreen','blue','red'], labels=None)
    for patch in ax.patches:
        patch.set_alpha(0.75)
    plt.title('SPARK class proportions', fontsize=24)
    plt.rcParams.update({'font.size': 24})
    plt.setp(ax.texts, size=22)
    plt.ylabel('')
    plt.savefig(f'figures/GFMM_pie_chart_5392_{ncomp}comp.png', bbox_inches='tight')
    plt.close()


def generate_summary_table(df_enriched_depleted, fold_enrichments):
    features_to_exclude = fold_enrichments.copy() # Fold enrichments + Cohen's d values filtered by significance
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    
    # exclusion criteria:
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

    # read in feature_to_category mapping
    features_to_category = pd.read_csv('../PhenotypeValidation/data/feature_to_category_mapping.csv', index_col=None)
    feature_to_category = dict(zip(features_to_category['feature'], features_to_category['category']))
    df = df_enriched_depleted.copy()
    df = df.fillna('NaN')
    if 'feature category' in df.columns:
        df = df.drop('feature category', axis=1)
    df = df.loc[~df['feature'].isin(features_to_exclude)]
    df['feature_category'] = df['feature'].map(feature_to_category)
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
    
    df = prop_df.drop(['class0_max', 'class1_max', 'class2_max', 'class3_max'], axis=1)
    df = df[df.index != 'somatic']
    df = df[df.index != 'other problems']
    df = df[df.index != 'thought problems']

    proportions = pd.DataFrame(index=df.index)
    for i in range(4):
        proportions[f'class{i}_enriched'] = df[f'class{i}_enriched']
        proportions[f'class{i}_depleted'] = df[f'class{i}_depleted']
    
    proportions = proportions.reset_index()
    proportions = proportions.set_index('feature_category').reindex(['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental']).reset_index()
    proportions_melted = pd.melt(proportions, id_vars=['feature_category'], value_vars=[f'class{i}_enriched' for i in range(4)] + [f'class{i}_depleted' for i in range(4)],
                                var_name='class', value_name='proportion')
    proportions_melted['type'] = proportions_melted['class'].apply(lambda x: x.split('_')[1])
    proportions_melted['class'] = proportions_melted['class'].apply(lambda x: x.split('_')[0])

    # plot variation figure
    fig, ax = plt.subplots(figsize=(12, 5))
    feature_categories = proportions['feature_category'].unique()
    classes = ['class0', 'class1', 'class2', 'class3']
    bar_width = 0.1
    n_classes = len(classes)
    spacing = 0.2
    group_width = n_classes * bar_width + spacing
    bar_positions = np.arange(len(feature_categories)) * group_width
    class_colors = {
        'class0': ['pink', 'violet'],
        'class1': ['lightcoral', 'red'],
        'class2': ['palegreen', 'limegreen'],
        'class3': ['lightblue', 'blue']
    }
    for idx, cls in enumerate(classes):
        enriched = proportions_melted[(proportions_melted['class'] == cls) & (proportions_melted['type'] == 'enriched')]
        depleted = proportions_melted[(proportions_melted['class'] == cls) & (proportions_melted['type'] == 'depleted')]
        ax.bar(bar_positions + idx * bar_width, depleted['proportion'], width=bar_width, label=f'{cls} depleted', linewidth=0.5, edgecolor='black', color=class_colors[cls][0])
        ax.bar(bar_positions + idx * bar_width, enriched['proportion'], width=bar_width, label=f'{cls} enriched', linewidth=0.5, edgecolor='black', color=class_colors[cls][1])
    ax.set_xticks(bar_positions + bar_width * 1.5)
    ax.set_xticklabels(['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'limited social/communication', 'restricted/repetitive', 'developmental delay'], rotation=30, ha='right')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('figures/GFMM_variation_summary_figure.png')
    
    # Horizontally plot the phenotype categories, have one line per class, y-axis = proportion of significant features
    prop_df = prop_df.drop(['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted'], axis=1)
    prop_df.columns = ['lowASD/lowDelay', 'highASD/highDelay', 'highASD/lowDelay', 'lowASD/highDelay']
    features_to_visualize = features_to_visualize[:-1]
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))    
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    palette = ['violet','red','limegreen','blue']
    ax = sns.lineplot(data=prop_df, dashes=False, markers=True, palette=palette, linewidth=3)    
    ax.set(xlabel="Phenotype Category", ylabel="")
    plt.xticks(ha='right', rotation=30, fontsize=16)
    plt.xticks(np.arange(len(features_to_visualize)), ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'limited social/communication', 'restricted/repetitive', 'developmental delay'])
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
    plt.savefig('figures/GFMM_4_pheno_categories_lineplot.png', bbox_inches='tight', dpi=300)
    plt.close()
        

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
    fig.tight_layout()
    plt.savefig('figures/GFMM_4class_age_density.png', bbox_inches='tight')
    plt.close()

    # sex breakdown by class
    fig, ax = plt.subplots(1,1,figsize=(7,4))
    mixed_data['sex'].replace({0: 'Female', 1: 'Male'}, inplace=True)
    grouped = mixed_data.groupby('mixed_pred')['sex'].value_counts(normalize=True).reset_index(name='proportion')
    sns.barplot(data=grouped, x='mixed_pred', y='proportion', hue='sex')
    plt.ylabel('Proportion', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.title('Sex distribution by class', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14)
    plt.savefig('figures/GFMM_4class_sex_distribution_normalized.png', bbox_inches='tight')
    plt.close()
    

if __name__ == '__main__':
    df_enriched_depleted, fold_enrichments = run_mixture_model_on_phenotypes(ncomp=4, summarize=True)
    generate_summary_table(df_enriched_depleted, fold_enrichments)
    