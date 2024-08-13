import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor

from utils import split_columns, get_feature_enrichments


def run_mixture_model_on_phenotypes(ncomp=4, summarize=True):
    # load unlabeled individual by feature matrix
    datadf = pd.read_csv(
        'data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0
        ) 
    datadf = datadf.round()
    age = datadf['age_at_eval_years']
    
    # define covariates and training data
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    
    # split columns into continuous, binary, and categorical features
    continuous_columns, binary_columns, categorical_columns = split_columns(
        list(X.columns)
        )

    # get mixed data and descriptor
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # define model
    model = StepMix(
        n_components=ncomp, 
        measurement=mixed_descriptor,
        structural='covariate',
        n_steps=1, 
        n_init=200
        )

    # fit model and predict classes
    model.fit(mixed_data, Z_p)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    mixed_data['age'] = age
    mixed_data.to_csv(
        'data/SPARK_5392_ninit_cohort_GFMM_labeled.csv'
        ) # save labeled data

    # get feature enrichments
    _, _, _, _, df_enriched_depleted, fold_enrichments = get_feature_enrichments(
        mixed_data, 
        summarize=summarize
        )
    
    # plot pie chart of class proportions (supplementary figure)
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax = mixed_data['mixed_pred'].value_counts().plot.pie(
        autopct='%1.0f%%', 
        startangle=90, 
        colors=['#FBB040','#39B54A','#27AAE1','#EE2A7B'], 
        labels=None)
    for patch in ax.patches:
        patch.set_alpha(0.75)
    plt.title('SPARK class proportions', fontsize=24)
    plt.rcParams.update({'font.size': 24})
    plt.setp(ax.texts, size=22)
    plt.ylabel('')
    plt.savefig(
        f'figures/class_breakdown_pie_chart_{ncomp}comp.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()

    # plot age and sex breakdown by class (supplementary figure)
    mixed_data = mixed_data.merge(
        datadf['sex'], left_index=True, right_index=True)
    get_age_sex_distributions_for_classes(mixed_data, ncomp) 
    
    return df_enriched_depleted, fold_enrichments


def generate_summary_table(df_enriched_depleted, fold_enrichments):
    """
    Generate figures to summarize phenotypes enriched and depleted in each class.
    """
    features_to_exclude = fold_enrichments.copy() 
    features_to_exclude['class0'] = features_to_exclude['class0'].abs()
    features_to_exclude['class1'] = features_to_exclude['class1'].abs()
    features_to_exclude['class2'] = features_to_exclude['class2'].abs()
    features_to_exclude['class3'] = features_to_exclude['class3'].abs()
    
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
    
    nan_features = features_to_exclude.loc[
        (features_to_exclude['class0'].isna()) &
        (features_to_exclude['class1'].isna()) &
        (features_to_exclude['class2'].isna()) &
        (features_to_exclude['class3'].isna())
    ]
    
    low_features_continuous = features_to_exclude.loc[
        ~features_to_exclude['feature'].isin(binary_features) &
        (features_to_exclude['class0'] < 0.2) & 
        (features_to_exclude['class1'] < 0.2) &
        (features_to_exclude['class2'] < 0.2) &
        (features_to_exclude['class3'] < 0.2)
    ]
    
    low_features_binary = features_to_exclude.loc[
        features_to_exclude['feature'].isin(binary_features) &
        (features_to_exclude['class0'] < 1.5) & 
        (features_to_exclude['class1'] < 1.5) & 
        (features_to_exclude['class2'] < 1.5) &
        (features_to_exclude['class3'] < 1.5)
    ]
    
    features_to_exclude = pd.concat([nan_features, low_features_continuous, low_features_binary])
    features_to_exclude = features_to_exclude['feature'].unique()

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
    
    # mark features as enriched or depleted based on p-value threshold
    for cls in range(4):
        df[f'class{cls}_enriched'] = df[f'class{cls}_enriched'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
        df[f'class{cls}_depleted'] = df[f'class{cls}_depleted'].astype(float).apply(
            lambda x: 1 if x < 0.05 else 0
        )
    
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
    
    for row in flip_rows: # flip enriched and depleted columns
        for cls in range(4):
            df.loc[df['feature'] == row, [f'class{cls}_enriched', f'class{cls}_depleted']] = \
            df.loc[df['feature'] == row, [f'class{cls}_depleted', f'class{cls}_enriched']].values
    
    prop_df = pd.DataFrame()
    
    # calculate proportion of enriched and depleted features by category
    for cls in range(4):
        prop_df[f'class{cls}_enriched'] = df.groupby(['feature_category'])[f'class{cls}_enriched'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_enriched'].count()
        prop_df[f'class{cls}_depleted'] = df.groupby(['feature_category'])[f'class{cls}_depleted'].sum() / \
                                          df.groupby(['feature_category'])[f'class{cls}_depleted'].count()
        prop_df[f'class{cls}_depleted'] = -prop_df[f'class{cls}_depleted']
        prop_df[f'class{cls}_max'] = prop_df[[f'class{cls}_enriched', f'class{cls}_depleted']].sum(axis=1)
    
    df = prop_df.drop([f'class{cls}_max' for cls in range(4)], axis=1)
    df = df.loc[~df.index.isin(['somatic', 'other problems', 'thought problems'])]

    proportions = pd.DataFrame(index=df.index)
    
    for cls in range(4):
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
        value_vars=[f'class{cls}_enriched' for cls in range(4)] + [f'class{cls}_depleted' for cls in range(4)],
        var_name='class', 
        value_name='proportion'
    )
    
    proportions_melted['type'] = proportions_melted['class'].apply(lambda x: x.split('_')[1])
    proportions_melted['class'] = proportions_melted['class'].apply(lambda x: x.split('_')[0])

    # plot variation figure (supplementary figure)
    fig, ax = plt.subplots(figsize=(12, 5))    
    feature_categories = proportions['feature_category'].unique()
    classes = ['class0', 'class1', 'class2', 'class3']
    bar_width = 0.1
    n_classes = len(classes)
    spacing = 0.2
    group_width = n_classes * bar_width + spacing
    bar_positions = np.arange(len(feature_categories)) * group_width
    
    class_colors = {
        'class0': '#FBB040',
        'class1': '#EE2A7B',
        'class2': '#39B54A',
        'class3': '#27AAE1'
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
    plt.savefig('figures/GFMM_variation_figure.png', dpi=600)
    plt.close()

    # Prepare data for the main horizontal line plot
    prop_df = prop_df.drop(
        [
            'class0_enriched', 'class0_depleted', 'class1_enriched', 
            'class1_depleted', 'class2_enriched', 'class2_depleted', 
            'class3_enriched', 'class3_depleted'
        ], 
        axis=1
    )
    prop_df.columns = ['0', '1', '2', '3']

    features_to_visualize = [
        'anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 
        'restricted/repetitive', 'social/communication', 'developmental'
    ]
    prop_df = prop_df.loc[features_to_visualize]
    prop_df.index = np.arange(len(prop_df))

    # Create the main horizontal line plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    palette = ['#FBB040', '#EE2A7B', '#39B54A', '#27AAE1']

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
    plt.title('Feature enrichment by category', fontsize=24)
    ax.set_ylabel('Proportion and direction', fontsize=18)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')

    plt.savefig(
        'figures/4_phenotype_categories_horizontal_lineplot.png', 
        bbox_inches='tight', dpi=300
    )
    plt.close()


def get_age_sex_distributions_for_classes(mixed_data, ncomp):
    """
    Plot sex and age distributions by class.
    """
    colors = ['#FBB040','#EE2A7B','green','#27AAE1']
    fig, ax = plt.subplots(2,2,figsize=(10,6))
    
    sns.distplot(
        list(mixed_data[mixed_data['mixed_pred'] == 0]['age']), 
        hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3}, 
        label=f'class {0}', color=colors[0], ax=ax[0,0])
    ax[0, 0].set_xlabel('Age', fontsize=14)
    ax[0, 0].set_title('Class 0', fontsize=14)
    ax[0, 0].set_ylabel('Density', fontsize=14)
    sns.distplot(
        list(mixed_data[mixed_data['mixed_pred'] == 1]['age']), 
        hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3}, 
        label=f'class {1}', color=colors[1], ax=ax[0,1])
    ax[0, 1].set_xlabel('Age', fontsize=14)
    ax[0, 1].set_title('Class 1', fontsize=14)
    ax[0, 1].set_ylabel('Density', fontsize=14)
    sns.distplot(
        list(mixed_data[mixed_data['mixed_pred'] == 2]['age']), 
        hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3}, 
        label=f'class {2}', color=colors[2], ax=ax[1,0])
    ax[1, 0].set_xlabel('Age', fontsize=14)
    ax[1, 0].set_title('Class 2', fontsize=14)
    ax[1, 0].set_ylabel('Density', fontsize=14)
    sns.distplot(
        list(mixed_data[mixed_data['mixed_pred'] == 3]['age']), 
        hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3}, 
        label=f'class {3}', color=colors[3], ax=ax[1,1])
    ax[1, 1].set_xlabel('Age', fontsize=14)
    ax[1, 1].set_title('Class 3', fontsize=14) 
    ax[1, 1].set_ylabel('Density', fontsize=14)
    for i in range(2):
        for j in range(2):
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
    for i in range(2):
        for j in range(2):
            ax[i, j].spines['bottom'].set_linewidth(1.5)
            ax[i, j].spines['left'].set_linewidth(1.5)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Age distributions by class', fontsize=16)
    plt.savefig(f'figures/age_distributions_{ncomp}comp.png', 
                bbox_inches='tight', dpi=600)
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(7,4))
    mixed_data['sex'].replace({0: 'Female', 1: 'Male'}, 
                              inplace=True)
    grouped = mixed_data.groupby('mixed_pred')['sex'].value_counts(
        normalize=True).reset_index(name='proportion')
    sns.barplot(
        data=grouped, x='mixed_pred', y='proportion', hue='sex'
        )
    plt.ylabel('Proportion', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.title('Sex distribution by class', fontsize=14)
    plt.legend(loc='upper right', 
               bbox_to_anchor=(1.3, 1), fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    plt.savefig(
        f'figures/sex_distribution_{ncomp}comp.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()
    

if __name__ == '__main__':
    df_enriched_depleted, fold_enrichments = run_mixture_model_on_phenotypes(ncomp=4, summarize=True)
    generate_summary_table(df_enriched_depleted, fold_enrichments)
    