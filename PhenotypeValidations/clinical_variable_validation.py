import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import binomtest, ttest_ind

from utils import split_columns


def main_clinical_validation(only_sibs=False):
    # load sample data and class labels
    mixed_data = pd.read_csv(
        'data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', 
        index_col=0, 
        header=0
        )

    # get medical data for validation
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    bmsdf = pd.read_csv(
        f'{BASE_PHENO_DIR}/basic_medical_screening_2022-12-12.csv'
        )
    bms_data = bmsdf.set_index('subject_sp_id',drop=True)
    bms_data = bms_data.replace(np.nan, 0)
    
    # compute 'birth_defect' binary feature
    birth_defect_features = [
        'birth_def_cns', 'birth_def_bone', 'birth_def_fac', 
        'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen'
        ]
    bms_data['birth_defect'] = np.where(
        bms_data[birth_defect_features].sum(axis=1) > 0, 1, 0
        )
    
    # retrieve features of interest for three categories
    neurodev = ['growth_macroceph', 'growth_microceph', 'dev_id', 
                'neuro_sz', 'birth_defect', 'dev_lang_dis']
    mental_health = ['tics', 'mood_ocd',  'mood_dep', 'mood_anx', 'behav_adhd']
    cooccurring = ['feeding_dx', 'sleep_dx', 'dev_motor']
    group = neurodev + mental_health + cooccurring
    subset_validation_data = bms_data[group]

    # labels for diagnostic features
    neuro_labels = ['Macrocephaly', 'Microcephaly', 'ID', 
                    'Seizures/Epilepsy', 'Birth Defect', 'Language Delay']
    mental_health_labels = ['Tics', 'OCD', 'Depression', 'Anxiety', 'ADHD']
    cooccurring_labels = ['Feeding Disorder', 'Sleep Disorder', 'Motor Disorder']
    
    # merge with class labels
    mixed_data = pd.merge(
        mixed_data, subset_validation_data, left_index=True, right_index=True
        )
    
    # retrieve sibling data
    sibs = pd.read_csv(
        'data/spark_siblings_bms_validation.txt', 
        sep='\t', 
        index_col=0
        )
    sibling_list = 'data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(
        sibling_list, sep='\t', header=None, index_col=0
        )
    # subset to paired sibs with BMS information
    sibs = pd.merge(sibs, paired_sibs, left_index=True, right_index=True) 
    sibs['birth_defect'] = np.where(
        sibs[birth_defect_features].sum(axis=1) > 0, 1, 0
        ) # add birth defect feature to sibs
    sibs = sibs[group]
    mixed_data = pd.concat([sibs, mixed_data])
    mixed_data['mixed_pred'] = mixed_data['mixed_pred'].replace(np.nan, -1)

    category_names = ['Neurodevelopmental', 'Mental Health', 'Co-occurring']
    labels = [neuro_labels, mental_health_labels, cooccurring_labels]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(
        3, 1, figsize=(10.5, 12), 
        sharex=True, 
        gridspec_kw={'height_ratios': [1.7, 1.4, 1]}
        )
    for i, name, label, category in zip(
            np.arange(len(category_names)), 
            category_names, 
            labels, 
            [neurodev, mental_health, cooccurring]):
        p_values = get_feature_enrichments_with_sibs(
            mixed_data[category+['mixed_pred']], name, only_sibs
            )
        validation_subset = p_values.loc[:,category+['cluster']]
        validation_subset = pd.melt(validation_subset, id_vars=['cluster'])
        validation_subset['value'] = -np.log10(validation_subset['value'])
        fold_enrichments = get_fold_enrichment(
            mixed_data[category+['mixed_pred']], only_sibs)
        fold_enrichments = pd.melt(fold_enrichments, id_vars=['cluster'])
        validation_subset['Fold Enrichment'] = fold_enrichments['value']
        make_bubble_plot(validation_subset, category, label, name, ax=ax[i])
    
    # save figure
    fig.tight_layout()
    plt.savefig(
        'figures/clinical_bubble_plots_validation.png', 
        bbox_inches='tight', 
        dpi=600
        )
    plt.close()


def make_bubble_plot(validation_subset, category, y_labels, category_name, ax=None):    
    validation_subset['color'] = validation_subset['cluster'].map(
        {0: '#FBB040', 1: '#EE2A7B', 2: '#39B54A', 3: '#27AAE1'})
    validation_subset['marker'] = validation_subset['cluster'].map(
        {0: 'o', 1: 'o', 2: 'o', 3: 'o'})
        
    for i, row in validation_subset.iterrows():
        if row['value'] < -np.log10(0.05):
            ax.scatter(
                row['Fold Enrichment'], 
                row['variable'], 
                s=220, 
                c='white', 
                marker=row['marker'], 
                linewidth=2.5, 
                alpha=0.8, 
                edgecolors=row['color']
                )
        else:
            ax.scatter(
                row['Fold Enrichment'], 
                row['variable'], 
                s=250, 
                c=row['color'], 
                marker=row['marker'], 
                alpha=0.9
                )
    if category_name == 'Co-occurring':
        ax.set_xlabel('Fold Enrichment', fontsize=26)
    ax.set_xticks([x for x in range(1, 20, 2)])
    ax.set_ylabel('')
    ax.tick_params(labelsize=24, axis='y')
    ax.tick_params(labelsize=20, axis='x')
    ax.set_title(f'{category_name}', fontsize=26)
    ax.set_yticks(
        [x for x in range(len(y_labels))], 
        y_labels, fontsize=24)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def get_fold_enrichment(mixed_data, only_sibs=False):
    feature_sig_df_high = pd.DataFrame()
    feature_vector = list()

    sibs = mixed_data[mixed_data['mixed_pred'] == -1]
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  
            total_in_sibs = len(sibs[feature])
            if only_sibs:
                sibs_sum = int(np.sum(sibs[feature])) + 1 # add pseudocount to sibs
            else:
                sibs_sum = int(np.sum(sibs[feature])) 
            total_in_class0 = len(class0[feature])
            class0_sum = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            class1_sum = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            class2_sum = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            class3_sum = int(np.sum(class3[feature]))

            background_all = (sibs_sum+class0_sum+class1_sum+class2_sum+class3_sum)/(total_in_sibs+total_in_class0+total_in_class1+total_in_class2+total_in_class3)
            background_sibs = sibs_sum/total_in_sibs
            if only_sibs:
                background = background_sibs
            else:
                background = background_all
            fold_enrichment_class0 = (class0_sum/total_in_class0) / background
            fold_enrichment_class1 = (class1_sum/total_in_class1) / background
            fold_enrichment_class2 = (class2_sum/total_in_class2) / background
            fold_enrichment_class3 = (class3_sum/total_in_class3) / background

            feature_sig_df_high[feature] = [
                fold_enrichment_class0, 
                fold_enrichment_class1, 
                fold_enrichment_class2, 
                fold_enrichment_class3
                ]
            feature_vector.append(feature)
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]

    return feature_sig_norm_high


def get_feature_enrichments_with_sibs(mixed_data, name, only_sibs=False):
    feature_sig_df_high = pd.DataFrame()
    feature_vector = list()

    sibs = mixed_data[mixed_data['mixed_pred'] == -1]
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2: 
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_sibs = len(sibs[feature])
            subset_sibs = int(np.sum(sibs[feature]))
            background_sibs = subset_sibs/total_in_sibs
            background_all = background_prob/total

            if only_sibs:
                background = background_sibs
            else:
                background = background_all

            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            subset_class3 = int(np.sum(class3[feature]))
            
            sf0 = binomtest(
                subset_class0, n=total_in_class0, p=background, alternative='greater'
                ).pvalue
            sf1 = binomtest(
                subset_class1, n=total_in_class1, p=background, alternative='greater'
                ).pvalue
            sf2 = binomtest(
                subset_class2, n=total_in_class2, p=background, alternative='greater'
                ).pvalue
            sf3 = binomtest(
                subset_class3, n=total_in_class3, p=background, alternative='greater'
                ).pvalue

            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3]
            feature_vector.append(feature)

    # multiple hypothesis correction with Benjamini-Hochberg
    shape = feature_sig_df_high.shape
    feature_sig_df_high = feature_sig_df_high.values.flatten()
    feature_sig_df_high = multipletests(feature_sig_df_high, method='fdr_bh')[1]
    feature_sig_df_high = feature_sig_df_high.reshape(shape)
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]
    return feature_sig_norm_high
    

def individual_registration_validation():
    file = '../SPARK_collection_v9_2022-12-12/individuals_registration_2022-12-12.csv'
    data = pd.read_csv(file, index_col=0)
    vars_for_val = [
        'diagnosis_age', 'cognitive_impairment_at_enrollment', 'language_level_at_enrollment'
        ]
    
    # clean up data
    data['language_level_at_enrollment'] = data[
        'language_level_at_enrollment'].replace(
            'Uses longer sentences of his/her own and is able to tell you something that happened', 3)
    data['language_level_at_enrollment'] = data[
        'language_level_at_enrollment'].replace(
            'Combines 3 words together into short sentences', 2)
    data['language_level_at_enrollment'] = data[
        'language_level_at_enrollment'].replace(
            'Uses single words meaningfully (for example, to request)', 1)
    data['language_level_at_enrollment'] = data[
        'language_level_at_enrollment'].replace(
            'No words/does not speak', 0)

    gfmm_labels = pd.read_csv(
        'data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0
        ) 
    sibling_list = 'data/WES_5392_siblings_spids.txt' 
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)

    pro_data = pd.merge(
        data, gfmm_labels[['mixed_pred']], left_index=True, right_index=True
        )
    sib_data = pd.merge(data, paired_sibs, left_index=True, right_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    ax = ax.ravel()
    variable_names = ['Diagnosis Age', 'Cognitive Impairment', 'Language']
    for i, var in enumerate(vars_for_val):
        if var == 'diagnosis_age':
            fig, ax = plt.subplots(figsize=(3.2, 3))
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
        var_data = pro_data[[var, 'mixed_pred']]

        var_data = var_data.dropna()
        sibling_data = sib_data[[var]]
        sibling_data = sibling_data.dropna()
        sibling_data['mixed_pred'] = 4
        var_data = pd.concat([sibling_data, var_data])
            
        if var == 'diagnosis_age':
            sns.boxplot(x='mixed_pred', y=var, data=var_data, showfliers=True, 
                        palette=['#FBB040','#EE2A7B','#39B54A','#27AAE1'],
                        whiskerprops = dict(color = "black", linewidth=2), 
                        capprops = dict(color = "black", linewidth=2),
                        medianprops=dict(color='white', linewidth=2), 
                        boxprops=dict(edgecolor='white', linewidth=0.5))
        else:
            sns.barplot(x='mixed_pred', y=var, data=var_data, 
                        palette=['#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
                        dodge=False)
        
        ax.set_xlabel('')
        if var == 'diagnosis_age':
            ax.set_ylabel('Months', fontsize=16)
        elif var == 'language_level_at_enrollment':
            ax.set_ylabel('Level', fontsize=16)
        elif var == 'cognitive_impairment_at_enrollment':
            ax.set_ylabel('Proportion', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{variable_names[i]}', fontsize=16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticklabels([])

        plt.tight_layout()
        plt.savefig(
            f'figures/{var}_individual_registration_validation.png', 
            bbox_inches='tight', 
            dpi=600)
        plt.close()


def scq_and_developmental_milestones_validation():
    gfmm_labels = pd.read_csv(
        'data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', 
        index_col=0, 
        header=0
        )
    
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
    sibling_list = 'data/WES_5392_siblings_spids.txt'
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
    
    plt.savefig('figures/Figure1_C_D.png', bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == "__main__":
    scq_and_developmental_milestones_validation()
    individual_registration_validation()
    main_clinical_validation(only_sibs=True)
