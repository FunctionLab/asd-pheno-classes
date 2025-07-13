import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import binomtest, ttest_ind

from utils import split_columns, cohens_d, compute_fold_enrichment, draw_lines_and_stars, get_star_labels


def main_clinical_validation(only_sibs=False):
    # load sample data and class labels
    mixed_data = pd.read_csv(
        '../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', 
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
    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
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
    

def individual_registration_validation(gfmm_labels):
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

    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt' 
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)

    pro_data = pd.merge(
        data, gfmm_labels[['mixed_pred']], left_index=True, right_index=True
        )
    sib_data = pd.merge(data, paired_sibs, left_index=True, right_index=True)

    supp_table = pd.DataFrame()
    class_names = ['Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD with DD']

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

        # expand y-axis limit to add stars
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([0, ymax*1.25])

        classes = [var_data[var_data['mixed_pred'] == i][var].astype(float).to_list() for i in range(4)]
        # Hypothesis testing between groups
        if var == 'language_level_at_enrollment':
            class_pairs = [
                (0, 1),
                (1, 3),
                (2, 0),
                (2, 3),
                (2, 1),
                (0, 3),
            ]
            
            # Compute p-values and fold enrichments
            pvals = [ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue for i, j in class_pairs]
            fe_list = [compute_fold_enrichment(classes[i], classes[j]) for i, j in class_pairs]
            
            # multiple hypothesis correction with Benjamini-Hochberg
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'variable': var, 'group1': [class_names[0], class_names[1], class_names[2], class_names[2], class_names[2], class_names[0]],
                'vs': [class_names[1], class_names[3], class_names[0], class_names[3], class_names[1], class_names[3]],
                'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': fe_list}
                ))

            custom_thresholds = {
                0.01: '***',
                0.05: '**',
                0.1: '*',
                1: 'ns'
            }

            custom_pvalues = list(pvals)[:3]
            star_labels = get_star_labels(custom_pvalues, custom_thresholds)
            pairs = [(0,1), (1, 3), (2, 0)] 
            y_positions = [3, 3.25, 3.5]

            # call the function to draw lines and stars
            draw_lines_and_stars(ax, pairs, y_positions, star_labels, star_size=16)

        elif var == 'cognitive_impairment_at_enrollment':
            pvals = []

            class_pairs = [
                (1, 0),
                (1, 3),
                (2, 0),
                (1, 2),
                (3, 0),
                (3, 2),
            ]
            
            # binomial tests
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
            
            # fold enrichment
            fe_list = [compute_fold_enrichment(classes[i], classes[j]) for i, j in class_pairs]
            
            # multiple hypothesis correction with Benjamini-Hochberg
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'variable': var, 'group1': [class_names[1], class_names[1], class_names[2], class_names[1], class_names[3], class_names[3]],
                'vs': [class_names[0], class_names[3], class_names[0], class_names[2], class_names[0], class_names[2]],
                'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': fe_list}
                ))

            custom_thresholds = {
                0.01: '***',
                0.05: '**',
                0.1: '*',
                1: 'ns'
            }

            custom_pvalues = list(pvals)[:3]
            star_labels = get_star_labels(custom_pvalues, custom_thresholds)
            pairs = [(0,1), (1, 3), (2, 0)]  # Pairs of x indices to connect
            y_positions = [0.36, 0.39, 0.42]  # Y positions for stars

            # Call the function to draw lines and stars
            draw_lines_and_stars(ax, pairs, y_positions, star_labels, star_size=16)

        elif var == 'diagnosis_age':
            class_pairs = [
                (2, 3),
                (2, 0),
                (1, 3),
                (2, 1),
                (0, 3),
                (0, 1),
            ]
            
            # compute p-values and Cohen's d
            pvals = [ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue for i, j in class_pairs]
            cohens_d_list = [cohens_d(classes[i], classes[j]) for i, j in class_pairs]
            
            # multiple hypothesis correction with Benjamini-Hochberg
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'variable': var, 'group1': [class_names[2], class_names[2], class_names[1], class_names[2], class_names[0], class_names[0]],
                'vs': [class_names[3], class_names[0], class_names[3], class_names[1], class_names[3], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': cohens_d_list}
                ))
        
            custom_thresholds = {
                0.01: '***',
                0.05: '**',
                0.1: '*',
                1: 'ns'
            }

            custom_pvalues = list(pvals)[:3]
            star_labels = get_star_labels(custom_pvalues, custom_thresholds)
            pairs = [(2,3), (2,0), (1,3)]  # Pairs of x indices to connect
            y_positions = [220, 232, 250]  # Y positions for stars

            # Call the function to draw lines and stars
            draw_lines_and_stars(ax, pairs, y_positions, star_labels, star_size=16)

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
            dpi=900)
        plt.close()
    
    supp_table.to_csv('../supp_tables/Supp_Table_group_comp_individual_registration_validation.csv', index=False)


def scq_and_developmental_milestones_validation(gfmm_labels, ncomp):
    # get sibling data for background history
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    bhdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv')
    bhdf = bhdf.loc[(bhdf['age_at_eval_years'] <= 18) & (bhdf['age_at_eval_years'] >= 4)]
    
    dev_milestones = [
        'smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 
        'walked_age_mos', 'fed_self_spoon_age_mos', 'used_words_age_mos', 
        'combined_words_age_mos', 'combined_phrases_age_mos', 
        'bladder_trained_age_mos', 'bowel_trained_age_mos'
    ]
    bhdf = bhdf.set_index('subject_sp_id', drop=True)[dev_milestones]

    # subset to paired sibs
    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sib_data = pd.merge(bhdf, paired_sibs, left_index=True, right_index=True)
    sib_bh_data = sib_data[dev_milestones].dropna().astype(float)

    supp_table = pd.DataFrame()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'axes.titlepad': 20})
    fig, axs = plt.subplots(3, 1, figsize=(9, 17))

    # Milestone: used_words_age_mos
    milestone = 'used_words_age_mos'
    milestone_data = gfmm_labels[[milestone, 'mixed_pred']]

    class_names = ['Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD with DD']
    
    # Dynamically get the classes based on ncomp
    classes = [milestone_data[milestone_data['mixed_pred'] == i][milestone].astype(float).to_list() for i in range(ncomp)]

    # Hypothesis testing
    comparisons = []
    pvals = [ttest_ind(cls, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue for cls in classes]
    cohens_d_list = [cohens_d(cls, sib_bh_data[milestone]) for cls in classes]
    comparisons += [('class0', 'siblings'), ('class1', 'siblings'), ('class2', 'siblings'), ('class3', 'siblings')]

    class_pairs = [
        (1, 0),  # class1 > class0
        (0, 2),  # class0 > class2
        (3, 1),  # class3 > class1
        (1, 2),  # class1 > class2
        (3, 0),  # class3 > class0
        (3, 2),  # class3 > class2
    ]
    
    for i, j in class_pairs:
        pvals.append(ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue)
        comparisons.append((f'class{i}', f'class{j}'))
        cohens_d_list.append(cohens_d(classes[i], classes[j]))
    
    # multiple hypothesis correction with Benjamini-Hochberg
    uncorrected_pvals = pvals
    pvals = multipletests(pvals, method='fdr_bh')[1]
    print(milestone)
    print(pvals)
    supp_table = supp_table.append(pd.DataFrame(
        {'milestone': milestone, 'group1': [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], class_names[0], class_names[3], class_names[1], class_names[3], class_names[3]],
        'vs': ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[2], class_names[1], class_names[2], class_names[0], class_names[2]],
        'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': cohens_d_list}
        ))

    class_colors = {
        'dimgray': 'dimgray',
        'class0': '#F85C50',
        'class1': '#6E1E76',
        'class2': '#1CA4B8',
        'class3': '#0073B7',
        'class4': '#8E44AD',
        'class5': '#2C3E50'
    }
    palette = list(class_colors.values())[:ncomp+1]

    # Boxplot
    sns.boxplot(data=[sib_bh_data[milestone].to_list()] + classes, 
                showfliers=True, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
                whiskerprops=dict(color="black", linewidth=2), 
                capprops=dict(color="black", linewidth=2),
                medianprops=dict(color='white', linewidth=2), 
                boxprops=dict(edgecolor='white', linewidth=0.5), ax=axs[0])
    axs[0].set_ylim([0, 90])
    axs[0].set_ylabel('Months', fontsize=20)
    axs[0].set_title('Age first used words', fontsize=22)
    axs[0].set_xticklabels([])
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim([0, ymax*1.25])
    # make bottom and left axes bold and remove top and right axes
    for axis in ['top','bottom','left','right']:
        axs[0].spines[axis].set_linewidth(1.5)
        axs[0].spines[axis].set_color('black')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    y_values = [86, 86, 86, 86]
    for grpidx in range(ncomp):
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx]
        ypos = y_position + 0.007
        if p_value < 0.01:
            axs[0].annotate('***', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.05:
            axs[0].annotate('**', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.1:
            axs[0].annotate('*', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')

    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }

    custom_pvalues = list(pvals)[4:7]
    star_labels = get_star_labels(custom_pvalues, custom_thresholds)
    pairs = [(1, 2), (1, 3), (2, 4)]  # Pairs of x indices to connect
    y_positions = [98, 105, 112]  # Y positions for stars

    # Call the function to draw lines and stars
    draw_lines_and_stars(axs[0], pairs, y_positions, star_labels)

    # Milestone: walked_age_mos
    milestone = 'walked_age_mos'
    milestone_data = gfmm_labels[[milestone, 'mixed_pred']]
    comparisons = []

    # Repeat class assignment for ncomp
    classes = [milestone_data[milestone_data['mixed_pred'] == i][milestone].astype(float).to_list() for i in range(ncomp)]
    pvals = [ttest_ind(cls, sib_bh_data[milestone], equal_var=False, alternative='greater').pvalue for cls in classes]
    cohens_d_list = [cohens_d(cls, sib_bh_data[milestone]) for cls in classes]
    comparisons += (('class0', 'siblings'), ('class1', 'siblings'), ('class2', 'siblings'), ('class3', 'siblings'))

    # compute pvals, cohen's d
    class_pairs = [
        (1, 0),  # class1 > class0
        (2, 0),  # class2 > class0
        (1, 3),  # class1 > class3
        (1, 2),  # class1 > class2
        (3, 0),  # class3 > class0
        (3, 2),  # class3 > class2
    ]
    
    for i, j in class_pairs:
        pvals.append(ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue)
        comparisons.append((f'class{i}', f'class{j}'))
        cohens_d_list.append(cohens_d(classes[i], classes[j]))
    
    # multiple hypothesis correction with Benjamini-Hochberg
    uncorrected_pvals = pvals
    pvals = multipletests(pvals, method='fdr_bh')[1]
    print(milestone)
    print(pvals)
    supp_table = supp_table.append(pd.DataFrame(
        {'milestone': milestone, 'group1': [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], class_names[2], class_names[1], class_names[1], class_names[3], class_names[3]],
        'vs': ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[0], class_names[3], class_names[2], class_names[0], class_names[2]],
        'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': cohens_d_list}
        ))

    sns.boxplot(data=[sib_bh_data[milestone].to_list()] + classes, 
                showfliers=True, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'],
                whiskerprops=dict(color="black", linewidth=2), 
                capprops=dict(color="black", linewidth=2),
                medianprops=dict(color='white', linewidth=2), 
                boxprops=dict(edgecolor='white', linewidth=0.5), ax=axs[1])
    axs[1].set_ylim([0, 45])
    axs[1].set_ylabel('Months', fontsize=20)
    axs[1].set_title('Age first walked', fontsize=22)
    axs[1].set_xticklabels([])
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_ylim([0, ymax*1.25])
    # make bottom and left axes bold and remove top and right axes
    for axis in ['top','bottom','left','right']:
        axs[1].spines[axis].set_linewidth(1.5)
        axs[1].spines[axis].set_color('black')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # plot significance stars
    y_values = [43, 43, 43, 43]
    for grpidx in range(ncomp):
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx]
        ypos = y_position + 0.007
        if p_value < 0.01:
            axs[1].annotate('***', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.05:
            axs[1].annotate('**', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.1:
            axs[1].annotate('*', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')

    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }

    custom_pvalues = list(pvals)[4:7]
    star_labels = get_star_labels(custom_pvalues, custom_thresholds)
    pairs = [(1, 2), (1, 3), (2, 4)]  # Pairs of x indices to connect
    y_positions = [48, 51.5, 55]  # Y positions for stars

    # Call the function to draw lines and stars
    draw_lines_and_stars(axs[1], pairs, y_positions, star_labels)

    # SCQ total score
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv')
    scqdf = scqdf.loc[(scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) & (scqdf['age_at_eval_years'] >= 4)]
    scqdf = scqdf.set_index('subject_sp_id', drop=True).drop(
        ['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id',
         'current_depend_adult', 'age_at_eval_months', 'scq_measure_validity_flag',
         'eval_year', 'missing_values', 'summary_score'], axis=1)
    scqdf = scqdf[scqdf['asd'] == 0]
    
    sib_data = pd.merge(scqdf, paired_sibs, left_index=True, right_index=True)
    sib_scq_data = sib_data['final_score'].dropna().astype(int).to_list()

    final_score = gfmm_labels[['final_score', 'mixed_pred']]
    classes = [final_score[final_score['mixed_pred'] == i]['final_score'].dropna().astype(int).to_list() for i in range(ncomp)]
    comparisons = []

    # compute pvals, cohen's d
    pvals = [ttest_ind(cls, sib_scq_data, equal_var=False, alternative='greater').pvalue for cls in classes]
    cohens_d_list = [cohens_d(cls, sib_scq_data) for cls in classes]
    comparisons += (('siblings', 'class0'), ('siblings', 'class1'), ('siblings', 'class2'), ('siblings', 'class3'))
    class_pairs = [
        (1, 0),  # class1 > class0
        (2, 0),  # class2 > class0
        (1, 3),  # class1 > class3
        (1, 2),  # class1 > class2
        (3, 0),  # class3 > class0
        (3, 2),  # class3 > class2
    ]
    
    # Perform tests and store results
    for i, j in class_pairs:
        pvals.append(ttest_ind(classes[i], classes[j], equal_var=False, alternative='greater').pvalue)
        comparisons.append((f'class{i}', f'class{j}'))
        cohens_d_list.append(cohens_d(classes[i], classes[j]))
    
    # multiple hypothesis correction with Benjamini-Hochberg
    uncorrected_pvals = pvals
    pvals = multipletests(pvals, method='fdr_bh')[1]
    print('SCQ total score')
    print(pvals)
    supp_table = supp_table.append(pd.DataFrame(
        {'milestone': 'SCQ total score', 'group1': [class_names[0], class_names[1], class_names[2], class_names[3], class_names[1], class_names[2], class_names[1], class_names[1], class_names[3], class_names[3]],
        'vs': ['siblings', 'siblings', 'siblings', 'siblings', class_names[0], class_names[0], class_names[3], class_names[2], class_names[0], class_names[2]],
        'p': uncorrected_pvals, 'fdr': pvals, 'Cohen\'s d': cohens_d_list}
        ))

    sns.boxplot(data=[sib_scq_data] + classes, palette=['dimgray','#FBB040','#EE2A7B','#39B54A','#27AAE1'], 
                showfliers=True, whiskerprops=dict(color="black", linewidth=2), 
                capprops=dict(color="black", linewidth=2),
                medianprops=dict(color='white', linewidth=2), 
                boxprops=dict(edgecolor='white', linewidth=0.5), ax=axs[2])
    axs[2].set_ylabel('Total Score', fontsize=20)
    axs[2].set_title('Social Communication Questionnaire', fontsize=22)
    axs[2].set_xticklabels([])
    ymin, ymax = axs[2].get_ylim()
    axs[2].set_ylim([0, ymax*1.25])
    # make bottom and left axes bold and remove top and right axes
    for axis in ['top','bottom','left','right']:
        axs[2].spines[axis].set_linewidth(1.5)
        axs[2].spines[axis].set_color('black')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)

    # plot significance stars
    y_values = [39, 39, 39, 39]
    for grpidx in range(ncomp):
        p_value = pvals[grpidx]
        x_position = grpidx+1
        y_position = y_values[grpidx]
        ypos = y_position + 0.007
        if p_value < 0.01:
            axs[2].annotate('***', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.05:
            axs[2].annotate('**', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')
        elif p_value < 0.1:
            axs[2].annotate('*', xy=(x_position, ypos), 
                           ha='center', size=26, fontweight='bold')

    custom_thresholds = {
        0.01: '***',
        0.05: '**',
        0.1: '*',
        1: 'ns'
    }

    custom_pvalues = list(pvals)[4:7]
    star_labels = get_star_labels(custom_pvalues, custom_thresholds)
    pairs = [(1, 2), (1, 3), (2, 4)]  # Pairs of x indices to connect
    y_positions = [44, 47, 50]  # Y positions for stars

    # Call the function to draw lines and stars
    draw_lines_and_stars(axs[2], pairs, y_positions, star_labels)

    plt.savefig(f'figures/{ncomp}classes_pheno_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close()

    supp_table.to_csv(f'../supp_tables/Supp_Table_{ncomp}classes_pheno_comparisons.csv')


def hypothesis_testing_BMS_features(mixed_data):
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
    sibling_list = '../PhenotypeClasses/data/WES_5392_paired_siblings_sfid.txt'
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

    class_names = ['Siblings', 'Moderate Challenges', 'Broadly Impacted', 'Social/Behavioral', 'Mixed ASD with DD']
    
    # compare each feature between classes
    supp_table = pd.DataFrame()
    for feature in neurodev + mental_health + cooccurring:
        print(feature)
        classes = [mixed_data[mixed_data['mixed_pred'] == i][feature].astype(float) for i in range(-1, 4)]
        
        pvals = []
        fold_changes = []

        # first compare to siblings
        for i in range(1, 5):
            # Count the number of successes in the class i
            successes = int(sum(classes[i]))  # Assuming binary data: 1 for success, 0 for failure
            total = int(len(classes[i]))      # Total trials in the class
            
            # Calculate the expected proportion (from classes[0])
            expected_proportion = sum(classes[0]) / len(classes[0])
            
            # Perform a one-sided binomial test (alternative: greater)
            pval = binomtest(successes, total, expected_proportion, alternative='greater').pvalue
            pvals.append(pval)

        fold_changes += [compute_fold_enrichment(classes[1], classes[0]),
                            compute_fold_enrichment(classes[2], classes[0]),
                            compute_fold_enrichment(classes[3], classes[0]),
                            compute_fold_enrichment(classes[4], classes[0])]

        binom_configs = [
            (1, [2, 3, 4]),
            (2, [1, 3, 4]),
            (3, [1, 2, 4]),
            (4, [1, 2, 3]),
        ]
        
        # run binomial tests (greater and less) and compute fold enrichment
        for test_idx, comp_idxs in binom_configs:
            test_class = classes[test_idx]
            comp_class = np.concatenate([classes[i] for i in comp_idxs])
            
            successes = int(np.sum(test_class))
            total = len(test_class)
            expected_proportion = np.sum(comp_class) / len(comp_class)
            
            # binomial tests (greater and less)
            pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
            pvals.append(binomtest(successes, total, expected_proportion, alternative='less').pvalue)
        
            # fold enrichment (both directions)
            fold_changes.append(compute_fold_enrichment(test_class, comp_class))
            fold_changes.append(compute_fold_enrichment(comp_class, test_class))
        
        # comparison between groups
        if feature in ['growth_macroceph', 'growth_microceph', 'dev_id', 'neuro_sz', 
                        'birth_defect', 'dev_motor', 'feeding_dx']:
            class_pairs = [
                (2, 1),
                (2, 3),
                (2, 4),
                (4, 3),
                (4, 1),
                (3, 1),
            ]
            
            # compute p-values and fold changes
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
                
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
                fold_changes.append(compute_fold_enrichment(classes[i], classes[j]))
            
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'feature': feature, 'group1': [class_names[1], class_names[2], class_names[3], class_names[4], class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], 'All other probands', class_names[2], class_names[2], class_names[2], class_names[4], class_names[4], class_names[3]],
                'vs': ['siblings', 'siblings', 'siblings', 'siblings', 'All other probands', class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], class_names[1], class_names[3], class_names[4], class_names[3], class_names[1], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'fold_change': fold_changes}
                ))
        
        elif feature in ['dev_lang_dis']:
            class_pairs = [
                (2, 1),
                (2, 3),
                (4, 2),
                (4, 3),
                (4, 1),
                (1, 3),
            ]
            
            # compute p-values and fold changes
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
                
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
                fold_changes.append(compute_fold_enrichment(classes[i], classes[j]))
            
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'feature': feature, 'group1': [class_names[1], class_names[2], class_names[3], class_names[4],  class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], 'All other probands', class_names[2], class_names[2], class_names[4], class_names[4], class_names[4], class_names[3]],
                'vs': ['siblings', 'siblings', 'siblings', 'siblings', 'All other probands', class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], class_names[1], class_names[3], class_names[2], class_names[3], class_names[1], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'fold_change': fold_changes}
                ))
        
        elif feature in ['sleep_dx']:
            class_pairs = [
                (2, 1),
                (2, 3),
                (2, 4),
                (3, 4),
                (4, 1),
                (3, 1),
            ]
            
            # compute p-values and fold enrichments
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
            
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
                fold_changes.append(compute_fold_enrichment(classes[i], classes[j]))
            
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'feature': feature, 'group1': [class_names[1], class_names[2], class_names[3], class_names[4], class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], 'All other probands', class_names[2], class_names[2], class_names[2], class_names[3], class_names[4], class_names[3]],
                'vs': ['siblings', 'siblings', 'siblings', 'siblings', 'All other probands', class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], class_names[1], class_names[3], class_names[4], class_names[4], class_names[1], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'fold_change': fold_changes}
                ))

        elif feature in ['behav_adhd', 'mood_dep']:
            class_pairs = [
                (2, 1),
                (3, 2),
                (2, 4),
                (3, 4),
                (1, 4),
                (3, 1),
            ]
            
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
            
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
                fold_changes.append(compute_fold_enrichment(classes[i], classes[j]))
            
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'feature': feature, 'group1': [class_names[1], class_names[2], class_names[3], class_names[4], class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], 'All other probands', class_names[2], class_names[3], class_names[2], class_names[3], class_names[1], class_names[3]],
                'vs': ['siblings', 'siblings', 'siblings', 'siblings', 'All other probands', class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], class_names[1], class_names[2], class_names[4], class_names[4], class_names[4], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'fold_change': fold_changes}
                ))

        elif feature in ['mood_anx', 'mood_ocd', 'tics']:
            class_pairs = [
                (2, 1),
                (2, 3),
                (2, 4),
                (3, 4),
                (1, 4),
                (3, 1),
            ]
            
            for i, j in class_pairs:
                successes = int(sum(classes[i]))
                total = len(classes[i])
                expected_proportion = sum(classes[j]) / len(classes[j])
            
                pvals.append(binomtest(successes, total, expected_proportion, alternative='greater').pvalue)
                fold_changes.append(compute_fold_enrichment(classes[i], classes[j]))
            
            uncorrected_pvals = pvals
            pvals = multipletests(pvals, method='fdr_bh')[1]
            supp_table = supp_table.append(pd.DataFrame(
                {'feature': feature, 'group1': [class_names[1], class_names[2], class_names[3], class_names[4], class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], 'All other probands', class_names[2], class_names[2], class_names[2], class_names[3], class_names[1], class_names[3]],
                'vs': ['siblings', 'siblings', 'siblings', 'siblings', 'All other probands', class_names[1], 'All other probands', class_names[2], 'All other probands', class_names[3], 'All other probands', class_names[4], class_names[1], class_names[3], class_names[4], class_names[4], class_names[4], class_names[1]],
                'p': uncorrected_pvals, 'fdr': pvals, 'fold_change': fold_changes}
                ))

    supp_table.to_csv('../supp_tables/Supp_Table_BMS_enrichments.csv')


if __name__ == "__main__":
    gfmm_labels = pd.read_csv(
        '../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', 
        index_col=0, 
        header=0
        )

    main_clinical_validation(only_sibs=True)
    scq_and_developmental_milestones_validation(gfmm_labels, ncomp=4)
    individual_registration_validation(gfmm_labels)
    hypothesis_testing_BMS_features(gfmm_labels)
