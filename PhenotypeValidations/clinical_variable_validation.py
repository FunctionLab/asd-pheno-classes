import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats
import pickle as rick
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom
from latent_class_analysis import split_columns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binomtest, ttest_ind


def main_clinical_validation(only_sibs=False):
    mixed_data = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)

    # retrieve BMS data
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    bmsdf = pd.read_csv(f'{BASE_PHENO_DIR}/basic_medical_screening_2022-12-12.csv')
    bms_data = bmsdf.set_index('subject_sp_id',drop=True)
    bms_data = bms_data.replace(np.nan, 0)
    
    # construct birth defects feature    
    birth_defect_features = ['birth_def_cns', 'birth_def_bone', 'birth_def_fac', 'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen']
    bms_data['birth_defect'] = np.where(bms_data[birth_defect_features].sum(axis=1) > 0, 1, 0)
    
    neurodev = ['growth_macroceph', 'growth_microceph', 'dev_id', 'tics','neuro_sz', 'birth_defect']
    maternal = ['birth_ivh', 'birth_pg_inf', 'birth_prem']
    mental_health = ['mood_ocd',  'mood_dep', 'mood_anx', 'behav_adhd']
    daily_living = ['feeding_dx', 'sleep_dx', 'dev_motor', 'dev_lang_dis']
    group = neurodev+maternal+mental_health+daily_living
    subset_validation_data = bms_data[group] # get data for validation features
    mixed_data = pd.merge(mixed_data, subset_validation_data, left_index=True, right_index=True)

    # get labels for plotting
    neuro_labels = ['Macrocephaly', 'Microcephaly', 'ID', 'Tics', 'Seizures/Epilepsy', 'Birth Defect']
    maternal_labels = ['IVH', 'Prenatal Infection', 'Premature Birth']
    mental_health_labels = ['OCD', 'Depression', 'Anxiety', 'ADHD']
    daily_living_labels = ['Feeding Disorder', 'Sleep Disorder', 'Motor Disorder', 'Language Delay']
    
    sibs = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/spark_siblings_bms_validation.txt', sep='\t', index_col=0)
    sibling_list = 'asd-pheno-classes/PhenotypeClasses/data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sibs = pd.merge(sibs, paired_sibs, left_index=True, right_index=True) # subset to 1293 paired siblings who have BMS information
    sibs['birth_defect'] = np.where(sibs[birth_defect_features].sum(axis=1) > 0, 1, 0) # add birth defect feature to sibs
    sibs = sibs[group]
    mixed_data = pd.concat([sibs, mixed_data])
    mixed_data['mixed_pred'] = mixed_data['mixed_pred'].replace(np.nan, -1)

    category_names = ['Neurodevelopmental', 'Mental Health', 'Daily Living'] 
    labels = [neuro_labels, mental_health_labels, daily_living_labels]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(3, 1, figsize=(10.5, 12), sharex=True)
    for i, name, label, category in zip(np.arange(len(category_names)), category_names, labels, [neurodev, mental_health, daily_living]): # maternal
        p_values = get_feature_enrichments_with_sibs(mixed_data[category+['mixed_pred']], name, only_sibs)
        validation_subset = p_values.loc[:,category+['cluster']]
        validation_subset = pd.melt(validation_subset, id_vars=['cluster'])
        validation_subset['value'] = -np.log10(validation_subset['value'])
        fold_enrichments = get_fold_enrichment(mixed_data[category+['mixed_pred']], only_sibs)
        fold_enrichments = pd.melt(fold_enrichments, id_vars=['cluster'])
        validation_subset['Fold Enrichment'] = fold_enrichments['value']
        make_bubble_plot(validation_subset, category, label, name, ax=ax[i])
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.12, wspace=0.12)
    plt.savefig('figures/GFMM_clinical_bubble_plots_validation.png', bbox_inches='tight')
    plt.close()


def make_bubble_plot(validation_subset, category, y_labels, category_name, enrichment=True, ax=None):
    
    validation_subset = validation_subset[validation_subset['cluster'] != -1]
    colors = ['violet', 'red', 'limegreen', 'blue']
    validation_subset['color'] = validation_subset['cluster'].map({0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
    validation_subset['Cluster'] = validation_subset['cluster'].map({0: 'ASD-Lower Support Needs', 1: 'ASD-Higher Support Needs', 2: 'ASD-Social/RRB', 3: 'ASD-Developmentally Delayed'})
    markers = ['o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    
    if enrichment:
        for i, row in validation_subset.iterrows():
            if row['value'] < -np.log10(0.05):
                ax.scatter(row['Fold Enrichment'], row['variable'], s=250, c='white', marker=row['marker'], linewidth=2.5, alpha=0.8, edgecolors=row['color'])
            else:
                ax.scatter(row['Fold Enrichment'], row['variable'], s=250, c=row['color'], marker=row['marker'], alpha=0.85)
        if category_name == 'Daily Living':
            ax.set_xlabel('Fold Enrichment', fontsize=26)
        elif category_name == 'Maternal':
            ax.set_xlabel('Fold Enrichment', fontsize=24)
        ax.set_ylabel('')
        ax.tick_params(labelsize=20, axis='y')
        ax.tick_params(labelsize=20, axis='x')
        ax.set_title(f'{category_name}', fontsize=26)
        ax.set_yticks([x for x in range(len(y_labels))], y_labels, fontsize=20)
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(9,4.5))
        sns.scatterplot(data=validation_subset, x='value', y='variable', size='Fold Enrichment', hue='Cluster', palette=colors, markers=markers, sizes=(50, 500))
        for i, row in validation_subset.iterrows():
            if row['cluster'] == -1:
                ax.scatter(row['value'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
        plt.xlabel('-log10(q-value)', fontsize=20)
        plt.ylabel('')
        plt.yticks(fontsize=20)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = ['Siblings', 'ASD-Lower Support Needs', 'ASD-Higher Support Needs', 'ASD-Social/RRB', 'ASD-Developmentally Delayed']
        handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
        plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'{category_name}', fontsize=24)
        plt.yticks([x for x in range(len(y_labels))], y_labels, fontsize=20)
        plt.axvline(x=-np.log10(0.05), color='gray', linestyle='--', linewidth=1.4)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color('black')
        plt.savefig(f'figures/GFMM_clinical_bubble_plot_{category_name}_validation.png', bbox_inches='tight')
        plt.close()


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
                sibs_sum = int(np.sum(sibs[feature]))+1 # add pseudocount
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
            fold_enrichment_sib = (sibs_sum/total_in_sibs) / background_all

            feature_sig_df_high[feature] = [fold_enrichment_sib, fold_enrichment_class0, fold_enrichment_class1, fold_enrichment_class2, fold_enrichment_class3]
            feature_vector.append(feature)
    
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [-1, 0, 1, 2, 3]

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
            
            sfsib = binomtest(subset_sibs, n=total_in_sibs, p=background_all, alternative='greater').pvalue
            sf0 = binomtest(subset_class0, n=total_in_class0, p=background, alternative='greater').pvalue
            sf1 = binomtest(subset_class1, n=total_in_class1, p=background, alternative='greater').pvalue
            sf2 = binomtest(subset_class2, n=total_in_class2, p=background, alternative='greater').pvalue
            sf3 = binomtest(subset_class3, n=total_in_class3, p=background, alternative='greater').pvalue

            # if any p-value is 0, change p-value to epsilon
            if only_sibs:
                if sfsib == 0:
                    if name == 'Maternal':
                        sfsib = 1e-2
                if sf0 == 0:
                    if name == 'Maternal':
                        sf0 = 1e-25
                if sf1 == 0:
                    if name == 'Maternal':
                        sf1 = 1e-25
                    elif name == 'Daily Living':
                        sf1 = 1e-240
                if sf2 == 0:
                    if name == 'Maternal':
                        sf2 = 1e-25
                    elif name == 'Mental Health':
                        sf2 = 1e-160
                if sf3 == 0:
                    if name == 'Maternal':
                        sf3 = 1e-25
                    elif name == 'Daily Living':
                        sf3 = 1e-250
            
            feature_sig_df_high[feature] = [sfsib, sf0, sf1, sf2, sf3]
            feature_sig_df_high[feature] = multipletests(feature_sig_df_high[feature], method='fdr_bh', alpha=0.05)[1]
            feature_vector.append(feature)

    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [-1, 0, 1, 2, 3]
    return feature_sig_norm_high


def vineland_validation():
    mixed_data = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    
    vineland = pd.read_csv('data/spark_data_vineland_validation.txt', sep='\t', index_col=0)
    motor = pd.read_csv('data/spark_data_motor_validation.txt', sep='\t', index_col=0)
    vineland = vineland.dropna()
    motor = motor.dropna()

    mixed_data = pd.merge(mixed_data, vineland, left_index=True, right_index=True)
    mixed_data = pd.merge(mixed_data, motor, left_index=True, right_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))
    data = mixed_data[['abc_standard', 'dls_standard', 'communication_standard', 'soc_standard', 'motor_standard', 'mixed_pred']]
    data = pd.melt(data, id_vars=['mixed_pred'])
    sns.boxplot(x='variable', y='value', hue='mixed_pred', data=data, palette=['violet','red','limegreen','blue'], showfliers=False) # whiskerprops = dict(color = "white"), capprops = dict(color = "white"),
    plt.xlabel('')
    plt.ylabel('Vineland Scores', fontsize=24)
    plt.xticks([0,1,2,3,4], ['ABC', 'DLS', 'Comm', 'Soc', 'Motor'], fontsize=24)
    handles, labels = plt.gca().get_legend_handles_labels()
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    labels = ['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    plt.legend(handles, labels, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('figures/GFMM_motor_vineland_validation_boxplot.png', bbox_inches='tight')   
    plt.close()
    

def individual_registration_validation():
    file = '../SPARK_collection_v9_2022-12-12/individuals_registration_2022-12-12.csv'
    data = pd.read_csv(file, index_col=0)
    vars_for_val = ['num_asd_parents', 'num_asd_siblings', 'diagnosis_age', 'iep_asd', 'cognitive_impairment_at_enrollment', 'language_level_at_enrollment'] #
    
    data['num_asd_parents'] = data['num_asd_parents'].replace(999, np.nan)
    data['num_asd_siblings'] = data['num_asd_siblings'].replace(999, np.nan)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Uses longer sentences of his/her own and is able to tell you something that happened', 3)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Combines 3 words together into short sentences', 2)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Uses single words meaningfully (for example, to request)', 1)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('No words/does not speak', 0)

    gfmm_labels = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    sibling_list = 'asd-pheno-classes/PhenotypeClasses/data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)

    pro_data = pd.merge(data, gfmm_labels[['mixed_pred']], left_index=True, right_index=True)
    sib_data = pd.merge(data, paired_sibs, left_index=True, right_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(3, 2, figsize=(7, 10))
    ax = ax.ravel()
    variable_names = ['Number of ASD Parents', 'Number of ASD Siblings', 'Diagnosis Age', 'IEP for ASD', 'Cognitive Impairment', 'Language Level']
    for i, var in enumerate(vars_for_val):
        var_data = pro_data[[var, 'mixed_pred']]
        var_data = var_data.dropna()
        sibling_data = sib_data[[var]]
        sibling_data = sibling_data.dropna()
        sibling_data['mixed_pred'] = 4
        var_data = pd.concat([sibling_data, var_data])
        print(var_data['mixed_pred'].value_counts())

        num_unique_vals = len(var_data[var].unique())
        if num_unique_vals == 2:
            group0 = var_data[var_data['mixed_pred'] == 0][var]
            group1 = var_data[var_data['mixed_pred'] == 1][var]
            group2 = var_data[var_data['mixed_pred'] == 2][var]
            group3 = var_data[var_data['mixed_pred'] == 3][var]
            pvals = []
            pvals.append(binomtest(np.sum(group0), n=len(group0), p=np.sum(group3)/len(group3)).pvalue)
            pvals.append(binomtest(np.sum(group1), n=len(group1), p=np.sum(group3)/len(group3)).pvalue)
            pvals.append(binomtest(np.sum(group2), n=len(group2), p=np.sum(group3)/len(group3)).pvalue)
            pvals = multipletests(pvals, method='fdr_bh')[1]
        else:
            group0 = var_data[var_data['mixed_pred'] == 0][var]
            group1 = var_data[var_data['mixed_pred'] == 1][var]
            group2 = var_data[var_data['mixed_pred'] == 2][var]
            group3 = var_data[var_data['mixed_pred'] == 3][var]
            pvals = []
            pvals.append(ttest_ind(group0, group3, equal_var=False, alternative='greater').pvalue)
            pvals.append(ttest_ind(group1, group3, equal_var=False, alternative='greater').pvalue)
            pvals.append(ttest_ind(group2, group3, equal_var=False, alternative='greater').pvalue)
            pvals = multipletests(pvals, method='fdr_bh')[1]
            
        sns.barplot(x='mixed_pred', y=var, data=var_data, ax=ax[i], palette=['violet','red','limegreen','blue','dimgray'], linewidth = 1.5, edgecolor='black', alpha=0.85, dodge=False)
        
        if var in ['num_asd_parents', 'num_asd_siblings']:
            continue
        
        ax[i].set_xlabel('')
        ax[i].set_ylabel('Age (months)', fontsize=18)
        ax[i].set_title(f'{variable_names[i]}', fontsize=20)
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(1)
            ax[i].spines[axis].set_color('black')
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig('figures/GFMM_individual_registration_validation.png', bbox_inches='tight')
    plt.close()

    # plot age_at_registration_years
    var_data = pro_data[['age_at_registration_years', 'mixed_pred']]
    var_data = var_data.dropna()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(x='mixed_pred', y='age_at_registration_years', data=var_data, palette=['violet','red','green','blue'], linewidth = 1.5, edgecolor='black', alpha=0.95, dodge=False)
    ax.set_xlabel('')
    ax.set_ylabel('Age (years)', fontsize=16)
    ax.set_title(f'Age at registration', fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    plt.tight_layout()
    plt.savefig('figures/GFMM_age_at_registration.png', bbox_inches='tight')
    plt.close()


def scq_validation():
    gfmm_labels = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv')
    scqdf = scqdf.loc[(scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) & (scqdf['age_at_eval_years'] >= 4)]
    scqdf = scqdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','current_depend_adult','age_at_eval_months','scq_measure_validity_flag','eval_year','missing_values','summary_score'],axis=1)
    scqdf = scqdf[scqdf['asd'] == 0]
    
    sibling_list = 'asd-pheno-classes/PhenotypeClasses/data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sib_data = pd.merge(scqdf, paired_sibs, left_index=True, right_index=True)
    sib_scq_data = sib_data['final_score'].dropna().astype(int).to_list()

    final_score = gfmm_labels[['final_score', 'mixed_pred']]
    all_proband_scq_data = final_score['final_score'].dropna().astype(int).to_list()
    class0 = final_score[final_score['mixed_pred'] == 0]['final_score'].dropna().astype(int).to_list()
    class1 = final_score[final_score['mixed_pred'] == 1]['final_score'].dropna().astype(int).to_list()
    class2 = final_score[final_score['mixed_pred'] == 2]['final_score'].dropna().astype(int).to_list()
    class3 = final_score[final_score['mixed_pred'] == 3]['final_score'].dropna().astype(int).to_list()

    p_vals = []
    print(f"SCQ: {stats.ttest_ind(all_proband_scq_data, sib_scq_data, equal_var=False, alternative='greater').pvalue}")
    print(f"High-ASD/High-Delays: {stats.ttest_ind(class0, sib_scq_data, equal_var=False, alternative='greater').pvalue}")
    p_vals.append(stats.ttest_ind(class0, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    print(f"Low-ASD/Low-Delays: {stats.ttest_ind(class1, sib_scq_data, equal_var=False, alternative='greater').pvalue}")
    p_vals.append(stats.ttest_ind(class1, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    print(f"High-ASD/Low-Delays: {stats.ttest_ind(class2, sib_scq_data, equal_var=False, alternative='greater').pvalue}")
    p_vals.append(stats.ttest_ind(class2, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    print(f"Low-ASD/High-Delays: {stats.ttest_ind(class3, sib_scq_data, equal_var=False, alternative='greater').pvalue}")
    p_vals.append(stats.ttest_ind(class3, sib_scq_data, equal_var=False, alternative='greater').pvalue)
    print(p_vals)
    p_vals = multipletests(p_vals, method='fdr_bh')[1]
    print(p_vals)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    data = [sib_scq_data, all_proband_scq_data]
    ax = sns.boxplot(data=data, showfliers=False, palette=['dimgray','purple'])
    plt.xlabel('')
    plt.ylabel('SCQ Final Score', fontsize=20)
    plt.xticks([0,1], ['Siblings', 'Probands'], fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'Probands']
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1)
        plt.gca().spines[axis].set_color('black')
    plt.legend(handles, labels, fontsize=16)
    plt.text(0.5, 0.9, f'***', ha='center', va='center', transform=ax.transAxes, fontsize=20)
    plt.savefig('figures/GFMM_scq_validation_all_pros_vs_sibs.png', bbox_inches='tight')
    plt.close()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7.5, 5.5))
    data = [sib_scq_data, class0, class1, class2, class3]
    sns.boxplot(data=data, palette=['dimgray','violet','red','limegreen','blue'], showfliers=False, whiskerprops = dict(color = "black", linewidth=1.5), capprops = dict(color = "black"),
                medianprops=dict(color='black', linewidth=1.5), boxprops=dict(edgecolor='black', linewidth=1.5))
    sns.stripplot(data=data, palette=['dimgray','violet','red','limegreen','blue'], alpha=0.1)
    plt.xlabel('')
    plt.ylabel('SCQ Final Score', fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    plt.yticks(fontsize=16)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.legend(handles, labels, fontsize=16)
    plt.savefig('figures/GFMM_scq_validation_classes_vs_sibs.png', bbox_inches='tight')
    plt.close()


def developmental_milestones_validation():
    gfmm_labels = pd.read_csv('asd-pheno-classes/PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    
    BASE_PHENO_DIR = '../SPARK_collection_v9_2022-12-12'
    bhdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv')
    bhdf = bhdf.loc[(bhdf['age_at_eval_years'] <= 18) & (bhdf['age_at_eval_years'] >= 4)]
    dev_milestones = ['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
                                                        'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos', 'combined_phrases_age_mos',
                                                        'bladder_trained_age_mos', 'bowel_trained_age_mos']
    bhdf = bhdf.set_index('subject_sp_id',drop=True)[dev_milestones]

    sibling_list = 'asd-pheno-classes/PhenotypeClasses/data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sib_data = pd.merge(bhdf, paired_sibs, left_index=True, right_index=True)
    sib_bh_data = sib_data[dev_milestones].dropna().astype(float)

    #sib_bh_data = sib_bh_data.replace(888,0)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 5, figsize=(18.5, 8))

    comprehensive_score_class0 = []
    comprehensive_score_class1 = []
    comprehensive_score_class2 = []
    comprehensive_score_class3 = []
    comprehensive_score_sib = []

    for i, milestone in enumerate(dev_milestones):
        milestone_data = gfmm_labels[[milestone, 'mixed_pred']]
        milestone_data[milestone] = milestone_data[milestone].apply(lambda x: 216 if x > 216 else x) 
        all_proband_bh_data = milestone_data[milestone].astype(float).to_list()
        class0 = milestone_data[milestone_data['mixed_pred'] == 0][milestone].astype(float).to_list()
        class1 = milestone_data[milestone_data['mixed_pred'] == 1][milestone].astype(float).to_list()
        class2 = milestone_data[milestone_data['mixed_pred'] == 2][milestone].astype(float).to_list()
        class3 = milestone_data[milestone_data['mixed_pred'] == 3][milestone].astype(float).to_list()

        comprehensive_score_class0.extend(class0)
        comprehensive_score_class1.extend(class1)
        comprehensive_score_class2.extend(class2)
        comprehensive_score_class3.extend(class3)
        comprehensive_score_sib.extend(sib_bh_data[milestone].to_list())

        pvals = []
        pvals.append(stats.ttest_ind(class0, sib_bh_data[milestone], equal_var=False, alternative='two-sided').pvalue)
        pvals.append(stats.ttest_ind(class1, sib_bh_data[milestone], equal_var=False, alternative='two-sided').pvalue)
        pvals.append(stats.ttest_ind(class2, sib_bh_data[milestone], equal_var=False, alternative='two-sided').pvalue)
        pvals.append(stats.ttest_ind(class3, sib_bh_data[milestone], equal_var=False, alternative='two-sided').pvalue)
        pvals = multipletests(pvals, method='fdr_bh')[1]
        print(milestone)
        print(pvals)
        
        fig, ax = plt.subplots(1,1,figsize=(3.5, 4.5))
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.boxplot(data=[sib_bh_data[milestone].to_list(), class0, class1, class2, class3], showfliers=False, palette=['dimgray','violet','red','limegreen','blue'])
        ylims=ax.get_ylim()
        sns.stripplot(data=[sib_bh_data[milestone].to_list(), class0, class1, class2, class3], palette=['dimgray','violet','red','limegreen','blue'], alpha=0.1)
        plt.ylim(ylims)
        plt.xlabel('')
        plt.ylabel('months', fontsize=20)
        if f"{' '.join(milestone.split('_'))}" == 'used words age mos':
            plt.title('Age first used words', fontsize=22)
        elif f"{' '.join(milestone.split('_'))}" == 'walked age mos':
            plt.title('Age first walked', fontsize=22)
        plt.yticks(fontsize=16)
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)
            plt.gca().spines[axis].set_color('black')
        plt.savefig(f'figures/GFMM_milestone_validation_{milestone}.png', bbox_inches='tight')
        plt.close()
        
        # plot distributions for classes and sibs on one boxplot
        data = [sib_bh_data[milestone].to_list(), class0, class1, class2, class3]
        box_colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
        for j, (color, values) in enumerate(zip(box_colors, data)):
            parts = ax[0,0].violinplot(values, positions=[j], showmedians=True, showextrema=False, widths=0.65, bw_method=0.5)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(color)
                pc.set_edgecolor('black')

        ax[i//5, i%5].set_xlabel('')
        ax[i//5, i%5].set_ylabel('age (months)', fontsize=17)
        ax[i//5, i%5].set_title('Developmental milestones', fontsize=18)
        for axis in ['top','bottom','left','right']:
            ax[i//5, i%5].spines[axis].set_linewidth(1)
            ax[i//5, i%5].spines[axis].set_color('black')
        plt.tight_layout()
    
    plt.savefig(f'figures/GFMM_milestone_validation_classes_vs_sibs.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    developmental_milestones_validation()
    individual_registration_validation()
    vineland_validation()
    scq_validation()
    main_clinical_validation(only_sibs=True)
