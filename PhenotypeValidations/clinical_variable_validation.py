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


def get_figure_for_paper(only_sibs=False, impute=False):
    if impute:
        mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=0, header=0)
    else:
        mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
        #mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=0, header=0)

    # ISOLATING GROUPS OF VARIABLES FOR CLINICAL VALIDATION
    bms_data = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # 4700 probands with BMS for val
    
    birth_defect_features = ['birth_def_cns', 'birth_def_bone', 'birth_def_fac', 'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen']
    # combine into one feature - if any birth defect, then 1
    bms_data['birth_defect'] = np.where(bms_data[birth_defect_features].sum(axis=1) > 0, 1, 0)
    
    neurodev = ['growth_macroceph', 'growth_microceph', 'dev_id', 'tics','neuro_sz', 'birth_defect']
    maternal = ['birth_ivh', 'birth_pg_inf', 'birth_prem']
    mental_health = ['mood_ocd',  'mood_dep', 'mood_anx', 'behav_adhd']
    daily_living = ['feeding_dx', 'sleep_dx', 'dev_motor', 'dev_lang_dis']
    group = neurodev+maternal+mental_health+daily_living
    subset_validation_data = bms_data[group] # get data for validation features

    # get labels for plotting
    neuro_labels = ['Macrocephaly', 'Microcephaly', 'ID', 'Tics', 'Seizures/Epilepsy', 'Birth Defect']
    maternal_labels = ['IVH', 'Prenatal Infection', 'Premature Birth']
    mental_health_labels = ['OCD', 'Depression', 'Anxiety', 'ADHD']
    daily_living_labels = ['Feeding Disorder', 'Sleep Disorder', 'Motor Disorder', 'Language Delay']
    
    # validate against group of features from BMS - merge with BMS data for validation
    mixed_data = pd.merge(mixed_data, subset_validation_data, left_index=True, right_index=True)
    #print(mixed_data['birth_pg_inf'].value_counts())
    #print(list(mixed_data[(mixed_data['birth_pg_inf'] == 1) & (mixed_data['mixed_pred'] == 0)].index)); exit()
    # concatenate with sibling BMS data
    sibs = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_siblings_bms_validation.txt', sep='\t', index_col=0)
    if impute:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
        #sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt' # 1588 sibs
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sibs = pd.merge(sibs, paired_sibs, left_index=True, right_index=True) # subset to 1293 paired siblings who have BMS information
    sibs['birth_defect'] = np.where(sibs[birth_defect_features].sum(axis=1) > 0, 1, 0) # add birth defect feature to sibs
    # subset sibs to group of selected binary features from BMS
    sibs = sibs[group]
    # concat sibs to mixed_data and make mixed_pred = -1 as label for sibs
    mixed_data = pd.concat([sibs, mixed_data])
    mixed_data['mixed_pred'] = mixed_data['mixed_pred'].replace(np.nan, -1)

    # for every category, get fold enrichment and p-values and make bubble plot
    category_names = ['Neurodevelopmental', 'Mental Health', 'Daily Living'] # 'Maternal'
    labels = [neuro_labels, mental_health_labels, daily_living_labels] # maternal_labels
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(3, 1, figsize=(10.5, 12), sharex=True)
    for i, name, label, category in zip(np.arange(len(category_names)), category_names, labels, [neurodev, mental_health, daily_living]): # maternal
        # get p-values for each feature in category
        p_values = get_feature_enrichments_with_sibs(mixed_data[category+['mixed_pred']], name, only_sibs)
        validation_subset = p_values.loc[:,category+['cluster']]
        validation_subset = pd.melt(validation_subset, id_vars=['cluster'])
        # multi-hypothesis correction for 'value' column, for all hypotheses in plot
        #validation_subset['value'] = multipletests(validation_subset['value'], method='fdr_bh', alpha=0.05)[1]
        # take -log10(pval) of 'value' column
        validation_subset['value'] = -np.log10(validation_subset['value'])
        # print values for 'behav_adhd' for each cluster
        # get fold enrichments
        fold_enrichments = get_fold_enrichment(mixed_data[category+['mixed_pred']], only_sibs)
        fold_enrichments = pd.melt(fold_enrichments, id_vars=['cluster'])
        # incorporate fold_enrichments into validation_subset as column 'fold'
        validation_subset['Fold Enrichment'] = fold_enrichments['value']
        # subplots
        make_bubble_plot(validation_subset, category, label, name, impute=impute, ax=ax[i])
    
    # save figure
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.12, wspace=0.12)
    plt.savefig('GFMM_all_figures/GFMM_clinical_bubble_plots_validation.png', bbox_inches='tight')
    plt.close()


def make_bubble_plot(validation_subset, category, y_labels, category_name, enrichment=True, impute=False, ax=None):
    
    # x = -log10(pval), y = feature, size = fold enrichment, color = cluster
    validation_subset = validation_subset[validation_subset['cluster'] != -1] # exclude siblings (-1) from plot
    if impute:
        colors = ['violet', 'red', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({0: 'Low-ASD/Low-Delays', 1: 'High-ASD/High-Delays', 2: 'High-ASD/Low-Delays', 3: 'Low-ASD/High-Delays'})
    else:
        colors = ['violet', 'red', 'limegreen', 'blue']
        validation_subset['color'] = validation_subset['cluster'].map({0: 'violet', 1: 'red', 2: 'limegreen', 3: 'blue'})
        validation_subset['Cluster'] = validation_subset['cluster'].map({0: 'ASD-Lower Support Needs', 1: 'ASD-Higher Support Needs', 2: 'ASD-Social/RRB', 3: 'ASD-Developmentally Delayed'})
    markers = ['o', 'o', 'o', 'o']
    validation_subset['marker'] = validation_subset['cluster'].map({0: 'o', 1: 'o', 2: 'o', 3: 'o'})
    #validation_subset['color'] = np.where(validation_subset['value'] < -np.log10(0.05), 'gray', validation_subset['color'])
    
    if enrichment:
        #sns.scatterplot(data=validation_subset, x='Fold Enrichment', y='variable', hue='Cluster', palette=colors, markers=markers, s=200, alpha=0.95)
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
        #plt.savefig(f'GFMM_all_figures/GFMM_clinical_bubble_plot_FE_{category_name}_validation.png', bbox_inches='tight')
        #plt.close()
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(9,4.5))
        # scale size of bubbles to be more visible
        # change up the marker of the bubbles - X for sibs, o for others
        #scatter = ax.scatter(validation_subset['value'], validation_subset['variable'], s=validation_subset['Fold Enrichment']*85, c=validation_subset['color'], alpha=0.8)
        #    else:
        #        ax.scatter(row['value'], row['variable'], s=row['Fold Enrichment']*85, c=row['color'], alpha=0.8)
        sns.scatterplot(data=validation_subset, x='value', y='variable', size='Fold Enrichment', hue='Cluster', palette=colors, markers=markers, sizes=(50, 500))
        for i, row in validation_subset.iterrows():
            if row['cluster'] == -1:
                ax.scatter(row['value'], row['variable'], s=120, c='black', marker='x', alpha=0.8)
        # add manual X for legend
        #plt.txt = plt.text(0, 0, 'X', fontsize=16, color='black', ha='center', va='center', backgroundcolor='white')
        plt.xlabel('-log10(q-value)', fontsize=20)
        plt.ylabel('')
        # make yticks larger
        plt.yticks(fontsize=20)
        # rename legend labels to be more descriptive
        # make sure legend has appropriate colors
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = ['Siblings', 'ASD-Lower Support Needs', 'ASD-Higher Support Needs', 'ASD-Social/RRB', 'ASD-Developmentally Delayed']
        handles[0] = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10, label='Siblings')
        #red_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='High-ASD/High-Delays')
        #violet_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='violet', markersize=10, label='Low-ASD/Low-Delays')
        #green_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='High-ASD/Low-Delays')
        #blue_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Low-ASD/High-Delays')
        #handles = [gray_circle, red_circle, violet_circle, green_circle, blue_circle]
        plt.legend(handles, labels, fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        # legend of size of bubbles
        plt.legend(fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        # make legend with both cluster and size
        #legend = plt.legend(*scatter.legend_elements("sizes"), title="Fold Enrichment", loc="upper left", bbox_to_anchor=(1, 1))
        #plt.gca().add_artist(legend)
        #ax.legend(handles=[gray_circle, red_circle, violet_circle, green_circle, blue_circle], fontsize=14, title='Cluster', title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        # make legend for size of bubbles
        plt.title(f'{category_name}', fontsize=24)
        # if category_name in ['Maternal', 'Daily Living'], make y tick labels on the right side of the plot instead of left
        #if category_name in ['Maternal', 'Daily Living']:
        #    ax.yaxis.tick_right()
        #    ax.yaxis.set_label_position('right')
        #    ax.get_legend().remove()
        # make y axis labels more descriptive
        plt.yticks([x for x in range(len(y_labels))], y_labels, fontsize=20)
        # x axis line at -log10(0.05)
        plt.axvline(x=-np.log10(0.05), color='gray', linestyle='--', linewidth=1.4)
        # increase frame width
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
            # set color to black
            ax.spines[axis].set_color('black')
        plt.savefig(f'GFMM_all_figures/GFMM_clinical_bubble_plot_{category_name}_validation.png', bbox_inches='tight')
        plt.close()

def run_main_LCA_model_nobms(ncomp, feature_group):
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_revised_no_bms.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI, no BMS
    #datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_revised_no_bms_imputed.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI, no BMS
    age = datadf['age_at_eval_years']

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'], axis=1)  # drop asd label and convert to np array
    
    # ISOLATING GROUPS OF VARIABLES FOR CLINICAL VALIDATION
    bms_data = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI
    if feature_group == 'attn_behav':
        group = ['behav_adhd', 'behav_conduct', 'behav_intermitt_explos', 'behav_odd']
    elif feature_group == 'preg_comp':
        group = ['birth_ivh', 'birth_oxygen', 'birth_pg_inf', 'birth_prem', 'cog_med', 'birth_etoh_subst']
    elif feature_group == 'lang':
        group = ['dev_lang_dis', 'dev_ld', 'dev_motor','dev_soc_prag', 'dev_speech']
    elif feature_group == 'sleep':
        group = ['sleep_dx', 'sleep_probs', 'feeding_dx', 'eating_probs']
    elif feature_group == 'mood':
        group = ['mood_anx', 'mood_bipol', 'mood_dep', 'mood_dmd', 'mood_ocd', 'mood_sep_anx', 'mood_soc_anx']
    elif feature_group == 'comorbid':
        group = ['neuro_inf', 'neuro_sz', 'tics']
    elif feature_group == 'id':
        group = ['dev_id']
    elif feature_group == 'cephaly':
        group = ['growth_macroceph', 'growth_microceph']
    elif feature_group == 'combined':
        group = ['behav_adhd', 'mood_anx', 'mood_ocd', 'sleep_dx', 'dev_lang_dis', 'dev_motor', 'dev_id']
    elif feature_group == 'neurodev': # PAPER CATEGORY 1
        group = ['neuro_sz', 'tics', 'birth_def_cns_brain', 'growth_macroceph', 'growth_microceph', 'dev_id']
    elif feature_group == 'maternal': # PAPER CATEGORY 2
        group = ['birth_ivh', 'birth_oxygen', 'birth_pg_inf', 'birth_prem', 'birth_etoh_subst']
    elif feature_group == 'mental_health': # PAPER CATEGORY 3
        group = ['mood_anx', 'mood_soc_anx', 'mood_dep', 'mood_ocd', 'behav_adhd']
    elif feature_group == 'daily_living': # PAPER CATEGORY 4
        group = ['feeding_dx', 'eating_probs', 'eating_disorder', 'sleep_dx', 'sleep_probs', 'encopres', 'enures', 'dev_lang_dis', 'dev_speech', 'dev_motor']
    
    subset_validation_data = bms_data[group] # get data for validation features
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    # make sure the tested group is not in binary_columns (it shouldn't be, but just in case)
    binary_columns = [x for x in binary_columns if x not in group]

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
    print(mixed_data['mixed_pred'].value_counts())
    mixed_data['age'] = age
    
    # validate against group of features from BMS - merge with data for validation
    mixed_data = pd.merge(mixed_data, subset_validation_data, left_index=True, right_index=True)

    # subset mixed_data to group of selected binary features from BMS + mixed_pred
    mixed_data = mixed_data[group+['mixed_pred']]

    # merge with sibling BMS data
    sibs = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_siblings_bms_validation.txt', sep='\t', index_col=0)
    # subset sibs to group of selected binary features from BMS
    sibs = sibs[group]
    # concat sibs to mixed_data and make mixed_pred = -1
    mixed_data = pd.concat([sibs, mixed_data])
    mixed_data['mixed_pred'] = mixed_data['mixed_pred'].replace(np.nan, -1)
    
    # get feature enrichments - optional
    feature_sig_norm_high = get_feature_enrichments_with_sibs(mixed_data)
    fold_enrichments = get_fold_enrichment(mixed_data)
    # PVAL_DF IS CORRECTED FOR MULTIPLE HYPOTHESIS TESTING
    #plot_line_polar(mixed_data, feature_sig_norm_high, feature_vector, 'lca_plots/LCA_line_polar_plot_vineland_motor_validation')

    # PLOT CLINICAL FEATURE VALIDATIONS (FE = TRUE, else p-values)
    plot_validation(fold_enrichments, group, feature_group, fold=True)
    
    return mixed_data

def plot_validation(feature_sig_norm_high, features, feature_group, fold=False):

    # PLOT CLINICAL VALIDATION FEATURES VS. -LOG10(PVAL) FOR ENRICHMENT OF BINARY VALIDATION VARIABLES
    # convert feature_sig_norm_high to a dataframe and subset for the attn_behav_group
    # melt validation_subset so can plot on a scatterplot
    validation_subset = feature_sig_norm_high.loc[:,features+['cluster']]
    validation_subset = pd.melt(validation_subset, id_vars=['cluster'])
    # multi-hypothesis correction for 'value' column, for all hypotheses in plot
    if not fold:
        validation_subset['value'] = multipletests(validation_subset['value'], method='fdr_bh')[1]
        # take -log10(pval) of 'value' column
        validation_subset['value'] = -np.log10(validation_subset['value'])
        # replace 0 with 0.01
        #validation_subset['value'] = validation_subset['value'].replace(0, -0.5) # replace 0 so that two classes don't overlap
    # plot validation_subset on a scatterplot, color by cluster
    plt.style.use('seaborn-white')
    if feature_group == 'combined':
        plt.figure(figsize=(11, 8))
    elif len(features) > 4:
        plt.figure(figsize=(9.5, 8))
    else:
        plt.figure(figsize=(6, 8))
        plt.xlim([-0.5, len(features)-0.5])
    # scatter plot of validation_subset
    if fold:
        colors = ['gray', 'red', 'violet', 'green', 'blue']
    else:
        colors = ['gray', 'violet', 'green', 'blue', 'red']
    sns.scatterplot(x='variable', y='value', hue='cluster', data=validation_subset, palette=colors, marker='o', s=125, alpha=0.8)
    plt.xlabel('')
    if fold:
        plt.ylabel('Fold Enrichment', fontsize=20)
    else:
        plt.ylabel('-log10(p-value)', fontsize=20)
    # customize xticks
    if feature_group == 'combined':
        plt.xticks([x for x in range(len(features))], ['ADHD', 'Anxiety', 'OCD', 'Sleep', 'Lang', 'Motor', 'ID'], fontsize=16)
    else:
        plt.xticks([x for x in range(len(features))], features, fontsize=16)
        plt.xticks(rotation=20, ha='right')
    # update legend labels to be more descriptive
    import matplotlib.lines as mlines
    gray_square = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=10, label='Siblings')
    red_square = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='High-ASD/High-Delays')
    violet_square = mlines.Line2D([], [], color='violet', marker='o', linestyle='None',
                          markersize=10, label='Low-ASD/Low-Delays')
    green_square = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=10, label='ASYM-SOCO')
    blue_square = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=10, label='ASYM-RRB')
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels = ['Severe', 'High Function', 'Social+Anxiety', 'ID+RRB']
    plt.legend(handles=[gray_square, red_square, violet_square, green_square, blue_square], fontsize=14)
    # horizontal line at y=1.3
    if fold:
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)
        plt.savefig(f'lca_plots/LCA_{feature_group}_fold_enrichment_validation_scatter_wsib.png', bbox_inches='tight')
    else:
        plt.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=2)
        plt.savefig(f'lca_plots/LCA_{feature_group}_validation_scatter_wsib.png', bbox_inches='tight')
    plt.close()
    print(f"done with {feature_group} scatterplot.")

def get_fold_enrichment(mixed_data, only_sibs=False):
    '''x-fold enrichment = % of group with condition divided by percentage of background with condition.'''
    ## extract dataframes for each class
    feature_sig_df_high = pd.DataFrame()
    feature_vector = list()

    sibs = mixed_data[mixed_data['mixed_pred'] == -1]
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## make sure feature is binary
            total_in_sibs = len(sibs[feature])
            if only_sibs:
                sibs_sum = int(np.sum(sibs[feature]))+1 # add pseudocount
            else:
                sibs_sum = int(np.sum(sibs[feature])) # add pseudocount in case 0?
            total_in_class0 = len(class0[feature])
            class0_sum = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            class1_sum = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            class2_sum = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            class3_sum = int(np.sum(class3[feature]))

            # calculate fold enrichment for each class compared to sibs+other classes
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

    ### BINOMIAL TEST for each class and feature
    feature_sig_df_high = pd.DataFrame()
    feature_vector = list()

    ## extract dataframes for each class
    sibs = mixed_data[mixed_data['mixed_pred'] == -1]
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## only for binary validation features
            # perform a binomial test for each class
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
            
            # perform a binomial test for each class
            sfsib = binomtest(subset_sibs, n=total_in_sibs, p=background_all, alternative='greater').pvalue
            sf0 = binomtest(subset_class0, n=total_in_class0, p=background, alternative='greater').pvalue
            sf1 = binomtest(subset_class1, n=total_in_class1, p=background, alternative='greater').pvalue
            sf2 = binomtest(subset_class2, n=total_in_class2, p=background, alternative='greater').pvalue
            sf3 = binomtest(subset_class3, n=total_in_class3, p=background, alternative='greater').pvalue

            # if any p-value is 0, change p-value to epsilon
            if only_sibs:
                # need to do epsilon correction for pvalues of 0
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
            # FDR correction for multiple hypothesis testing and make sure order of p-values is preserved
            feature_sig_df_high[feature] = multipletests(feature_sig_df_high[feature], method='fdr_bh', alpha=0.05)[1]
            feature_vector.append(feature)

    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [-1, 0, 1, 2, 3]
    #feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    #feature_sig_norm_low['cluster'] = [-1, 0, 1, 2, 3]

    return feature_sig_norm_high

def vineland_validation(impute=False):
    '''
    Clinical variable validation for vineland.
    '''
    if impute:
        mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=0, header=0)
    else:
        mixed_data = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    
    # LOAD AND ADD CLINICAL VALIDATION FEATURES FROM VINELAND
    #cog_impair = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_cog_impair_validation.txt', sep='\t', index_col=0)
    #ml_predicted_iq = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_ml_iq_validation.txt', sep='\t', index_col=0)
    vineland = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_data_vineland_validation.txt', sep='\t', index_col=0)
    motor = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_data_motor_validation.txt', sep='\t', index_col=0)

    # drop rows with missing values
    vineland = vineland.dropna()
    motor = motor.dropna()

    # get kids in vineland who are not in motor
    #vineland = vineland[~vineland.index.isin(motor.index)]

    # merge with mixed data on the index
    mixed_data = pd.merge(mixed_data, vineland, left_index=True, right_index=True)
    mixed_data = pd.merge(mixed_data, motor, left_index=True, right_index=True)

    # PLOT mixed_data abc_standard, dls_standard, communication_standard, and soc_standard on on boxplot, color by cluster
    # plot mixed_data[['abc_standard', 'dls_standard', 'communication_standard', 'soc_standard', 'cluster']]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))
    data = mixed_data[['abc_standard', 'dls_standard', 'communication_standard', 'soc_standard', 'motor_standard', 'mixed_pred']]
    #data = mixed_data[['motor_standard', 'mixed_pred']]
    # melt data but keep cluster so can color by hue
    data = pd.melt(data, id_vars=['mixed_pred'])
    # plot and set colors=['red','violet','green','blue']
    if impute:
        sns.boxplot(x='variable', y='value', hue='mixed_pred', data=data, palette=['violet','red','limegreen','blue'], showfliers=False)
    else:
        sns.boxplot(x='variable', y='value', hue='mixed_pred', data=data, palette=['violet','red','limegreen','blue'], showfliers=False) # whiskerprops = dict(color = "white"), capprops = dict(color = "white"),
    plt.xlabel('')
    plt.ylabel('Vineland Scores', fontsize=24)
    #plt.title('Vineland Validation', fontsize=26)
    # customize xticks
    plt.xticks([0,1,2,3,4], ['ABC', 'DLS', 'Comm', 'Soc', 'Motor'], fontsize=24)
    # update legend labels to be more descriptive
    handles, labels = plt.gca().get_legend_handles_labels()
    if impute:
        labels = ['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    else:
        labels = ['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    # make borders black
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    # outside of plot area
    plt.legend(handles, labels, fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('GFMM_all_figures/GFMM_motor_vineland_validation_boxplot.png', bbox_inches='tight')   
    plt.close()
    

def plot_line_polar(data, feature_sig_norm_high, feature_vector, output):
    '''create line polar plots to summarize class identifying features'''
    
    # normalize feature vector
    scaler = StandardScaler()
    feature_vector.append('cluster')
    feature_sig_norm_high = pd.DataFrame(scaler.fit_transform(feature_sig_norm_high), columns=feature_vector)
    
    y = list(data['mixed_pred'])
    X = data.drop(['mixed_pred'], axis=1)

    # normalize:
    mixed_data_norm = scaler.fit_transform(X)
    mixed_data_norm = pd.DataFrame(mixed_data_norm, columns=X.columns)
    mixed_data_norm['cluster'] = y

    #features_to_visualize = ['social_problems_t_score', 'dsm5_anxiety_problems_t_score', 'i_stereotyped_behavior_score', 'vi_restricted_behavior_score', 'iii_compulsive_behavior_score', 'dev_id', 'ml_predicted_cog_impair', 'cluster'] # for cbcl model
    features_to_visualize = ['social_problems_t_score', 'dsm5_anxiety_problems_t_score', 'i_stereotyped_behavior_score', 'vi_restricted_behavior_score', 'dev_id', 'motor_standard', 'cluster'] # for cbcl model

    selected_features = mixed_data_norm[[x for x in features_to_visualize if x in feature_vector]]
    
    ### FINAL PLOT SUMMARIZING THE 4 CLASSES
    polar = selected_features.groupby('cluster').mean().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])

    ### option to CUSTOMIZE the plot to only plot one (or select) classes at a time
    polar1 = polar[polar['cluster'] == 0] # only cluster 1
    polar2 = polar[(polar['cluster'] == 0) | (polar['cluster'] == 1)] # only 1 and 2
    polar3 = polar[(polar['cluster'] == 0) | (polar['cluster'] == 1) | (polar['cluster'] == 2)]
    polar4 = polar # all classes
    
    # rename variables for plotting
    polar = polar4
    #polar['variable'] = polar['variable'].replace('final_score', 'SCQ Score')
    polar['variable'] = polar['variable'].replace('i_stereotyped_behavior_score', 'Stereotyped Behavior')
    polar['variable'] = polar['variable'].replace('social_problems_t_score', 'Social Problems')
    polar['variable'] = polar['variable'].replace('dsm5_anxiety_problems_t_score', 'Anxiety Problems')
    #polar['variable'] = polar['variable'].replace('iii_compulsive_behavior_score', 'Compulsive Behavior')
    polar['variable'] = polar['variable'].replace('vi_restricted_behavior_score', 'Restricted Behavior')
    #polar['variable'] = polar['variable'].replace('v_sameness_behavior_score', 'Sameness Behavior')
    polar['variable'] = polar['variable'].replace('dev_id', 'Intellectual Disability')
    #polar['variable'] = polar['variable'].replace('derived_cog_impair', 'Cognitive Impairment (Derived)')
    #polar['variable'] = polar['variable'].replace('ml_predicted_cog_impair', 'Cognitive Impairement (ML-Predicted)')
    #polar['variable'] = polar['variable'].replace('dev_lang_dis', 'Language Delay')
    # Vineland validation features:
    polar['variable'] = polar['variable'].replace('abc_standard', 'Vineland ABC Score')
    polar['variable'] = polar['variable'].replace('communication_standard', 'Vineland Communication Score')
    polar['variable'] = polar['variable'].replace('dls_standard', 'Vineland Daily Living Score')
    polar['variable'] = polar['variable'].replace('soc_standard', 'Vineland Socialization Score')
    polar['variable'] = polar['variable'].replace('motor_standard', 'Vineland Motor Score')

    colors = ['red','violet','green','blue']
    fig = px.line_polar(polar, r="value", theta="variable", color="cluster", template="plotly_dark",
                        color_discrete_sequence=colors, line_close=True, height=800, width=1400) # 
    fig.update_layout(
        font_size=20
    )
    fig.write_image(f"{output}.png")

def individual_registration_validation(impute=False):
    file = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12/individuals_registration_2022-12-12.csv'
    data = pd.read_csv(file, index_col=0)
    vars_for_val = ['num_asd_parents', 'num_asd_siblings', 'diagnosis_age', 'iep_asd', 'cognitive_impairment_at_enrollment', 'language_level_at_enrollment'] #
    
    # clean up data
    data['num_asd_parents'] = data['num_asd_parents'].replace(999, np.nan)
    data['num_asd_siblings'] = data['num_asd_siblings'].replace(999, np.nan)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Uses longer sentences of his/her own and is able to tell you something that happened', 3)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Combines 3 words together into short sentences', 2)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('Uses single words meaningfully (for example, to request)', 1)
    data['language_level_at_enrollment'] = data['language_level_at_enrollment'].replace('No words/does not speak', 0)

    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=0, header=0) #6406
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0) # 5280 probands
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=0, header=0) # 4700 probands
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt' # 1588 sibs
    
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)

    # merge each with data
    pro_data = pd.merge(data, gfmm_labels[['mixed_pred']], left_index=True, right_index=True)
    sib_data = pd.merge(data, paired_sibs, left_index=True, right_index=True)

    # for each variable, plot distribution of each class vs. sibs
    # make subplots
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

        # hypothesis testing (ttest) - is group 2 > group 3?
        num_unique_vals = len(var_data[var].unique())
        if num_unique_vals == 2:
            # binomial test
            group0 = var_data[var_data['mixed_pred'] == 0][var]
            group1 = var_data[var_data['mixed_pred'] == 1][var]
            group2 = var_data[var_data['mixed_pred'] == 2][var]
            group3 = var_data[var_data['mixed_pred'] == 3][var]
            pvals = []
            pvals.append(binomtest(np.sum(group0), n=len(group0), p=np.sum(group3)/len(group3)).pvalue)
            pvals.append(binomtest(np.sum(group1), n=len(group1), p=np.sum(group3)/len(group3)).pvalue)
            pvals.append(binomtest(np.sum(group2), n=len(group2), p=np.sum(group3)/len(group3)).pvalue)
            pvals = multipletests(pvals, method='fdr_bh')[1]
            print(var)
            print(pvals)
        else:
            # ttest
            group0 = var_data[var_data['mixed_pred'] == 0][var]
            group1 = var_data[var_data['mixed_pred'] == 1][var]
            group2 = var_data[var_data['mixed_pred'] == 2][var]
            group3 = var_data[var_data['mixed_pred'] == 3][var]
            pvals = []
            pvals.append(ttest_ind(group0, group3, equal_var=False, alternative='greater').pvalue)
            pvals.append(ttest_ind(group1, group3, equal_var=False, alternative='greater').pvalue)
            pvals.append(ttest_ind(group2, group3, equal_var=False, alternative='greater').pvalue)
            pvals = multipletests(pvals, method='fdr_bh')[1]
            print(var)
            print(pvals)
            
        # plot mean of each class as a bar plot
        if impute:
            sns.barplot(x='mixed_pred', y=var, data=var_data, ax=ax[i], palette=['violet','red','limegreen','blue','dimgray'], linewidth = 1.5, edgecolor='black', alpha=0.95, dodge=False)
        else:
            # BOXPLOT + STRIPPLOT
            #sns.boxplot(x='mixed_pred', y=var, data=var_data, ax=ax[i], showfliers=False, palette=['violet','red','limegreen','blue'])
            #sns.stripplot(x='mixed_pred', y=var, data=var_data, ax=ax[i], palette=['violet','red','limegreen','blue'], alpha=0.1)
            # BAR PLOT:
            sns.barplot(x='mixed_pred', y=var, data=var_data, ax=ax[i], palette=['violet','red','limegreen','blue','dimgray'], linewidth = 1.5, edgecolor='black', alpha=0.85, dodge=False)
        
        if var in ['num_asd_parents', 'num_asd_siblings']:
            continue
            #if impute:
            #    ax[i].set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], fontsize=14)
            #else:
            #    ax[i].set_xticklabels(['High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays', 'Siblings'], fontsize=14)
        elif var in ['cognitive_impairment_at_enrollment', 'language_level_at_enrollment']:
            if impute:
                ax[i].set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=14, rotation=35, ha='right')
            else:
                pass
                #ax[i].set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=14, rotation=35, ha='right')
        
        ax[i].set_xlabel('')
        ax[i].set_ylabel('Age (months)', fontsize=18)
        ax[i].set_title(f'{variable_names[i]}', fontsize=20)
        #ax[i].set_xticklabels(['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=14)
        #ax[i].set_yticklabels(fontsize=14)
        # make borders black
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(1)
            ax[i].spines[axis].set_color('black')
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig('GFMM_all_figures/GFMM_individual_registration_validation.png', bbox_inches='tight')
    plt.close()

    # plot age_at_registration_years
    var_data = pro_data[['age_at_registration_years', 'mixed_pred']]
    var_data = var_data.dropna()
    print(var_data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(x='mixed_pred', y='age_at_registration_years', data=var_data, palette=['violet','red','green','blue'], linewidth = 1.5, edgecolor='black', alpha=0.95, dodge=False)
    #ax.set_xticklabels(['Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=14, rotation=35, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Age (years)', fontsize=16)
    ax.set_title(f'Age at registration', fontsize=20)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color('black')
    plt.tight_layout()
    plt.savefig('GFMM_all_figures/GFMM_age_at_registration.png', bbox_inches='tight')
    plt.close()


def scq_validation(impute=False):
    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=0, header=0)
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=0, header=0)
    # get sibling data for SCQ
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv')
    scqdf = scqdf.loc[(scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) & (scqdf['age_at_eval_years'] >= 4)]
    scqdf = scqdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','current_depend_adult','age_at_eval_months','scq_measure_validity_flag','eval_year','missing_values','summary_score'],axis=1)
    scqdf = scqdf[scqdf['asd'] == 0]
    # get sibs
    if impute:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        #sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_4700_siblings_spids.txt'
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    # merge with sibling data
    sib_data = pd.merge(scqdf, paired_sibs, left_index=True, right_index=True)
    sib_scq_data = sib_data['final_score'].dropna().astype(int).to_list()

    # get final_score for each class
    final_score = gfmm_labels[['final_score', 'mixed_pred']]
    all_proband_scq_data = final_score['final_score'].dropna().astype(int).to_list()
    # get final_score list for each class
    class0 = final_score[final_score['mixed_pred'] == 0]['final_score'].dropna().astype(int).to_list()
    class1 = final_score[final_score['mixed_pred'] == 1]['final_score'].dropna().astype(int).to_list()
    class2 = final_score[final_score['mixed_pred'] == 2]['final_score'].dropna().astype(int).to_list()
    class3 = final_score[final_score['mixed_pred'] == 3]['final_score'].dropna().astype(int).to_list()

    # hypothesis testing vs. sibs
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
    # FDR correction
    p_vals = multipletests(p_vals, method='fdr_bh')[1]
    print(p_vals)

    # first plot all probands vs. sibs
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    data = [sib_scq_data, all_proband_scq_data]
    ax = sns.boxplot(data=data, showfliers=False, palette=['dimgray','purple']) # whiskerprops = dict(color = "white"), capprops = dict(color = "white"),
    plt.xlabel('')
    plt.ylabel('SCQ Final Score', fontsize=20)
    #plt.title('SCQ final scores', fontsize=24)
    # customize xticks
    plt.xticks([0,1], ['Siblings', 'Probands'], fontsize=20)
    # update legend labels to be more descriptive
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'Probands']
    # make borders black
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1)
        plt.gca().spines[axis].set_color('black')
    plt.legend(handles, labels, fontsize=16)
    # add p value to plot
    plt.text(0.5, 0.9, f'***', ha='center', va='center', transform=ax.transAxes, fontsize=20)
    plt.savefig('GFMM_all_figures/GFMM_scq_validation_all_pros_vs_sibs.png', bbox_inches='tight')
    plt.close()

    # plot final_score for each class vs. sibs
    # make one boxplot with all classes and sibs
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7.5, 5.5))
    data = [sib_scq_data, class0, class1, class2, class3]
    # plot and set colors=['red','violet','green','blue']
    if impute:
        sns.boxplot(data=data, palette=['dimgray','violet','red','limegreen','blue'], showfliers=False)
    else:
        sns.boxplot(data=data, palette=['dimgray','violet','red','limegreen','blue'], showfliers=False, whiskerprops = dict(color = "black", linewidth=1.5), capprops = dict(color = "black"),
                    medianprops=dict(color='black', linewidth=1.5), boxprops=dict(edgecolor='black', linewidth=1.5))
        sns.stripplot(data=data, palette=['dimgray','violet','red','limegreen','blue'], alpha=0.1)
    plt.xlabel('')
    plt.ylabel('SCQ Final Score', fontsize=20)
    #plt.title('SCQ final scores by class', fontsize=24)
    # customize xticks
    #if impute:
    #    plt.xticks([0,1,2,3,4], ['Siblings', 'lowASD/lowDelays', 'highASD/highDelays', 'highASD/lowDelays', 'lowASD/highDelays'], fontsize=18, rotation=45, ha='right')
    #else:
    #    plt.xticks([0,1,2,3,4], ['Siblings', 'lowASD/lowDelays', 'highASD/highDelays', 'highASD/lowDelays', 'lowASD/highDelays'], fontsize=18, rotation=45, ha='right')
    # update legend labels to be more descriptive
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Siblings', 'High-ASD/High-Delays', 'Low-ASD/Low-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays']
    # make y tick labels larger
    plt.yticks(fontsize=16)
    # make borders black
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.legend(handles, labels, fontsize=16)
    plt.savefig('GFMM_all_figures/GFMM_scq_validation_classes_vs_sibs.png', bbox_inches='tight')
    plt.close()


def developmental_milestones_validation(impute=False):
    if impute:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_imputed_labeled.csv', index_col=0, header=0)
    else:
        gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
        #gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/LCA_4classes_training_data_nobms_labeled.csv', index_col=0, header=0)
    
    # get sibling data for background HX
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
    bhdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv')
    bhdf = bhdf.loc[(bhdf['age_at_eval_years'] <= 18) & (bhdf['age_at_eval_years'] >= 4)]
    dev_milestones = ['smiled_age_mos', 'sat_wo_support_age_mos', 'crawled_age_mos', 'walked_age_mos',
                                                        'fed_self_spoon_age_mos', 'used_words_age_mos', 'combined_words_age_mos', 'combined_phrases_age_mos',
                                                        'bladder_trained_age_mos', 'bowel_trained_age_mos']
    bhdf = bhdf.set_index('subject_sp_id',drop=True)[dev_milestones]

    # get sibs
    if impute:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_6400_siblings_spids.txt'
    else:
        sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    # merge with sibling data
    sib_data = pd.merge(bhdf, paired_sibs, left_index=True, right_index=True)
    sib_bh_data = sib_data[dev_milestones].dropna().astype(float)

    # replace 888 with 0 in sib_bh_data
    sib_bh_data = sib_bh_data.replace(888,0)

    # make subplots for each developmental milestone
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 5, figsize=(18.5, 8))

    comprehensive_score_class0 = []
    comprehensive_score_class1 = []
    comprehensive_score_class2 = []
    comprehensive_score_class3 = []
    comprehensive_score_sib = []

    for i, milestone in enumerate(dev_milestones):
        
        # get distribution for each class
        milestone_data = gfmm_labels[[milestone, 'mixed_pred']]
        milestone_data[milestone] = milestone_data[milestone].apply(lambda x: 216 if x > 216 else x) # set max to 18 years old
        all_proband_bh_data = milestone_data[milestone].astype(float).to_list()
        # get final_score list for each class
        class0 = milestone_data[milestone_data['mixed_pred'] == 0][milestone].astype(float).to_list()
        class1 = milestone_data[milestone_data['mixed_pred'] == 1][milestone].astype(float).to_list()
        class2 = milestone_data[milestone_data['mixed_pred'] == 2][milestone].astype(float).to_list()
        class3 = milestone_data[milestone_data['mixed_pred'] == 3][milestone].astype(float).to_list()

        # build comprehensive score for each class
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
        #plt.title(f"{' '.join(milestone.split('_'))}", fontsize=24)
        # make borders black
        plt.yticks(fontsize=16)
        for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)
            plt.gca().spines[axis].set_color('black')
        plt.savefig(f'GFMM_all_figures/GFMM_milestone_validation_{milestone}.png', bbox_inches='tight')
        plt.close()
        
        continue
        # plot distributions for classes and sibs on one boxplot
        data = [sib_bh_data[milestone].to_list(), class0, class1, class2, class3]
        if impute:
            box_colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
            for j, (color, values) in enumerate(zip(box_colors, data)):
                boxprops = dict(facecolor=color, edgecolor='black')
                ax[i//5, i%5].boxplot(values, positions=[j], showfliers=False, patch_artist=True, boxprops=boxprops, capprops=dict(color='black'), medianprops=dict(color='black'), meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'), widths=0.65)
        else:
            box_colors = ['dimgray', 'violet', 'red', 'limegreen', 'blue']
            for j, (color, values) in enumerate(zip(box_colors, data)):
                #boxprops = dict(facecolor=color, edgecolor='black')
                # make violinplot and set colors = box_colors
                parts = ax[0,0].violinplot(values, positions=[j], showmedians=True, showextrema=False, widths=0.65, bw_method=0.5)
                # showfliers=False, patch_artist=True, boxprops=boxprops, capprops=dict(color='black'), medianprops=dict(color='black'), meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'), widths=0.65
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')

        ax[i//5, i%5].set_xlabel('')
        ax[i//5, i%5].set_ylabel('age (months)', fontsize=17)
        #ax[i//5, i%5].set_title(f"{' '.join(milestone.split('_'))}", fontsize=17)
        ax[i//5, i%5].set_title('Developmental milestones', fontsize=18)
        for axis in ['top','bottom','left','right']:
            ax[i//5, i%5].spines[axis].set_linewidth(1)
            ax[i//5, i%5].spines[axis].set_color('black')
        # make layout tight
        plt.tight_layout()
    
    plt.savefig(f'GFMM_all_figures/GFMM_milestone_validation_classes_vs_sibs.png', bbox_inches='tight')
    plt.close()

    # plot comprehensive score for each class vs. sibs
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    number_of_milestones = len(dev_milestones)
    #comprehensive_score_sib = [x/number_of_milestones for x in comprehensive_score_sib]
    #comprehensive_score_class0 = [x/number_of_milestones for x in comprehensive_score_class0]
    #comprehensive_score_class1 = [x/number_of_milestones for x in comprehensive_score_class1]
    #comprehensive_score_class2 = [x/number_of_milestones for x in comprehensive_score_class2]
    #comprehensive_score_class3 = [x/number_of_milestones for x in comprehensive_score_class3]

    data = [comprehensive_score_sib, comprehensive_score_class0, comprehensive_score_class1, comprehensive_score_class2, comprehensive_score_class3]
    
    if impute:
        sns.boxplot(data=data, showfliers=False, palette=['dimgray','violet','red','green','blue'], width=0.65)
    else:
        sns.boxplot(data=data, showfliers=False, palette=['dimgray','violet','red','green','blue'], width=0.65)
    plt.xlabel('')
    if impute:
        plt.xticks([0,1,2,3,4], ['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=18, rotation=35, ha='right')
    else:
        plt.xticks([0,1,2,3,4], ['Siblings', 'Low-ASD/Low-Delays', 'High-ASD/High-Delays', 'High-ASD/Low-Delays', 'Low-ASD/High-Delays'], fontsize=18, rotation=35, ha='right')
    plt.ylabel('Developmental Milestones Score (months)', fontsize=18)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)
        plt.gca().spines[axis].set_color('black')
    plt.savefig('GFMM_all_figures/GFMM_milestone_comprehensive_classes_vs_sibs_comprehensive.png', bbox_inches='tight')
    plt.close()


def scq_dev_embeddings():
    gfmm_labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0, header=0)
    
    # get sibling data for background HX
    BASE_PHENO_DIR = '/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/SPARK_collection_v9_2022-12-12'
    bhdf = pd.read_csv(f'{BASE_PHENO_DIR}/background_history_sibling_2022-12-12.csv')
    bhdf = bhdf.loc[(bhdf['age_at_eval_years'] <= 18) & (bhdf['age_at_eval_years'] >= 4)]
    dev_milestones = ['walked_age_mos', 'used_words_age_mos']
    bhdf = bhdf.set_index('subject_sp_id',drop=True)[dev_milestones]

    sibling_list = '/mnt/home/alitman/ceph/WES_V2_data/WES_5392_siblings_spids.txt'
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    # merge with sibling data
    sib_data = pd.merge(bhdf, paired_sibs, left_index=True, right_index=True)
    sib_dev_data = sib_data[dev_milestones].dropna().astype(float)
    # take average
    sib_dev_data['dev_embeddings'] = sib_dev_data.mean(axis=1)
    sib_dev_data = sib_dev_data.drop(['walked_age_mos', 'used_words_age_mos'], axis=1)
    
    scqdf = pd.read_csv(f'{BASE_PHENO_DIR}/scq_2022-12-12.csv')
    scqdf = scqdf.loc[(scqdf['age_at_eval_years'] <= 18) & (scqdf['missing_values'] < 1) & (scqdf['age_at_eval_years'] >= 4)]
    scqdf = scqdf.set_index('subject_sp_id',drop=True).drop(['respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id','current_depend_adult','age_at_eval_months','scq_measure_validity_flag','eval_year','missing_values','summary_score'],axis=1)
    scqdf = scqdf[scqdf['asd'] == 0]
    paired_sibs = pd.read_csv(sibling_list, sep='\t', header=None, index_col=0)
    sib_data = pd.merge(scqdf, paired_sibs, left_index=True, right_index=True)
    sib_scq_data = sib_data['final_score'].dropna().astype(int).to_list()

    # get final_score for each class
    gfmm_labels = gfmm_labels[['walked_age_mos', 'used_words_age_mos', 'final_score', 'mixed_pred']]
    # combine walked_age_mos and used_words_age_mos by taking average
    gfmm_labels['dev_embeddings'] = gfmm_labels[['walked_age_mos', 'used_words_age_mos']].mean(axis=1)
    gfmm_labels = gfmm_labels.drop(['walked_age_mos', 'used_words_age_mos'], axis=1)
    
    # color array
    colors = ['violet','red','green','blue']
    gfmm_labels['color'] = gfmm_labels['mixed_pred'].apply(lambda x: colors[x])
    # plot final score vs dev_embeddings, color with mixed_pred
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=gfmm_labels, x='dev_embeddings', y='final_score', hue='color', palette=colors, alpha=0.4)
    plt.xlabel('Developmental Milestones Embeddings', fontsize=18)
    plt.ylabel('SCQ Final Score', fontsize=18)
    plt.title('SCQ Final Score vs. Developmental Milestones Embeddings', fontsize=20)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.xlim(0, 60)
    plt.savefig('GFMM_all_figures/GFMM_scq_dev_embeddings.png', bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    #scq_dev_embeddings(); exit()
    #developmental_milestones_validation(impute=False); exit()
    #individual_registration_validation(impute=False); exit()
    #vineland_validation(impute=False); exit()
    #scq_validation(impute=False); exit()
    get_figure_for_paper(only_sibs=True, impute=False); exit()
    run_main_LCA_model_nobms(ncomp=4, feature_group='combined')