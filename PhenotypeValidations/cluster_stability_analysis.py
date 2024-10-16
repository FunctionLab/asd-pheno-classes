from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor

from utils import split_columns, \
    get_feature_enrichments, \
        match_class_labels, \
        get_correlation


def run_mixture_model_on_phenotypes(iterations=2000, ncomp=4):
    # load unlabeled individual by feature matrix
    model_results = []
    log_likelihoods = []
    for i in range(iterations):
        print(f'Iteration {i}')
        datadf = pd.read_csv(
            '../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0
            )

        datadf = datadf.round()
        
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

        # seed = np.random.randint(1000)
        model = StepMix(
            n_components=ncomp, 
            measurement=mixed_descriptor,
            structural='covariate',
            n_steps=1,
            # random_state=seed
            n_init=1
            )

        # fit model and predict classes
        model.fit(mixed_data, Z_p)
        log_likelihoods.append(model.score(mixed_data, Z_p))
        mixed_data['mixed_pred'] = model.predict(mixed_data)
        model_results.append(mixed_data) 

    # sort log likelihoods
    log_likelihoods = np.array(log_likelihoods)
    ranked_indices = np.argsort(log_likelihoods)[::-1]
    ranked_models = [model_results[i] for i in ranked_indices]

    # load spark labels
    spark_labels = pd.read_csv(
        '../PhenotypeClasses/data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0
        )

    overall_model_correlations = []
    category_correlations = []
    ranked_ll = []
    overlaps = defaultdict(list)
    for i, mixed_data in enumerate(ranked_models):
        # match class labels
        spark_labels_copy = spark_labels.copy()
        mixed_data = match_class_labels(spark_labels_copy, mixed_data)
        
        r, seven_correlations = get_correlation(spark_labels_copy, mixed_data)
        if not np.isnan(r):
            overall_model_correlations.append(r)
            category_correlations.append(seven_correlations)
            ranked_ll.append(log_likelihoods[ranked_indices[i]])

            for k in range(4):
                for j in range(4):
                    overlap = len(mixed_data[(mixed_data['mixed_pred'] == k)].index & spark_labels_copy[spark_labels_copy['mixed_pred'] == j].index)
                    overlaps[f'{k}+{j}'].append(overlap / len(spark_labels_copy[spark_labels_copy['mixed_pred'] == j]))
    
    # make category correlations into a dataframe
    category_correlations = pd.DataFrame(category_correlations)
    
    # save results
    category_correlations.to_csv('data/stability_analysis_category_correlations.csv', index=False)
    
    overall_model_correlations = pd.DataFrame(overall_model_correlations)
    overall_model_correlations.to_csv('data/stability_analysis_overall_model_correlations.csv', index=False)

    ranked_ll = pd.DataFrame(ranked_ll)
    ranked_ll.to_csv('data/stability_analysis_ranked_LL.csv', index=False)

    # convert overlaps to dataframe where every column is a list (i+j combo)
    overlaps = pd.DataFrame(overlaps)
    overlaps.to_csv(f'data/stability_analysis_overlaps.csv', index=False)


def plot_correlations():
    category_correlations = pd.read_csv('data/stability_analysis_category_correlations.csv')
    overall_model_correlations = pd.read_csv('data/stability_analysis_overall_model_correlations.csv')
    ranked_ll = pd.read_csv('data/stability_analysis_ranked_LL.csv')
    overlaps = pd.read_csv('data/stability_analysis_overlaps.csv')

    # get indices of top 100 models with highest log likelihood
    top_x = 100
    log_likelihoods = np.array(ranked_ll['0'])
    ranked_indices = np.argsort(log_likelihoods)[::-1]
    category_correlations = [category_correlations.iloc[i] for i in ranked_indices[:top_x]]
    category_correlations = pd.DataFrame(category_correlations)
    overall_model_correlations = [overall_model_correlations.iloc[i] for i in ranked_indices[:top_x]]
    overall_model_correlations = pd.DataFrame(overall_model_correlations)
    overlaps = [overlaps.iloc[i] for i in ranked_indices[:top_x]]
    overlaps = pd.DataFrame(overlaps)

    # compute 95% CI for category correlations
    category_to_ci = {}
    for category in category_correlations.columns:
        category_to_ci[category] = st.t.interval(
            0.95, len(category_correlations[category])-1, loc=np.mean(category_correlations[category]), scale=st.sem(category_correlations[category])
            )

    # Extract the mean correlation and CIs into lists
    means = [np.mean(category_correlations[category]) for category in category_correlations.columns]
    ci_lower = [ci[0] for ci in category_to_ci.values()]
    ci_upper = [ci[1] for ci in category_to_ci.values()]

    # Calculate the error bars (distance from mean to upper bound)
    ci_error_upper = [upper - mean for upper, mean in zip(ci_upper, means)]
    ci_error_lower = [mean - lower for lower, mean in zip(ci_lower, means)]

    # plot category correlations as horizontal barplot with 95% CI
    features_to_visualize = ['anxiety/mood', 'attention', 'disruptive behavior', 'self-injury', 'social/communication', 'restricted/repetitive', 'developmental'] 
    plt.style.use('seaborn-v0_8-whitegrid')
    # plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.barplot(y=features_to_visualize, x=means, 
                color='rosybrown', ax=ax, width=0.5)
    ax.errorbar(x=means, y=features_to_visualize, 
            xerr=[ci_error_lower, ci_error_upper], fmt='none', 
            capsize=5, capthick=2, elinewidth=2, ecolor='black')

    plt.xlabel('Pearson r(SPARK, SPARK-iter)', fontsize=14)
    plt.ylabel('')
    plt.title(f'Model stability:\nCorr(GFMM, Top 100 initializations (n=2,000))', fontsize=16)
    plt.xlim([0,1])
    ax.set_yticklabels(features_to_visualize, fontsize=14)
    plt.xticks(fontsize=12)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.savefig('figures/stability_analysis_category_correlations.png', dpi=900, bbox_inches='tight')
    plt.close()

    # plot overlaps with confidence intervals
    means = []
    cis = []
    for overlap in overlaps.columns:
        means.append(np.mean(overlaps[overlap]))
        cis.append(st.t.interval(0.95, len(overlaps[overlap])-1, loc=np.mean(overlaps[overlap]), scale=st.sem(overlaps[overlap])))
    
    ci_lower = [ci[0] for ci in cis]
    ci_upper = [ci[1] for ci in cis]
    ci_error_upper = [upper - mean for upper, mean in zip(ci_upper, means)]
    ci_error_lower = [mean - lower for lower, mean in zip(ci_lower, means)]

    # Prepare annotations with mean ± CI
    annot = [f"{mean:.2f}±{(upper-lower)/2:.2f}" for mean, lower, upper in zip(means, ci_lower, ci_upper)]
    
    # replace 'nan' with 0.00 in annot
    annot = [x.replace('nan', '0.00') for x in annot]
    
    # reshape means, annot to be 4x4
    means = np.array(means).reshape(4, 4)
    annot = np.array(annot).reshape(4, 4)

    # plot heatmap
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(means, annot=annot, cmap='PuRd', cbar=True, fmt='', linewidths=0.5, linecolor='b')
    plt.xlabel('SPARK GFMM', fontsize=16)
    plt.ylabel('SPARK-iter GFMM', fontsize=16)
    plt.title(f'Proportion Overlaps, Top 100 initializations (n=2,000)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'figures/stability_analysis_overlaps.png', dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_mixture_model_on_phenotypes()
    plot_correlations()
