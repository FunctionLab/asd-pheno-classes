import numpy as np
import pandas as pd
import pickle
from scipy.stats import ttest_ind, binomtest
from statsmodels.stats.multitest import multipletests


def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two groups.

    Parameters:
    - group1: array-like, containing data for group 1
    - group2: array-like, containing data for group 2

    Returns:
    - d: Cohen's d
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    d = mean_diff / pooled_std
    return d


def adjust_pvalues(p_values, method):
    """
    Pefrom multiple testing correction on p-values.
    """
    return multipletests(p_values, method=method)[1]


def split_columns(feature_subset):
    """
    Given list of features (strings), return stratified lists of 
    continuous, binary, and categorical lists containing the feature subset.
    """
    with open('data/binary_columns.pkl', 'rb') as f:
        binary_columns = pickle.load(f)

    with open('data/categorical_columns.pkl', 'rb') as f:
        categorical_columns = pickle.load(f)

    with open('data/continuous_columns.pkl', 'rb') as f:
        continuous_columns = pickle.load(f)

    continuous_columns_subset = []
    categorical_columns_subset = []
    binary_columns_subset = []
    for feature in feature_subset:
        if feature in continuous_columns:
            continuous_columns_subset.append(feature)
        elif feature in categorical_columns:
            categorical_columns_subset.append(feature)
        elif feature in binary_columns:
            binary_columns_subset.append(feature)

    return continuous_columns_subset, binary_columns_subset, categorical_columns_subset


def get_feature_enrichments(mixed_data, summarize=False):
    """
    Compute feature enrichments and depletions for each class.
    """
    feature_to_pval = dict()
    feature_sig_df_high = pd.DataFrame()
    feature_sig_df_low = pd.DataFrame()
    feature_vector = list()
    fold_enrichments = pd.DataFrame()
    mean_values = pd.DataFrame()

    # separate the four classes
    class0 = mixed_data[mixed_data['mixed_pred'] == 0]
    class1 = mixed_data[mixed_data['mixed_pred'] == 1]
    class2 = mixed_data[mixed_data['mixed_pred'] == 2]
    class3 = mixed_data[mixed_data['mixed_pred'] == 3]

    binary_features = []
    for feature in mixed_data.drop(['mixed_pred'],axis=1).columns:

        unique = mixed_data[feature].unique()
        if len(unique) == 2:  ## binary feature
            binary_features.append(feature)
            background_prob = int(np.sum(mixed_data[feature]))
            total = len(mixed_data[feature])
            total_in_class0 = len(class0[feature])
            subset_class0 = int(np.sum(class0[feature]))
            total_in_class1 = len(class1[feature])
            subset_class1 = int(np.sum(class1[feature]))
            total_in_class2 = len(class2[feature])
            subset_class2 = int(np.sum(class2[feature]))
            total_in_class3 = len(class3[feature])
            subset_class3 = int(np.sum(class3[feature]))

            # binomial hypothesis test in both directions
            sf0 = binomtest(
                subset_class0, n=total_in_class0, p=background_prob/total, 
                alternative='greater').pvalue
            sf0_less = binomtest(
                subset_class0, n=total_in_class0, p=background_prob/total, 
                alternative='less').pvalue

            sf1 = binomtest(
                subset_class1, n=total_in_class1, p=background_prob/total, 
                alternative='greater').pvalue
            sf1_less = binomtest(
                subset_class1, n=total_in_class1, p=background_prob/total, 
                alternative='less').pvalue

            sf2 = binomtest(
                subset_class2, n=total_in_class2, p=background_prob/total, 
                alternative='greater').pvalue
            sf2_less = binomtest(
                subset_class2, n=total_in_class2, p=background_prob/total, 
                alternative='less').pvalue

            sf3 = binomtest(
                subset_class3, n=total_in_class3, p=background_prob/total, 
                alternative='greater').pvalue
            sf3_less = binomtest(
                subset_class3, n=total_in_class3, p=background_prob/total, 
                alternative='less').pvalue

            # compute fold enrichment
            background = background_prob/total
            fe0 = (subset_class0/total_in_class0)/background
            fe1 = (subset_class1/total_in_class1)/background
            fe2 = (subset_class2/total_in_class2)/background
            fe3 = (subset_class3/total_in_class3)/background

            # compute mean values
            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            # store results for feature
            feature_to_pval[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_high[feature] = [sf0, sf1, sf2, sf3]
            feature_sig_df_low[feature] = [sf0_less, sf1_less, sf2_less, sf3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        elif len(unique) > 2:  ## continuous or categorical feature
            #  t-test hypothesis testing in both directions
            pval_class0 = ttest_ind(class0[feature],
                            pd.concat([class1[feature], class2[feature], class3[feature]],
                                    ignore_index=True, sort=False), equal_var=False,
                            alternative='greater').pvalue
            pval_class1 = ttest_ind(class1[feature],
                            pd.concat([class0[feature], class2[feature], class3[feature]],
                                    ignore_index=True, sort=False), equal_var=False,
                            alternative='greater').pvalue
            pval_class2 = ttest_ind(class2[feature],
                            pd.concat([class0[feature], class1[feature], class3[feature]],
                                    ignore_index=True, sort=False), equal_var=False,
                            alternative='greater').pvalue
            pval_class3 = ttest_ind(class3[feature],
                            pd.concat([class0[feature], class1[feature], class2[feature]],
                                    ignore_index=True, sort=False), equal_var=False,
                            alternative='greater').pvalue

            pval_class0_less = ttest_ind(class0[feature], pd.concat(
                [class1[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class1_less = ttest_ind(class1[feature], pd.concat(
                [class0[feature], class2[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class2_less = ttest_ind(class2[feature], pd.concat(
                [class0[feature], class1[feature], class3[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            pval_class3_less = ttest_ind(class3[feature], pd.concat(
                [class0[feature], class1[feature], class2[feature]], ignore_index=True, sort=False),
                                               equal_var=False, alternative='less').pvalue
            
            # compute Cohen's d
            total = mixed_data[feature]
            fe0 = cohens_d(class0[feature], total)
            fe1 = cohens_d(class1[feature], total)
            fe2 = cohens_d(class2[feature], total)
            fe3 = cohens_d(class3[feature], total)

            # compute mean values
            mean0 = np.mean(class0[feature])
            mean1 = np.mean(class1[feature])
            mean2 = np.mean(class2[feature])
            mean3 = np.mean(class3[feature])

            # store results for feature
            feature_to_pval[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_high[feature] = [pval_class0, pval_class1, pval_class2, pval_class3]
            feature_sig_df_low[feature] = [pval_class0_less, pval_class1_less, pval_class2_less, 
                                           pval_class3_less]
            feature_vector.append(feature)
            fold_enrichments[feature] = [fe0, fe1, fe2, fe3]
            mean_values[feature] = [mean0, mean1, mean2, mean3]

        else:
            continue
    
    # format results
    feature_sig_norm_high = pd.DataFrame(feature_sig_df_high, columns=feature_vector)
    feature_sig_norm_high['cluster'] = [0, 1, 2, 3]
    feature_sig_norm_low = pd.DataFrame(feature_sig_df_low, columns=feature_vector)
    feature_sig_norm_low['cluster'] = [0, 1, 2, 3]
    fold_enrichments['cluster'] = [0, 1, 2, 3]
    mean_values['cluster'] = [0, 1, 2, 3]
    pval_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)
    pval_classification_df = pd.DataFrame(columns=np.arange(4), index=mixed_data.columns)

    # perform multiple testing correction for each class and direction
    for tested_class in range(4):
        enriched_class_high = feature_sig_norm_high[
            feature_sig_norm_high['cluster'] == tested_class].drop('cluster',
                        axis=1).T.dropna(axis=0)
        adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
        enriched_class_high[f'{tested_class}_corrected'] = adjusted_pvals
        enriched_class_high_dict = enriched_class_high[
            enriched_class_high[f'{tested_class}_corrected'] < 0.05].loc[:,
                                   f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_high_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = 1  

        enriched_class_low = feature_sig_norm_low[
            feature_sig_norm_low['cluster'] == tested_class].drop('cluster', axis=1).T.dropna(
                axis=0)
        
        adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
        enriched_class_low[f'{tested_class}_corrected'] = adjusted_pvals_low
        enriched_class_low_dict = enriched_class_low[
            enriched_class_low[f'{tested_class}_corrected'] < 0.05].loc[:,
                                  f'{tested_class}_corrected'].sort_values(ascending=True).to_dict()

        for key, val in enriched_class_low_dict.items():
            pval_df.loc[key, tested_class] = val  
            pval_classification_df.loc[key, tested_class] = -1

    pval_classification_df = pval_classification_df.replace(np.nan, 0)

    if summarize:
        df = pd.DataFrame(columns=np.arange(8), index=mixed_data.columns)
        for i in range(4):
            enriched_class_high = feature_sig_norm_high[
                feature_sig_norm_high['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            enriched_class_low = feature_sig_norm_low[
                feature_sig_norm_low['cluster'] == i].drop('cluster', axis=1).T.dropna(axis=0)
            adjusted_pvals = list(adjust_pvalues(list(enriched_class_high.iloc[:, 0]), 'fdr_bh'))
            enriched_class_high[f'{i}_corrected'] = adjusted_pvals
            enriched_class_high_dict = enriched_class_high[
                enriched_class_high[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(
                    ascending=True).to_dict()
            for key, val in enriched_class_high_dict.items():
                df.loc[key, i] = val
            
            adjusted_pvals_low = list(adjust_pvalues(list(enriched_class_low.iloc[:, 0]), 'fdr_bh'))
            enriched_class_low[f'{i}_corrected'] = adjusted_pvals_low
            enriched_class_low_dict = enriched_class_low[
                enriched_class_low[f'{i}_corrected'] < 0.05].loc[:, f'{i}_corrected'].sort_values(
                    ascending=True).to_dict()
            for key, val in enriched_class_low_dict.items():
                df.loc[key, i+4] = val
        
        df = df[[0,4,1,5,2,6,3,7]] # rearrange columns
        df.columns = ['class0_enriched', 'class0_depleted', 'class1_enriched', 'class1_depleted', 
                      'class2_enriched', 'class2_depleted', 'class3_enriched', 'class3_depleted']
        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df_enriched_depleted = df.copy()

        # process fold_enrichments (FE + cohen's d)
        fold_enrichments = fold_enrichments.drop('cluster', axis=1).T.dropna(axis=0)
        fold_enrichments = pd.concat([fold_enrichments[0], fold_enrichments[0],
                                      fold_enrichments[1], fold_enrichments[1],
                                      fold_enrichments[2], fold_enrichments[2],
                                      fold_enrichments[3], fold_enrichments[3]], axis=1)
        fold_enrichments.columns = df.columns[1:]
        df_binary = df.copy()
        df_binary = df_binary.set_index('feature')
        df_binary = df_binary.applymap(lambda x: 1 if x < 0.05 else 0)
        df_binary = df_binary * fold_enrichments
        df_binary = df_binary.groupby(lambda x:x.split('_')[0], axis=1).sum()
        df_binary = df_binary.replace(0, np.nan)
        df_binary = df_binary.drop('mixed_pred', axis=0)
        fold_enrichments = df_binary.copy()
        fold_enrichments = fold_enrichments.reset_index()
        fold_enrichments.rename(columns={'index':'feature'}, inplace=True)

        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector, df_enriched_depleted, fold_enrichments
    else:
        return pval_classification_df, feature_sig_norm_high, feature_sig_norm_low, feature_vector
