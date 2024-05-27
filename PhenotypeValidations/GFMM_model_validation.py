import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from sklearn.model_selection import GridSearchCV, ParameterGrid
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats
import pickle as rick
from statsmodels.stats.multitest import multipletests

from ../PhenotypeClasses/GFMM import get_feature_enrichments, split_columns
from utils import sabic, c_aic, awe, scramble_column


def compute_LL(ncomp=4, num_iter=200):
    datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  
    Z_p = datadf[['sex', 'age_at_eval_years']] # covariates
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    val_log_likelihood = defaultdict(list)
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=random_state_int)
        gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
        gs.fit(mixed_data, Z_p)
        results = pd.DataFrame(gs.cv_results_)
        results["Val_Log_Likelihood"] = results['mean_test_score']
        for n_components, vll in zip(results['param_n_components'], results['Val_Log_Likelihood']): # zipping two values
            val_log_likelihood[n_components].append(vll) # add one value from one iteration, and keep adding values from each iteration
        
    with open(f'pickles/GFMM_val_log_likelihood_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(val_log_likelihood, f)


def get_AWE(num_iter=50):
    datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  
    Z_p = datadf[['sex', 'age_at_eval_years']] # covariates
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    awe_dict = defaultdict(list)
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        for g in ParameterGrid(grid):
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                    structural='covariate', n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            awe_dict[g['n_components']].append(awe(model, mixed_data, Z_p))
    
    with open(f'figures/LCA_AWE_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(awe_dict, f)
    
    for n_components in grid['n_components']:
        print(f"Mean AWE for {n_components} components: {np.mean(awe_dict[n_components])}")
    
    plt.style.use('dark_background')
    awe_mean = [np.mean(awe_dict[n_components]) for n_components in grid['n_components']]
    awe_std = [np.std(awe_dict[n_components]) for n_components in grid['n_components']]
    plt.figure(figsize=(10, 8))
    plt.errorbar(grid['n_components'], awe_mean, yerr=awe_std, fmt='o', color='yellow')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('AWE', fontsize=20)
    plt.savefig(f'figures/LCA_AWE_{num_iter}_iterations.png')
    plt.close()
        

def get_class_sizes(num_iter=50):
    datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  
    Z_p = datadf[['sex', 'age_at_eval_years']] # covariates
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    smallest_n = defaultdict(list)
    smallest_prop = defaultdict(list)
    alcpp = defaultdict(list)
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        for g in ParameterGrid(grid):
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                    structural='covariate', n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            pred = model.predict(mixed_data)
            class_probs = model.predict_proba(mixed_data)
            max_probs = np.max(class_probs, axis=1)
            alcpp[g['n_components']].append(np.mean(max_probs))
            for class_size in np.unique(pred, return_counts=True)[1]:
                smallest_n[g['n_components']].append(class_size)
                smallest_prop[g['n_components']].append(class_size / len(pred))
          
    with open(f'pickles/LCA_smallest_n_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(smallest_n, f)
    with open(f'pickles/LCA_smallest_prop_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(smallest_prop, f)
    with open(f'pickles/LCA_alcpp_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(alcpp, f)
    
    for n_components in grid['n_components']:
        print(f"Min smallest_n for {n_components} components: {np.min(smallest_n[n_components])}")
        print(f"Min smallest_prop for {n_components} components: {np.min(smallest_prop[n_components])}")
        print(f"Mean ALCPP for {n_components} components: {np.mean(alcpp[n_components])}")


def compute_main_indicators(num_iter=200):
        datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  
    Z_p = datadf[['sex', 'age_at_eval_years']] # covariates
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    aic = defaultdict(list)
    bic = defaultdict(list)
    ssabic = defaultdict(list)
    entropy = defaultdict(list)
    caic = defaultdict(list)
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        for g in ParameterGrid(grid):
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            ssabic[g['n_components']].append(sabic(model, mixed_data, Z_p))
            aic[g['n_components']].append(model.aic(mixed_data, Z_p))
            bic[g['n_components']].append(model.bic(mixed_data, Z_p))
            entropy[g['n_components']].append(model.relative_entropy(mixed_data, Z_p))
            caic[g['n_components']].append(c_aic(model, mixed_data, Z_p))
        
    # save the val log likelihood, AIC, BIC, and entropy dicts to pkl
    with open(f'pickles/GFMM_AIC_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(aic, f)
    with open(f'pickles/GFMM_BIC_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(bic, f)
    with open(f'pickles/GFMM_entropy_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(entropy, f)
    with open(f'pickles/GFMM_sabic_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(ssabic, f)
    with open(f'pickles/GFMM_caic_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(caic, f)


def plot_main_indicators(num_iter=200):
    
    with open(f'pickles/GFMM_AIC_{num_iter}_iterations.pkl', 'rb') as f:
        aic = rick.load(f)
    with open(f'pickles/GFMM_BIC_{num_iter}_iterations.pkl', 'rb') as f:
        bic = rick.load(f)
    with open(f'pickles/LCA_entropy_50_iterations.pkl', 'rb') as f:
        entropy = rick.load(f)
    with open(f'pickles/GFMM_sabic_{num_iter}_iterations.pkl', 'rb') as f:
        ssabic = rick.load(f)
    with open(f'pickles/GFMM_caic_{num_iter}_iterations.pkl', 'rb') as f:
        caic = rick.load(f)
    with open(f'pickles/GFMM_val_log_likelihood_{num_iter}_iterations.pkl', 'rb') as f:
        val_log_likelihood = rick.load(f)

    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # subplots for LL, AIC, BIC, SABIC, CAIC, LMR-LRT
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # LL
    val_log_likelihood_mean = [np.mean(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    val_log_likelihood_std = [np.std(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    ax[0, 0].errorbar(grid['n_components'], val_log_likelihood_mean, yerr=val_log_likelihood_std, fmt='o', color='dodgerblue', markersize=12)
    ax[0, 0].set_xlabel('Number of Components', fontsize=20)
    ax[0, 0].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[0, 0].spines[axis].set_linewidth(1.5)
        ax[0, 0].spines[axis].set_color('black')
    ax[0, 0].axvline(x=4, color='black', linestyle='--')
    ax[0, 0].set_title('Val Log Likelihood', fontsize=20)

    # AIC
    aic_mean = [np.mean(aic[n_components]) for n_components in grid['n_components']]
    aic_std = [np.std(aic[n_components]) for n_components in grid['n_components']]
    ax[0, 1].errorbar(grid['n_components'], aic_mean, yerr=aic_std, fmt='o', color='darkgreen', markersize=12)
    ax[0, 1].set_xlabel('Number of Components', fontsize=20)
    ax[0, 1].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[0, 1].spines[axis].set_linewidth(1.5)
        ax[0, 1].spines[axis].set_color('black')
    ax[0, 1].axvline(x=4, color='black', linestyle='--')
    ax[0, 1].set_title('AIC', fontsize=20)
    
    # BIC
    bic_mean = [np.mean(bic[n_components]) for n_components in grid['n_components']]
    bic_std = [np.std(bic[n_components]) for n_components in grid['n_components']]
    ax[0, 2].errorbar(grid['n_components'], bic_mean, yerr=bic_std, fmt='o', color='darkorange', markersize=12)
    ax[0, 2].set_xlabel('Number of Components', fontsize=20)
    ax[0, 2].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[0, 2].spines[axis].set_linewidth(1.5)
        ax[0, 2].spines[axis].set_color('black')
    ax[0, 2].axvline(x=4, color='black', linestyle='--')
    ax[0, 2].set_title('BIC', fontsize=20)

    # SABIC
    sabic_mean = [np.mean(ssabic[n_components]) for n_components in grid['n_components']]
    sabic_std = [np.std(ssabic[n_components]) for n_components in grid['n_components']]
    ax[1, 0].errorbar(grid['n_components'], sabic_mean, yerr=sabic_std, fmt='o', color='gold', markersize=12)
    ax[1, 0].set_xlabel('Number of Components', fontsize=20)
    ax[1, 0].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 0].spines[axis].set_linewidth(1.5)
        ax[1, 0].spines[axis].set_color('black')
    ax[1, 0].axvline(x=4, color='black', linestyle='--')
    ax[1, 0].set_title('SABIC', fontsize=20)

    # CAIC
    caic_mean = [np.mean(caic[n_components]) for n_components in grid['n_components']]
    caic_std = [np.std(caic[n_components]) for n_components in grid['n_components']]
    ax[1, 1].errorbar(grid['n_components'], caic_mean, yerr=caic_std, fmt='o', color='tomato', markersize=12)
    ax[1, 1].set_xlabel('Number of Components', fontsize=20)
    ax[1, 1].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 1].spines[axis].set_linewidth(1.5)
        ax[1, 1].spines[axis].set_color('black')
    ax[1, 1].axvline(x=4, color='black', linestyle='--')
    ax[1, 1].set_title('CAIC', fontsize=20)

    # LMR-LRT
    with open(f'pickles/LCA_LMR_50_iterations_no_cv.pkl', 'rb') as f:
        p_vals = rick.load(f)
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    lrm_p_vals = []
    for n_components in grid['n_components']:
        if n_components != 6: # skip because nan
            lrm_p_vals.append(np.mean(p_vals[n_components]))
    p_vals_mean = lrm_p_vals
    p_vals_std = [np.std(p_vals[n_components]) for n_components in grid['n_components'] if n_components != 6]
    ax[1, 2].errorbar([2,3,4,5,6], p_vals_mean, yerr=p_vals_std, fmt='o', color='magenta', markersize=12)
    ax[1, 2].set_xlabel('Number of Components', fontsize=20)
    ax[1, 2].set_ylabel('p-value', fontsize=20)
    ax[1, 2].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 2].spines[axis].set_linewidth(1.5)
        ax[1, 2].spines[axis].set_color('black')
    ax[1, 2].set_xlim([1.5, 6.5])
    ax[1, 2].set_title('LMR-LRT', fontsize=20)
    ax[1, 2].axvline(x=4, color='black', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'figures/GFMM_main_indicators_{num_iter}_iterations.png')
    plt.close()


def lmr_likelihood_ratio_test(n_iter=50):
    datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  
    Z_p = datadf[['sex', 'age_at_eval_years']] # covariates
    X = datadf.drop(['sex', 'age_at_eval_years'], axis=1) 
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(measurement=mixed_descriptor, structural='covariate', n_steps=1)
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    p_vals = defaultdict(list)
    for i in range(n_iter):
        random_state_int = np.random.randint(0, 1000) # pick random state using random seed
        model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=random_state_int)
        gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
        gs.fit(mixed_data, Z_p)
        results = pd.DataFrame(gs.cv_results_)
        results["Val_Log_Likelihood"] = results['mean_test_score']
        ll_1 = results['Val_Log_Likelihood'][0]
        ll_2 = results['Val_Log_Likelihood'][1]
        ll_3 = results['Val_Log_Likelihood'][2]
        ll_4 = results['Val_Log_Likelihood'][3]
        ll_5 = results['Val_Log_Likelihood'][4]
        ll_6 = results['Val_Log_Likelihood'][5]
        ratio1 = -2 * (ll_1 - ll_2)
        ratio2 = -2 * (ll_2 - ll_3)
        ratio3 = -2 * (ll_3 - ll_4)
        ratio4 = -2 * (ll_4 - ll_5)
        ratio5 = -2 * (ll_5 - ll_6)
        p1 = stats.chi2.sf(ratio1, 1)
        p2 = stats.chi2.sf(ratio2, 1)
        p3 = stats.chi2.sf(ratio3, 1)
        p4 = stats.chi2.sf(ratio4, 1)
        p5 = stats.chi2.sf(ratio5, 1)
        p_vals[2].append(p1)
        p_vals[3].append(p2)
        p_vals[4].append(p3)
        p_vals[5].append(p4)
        p_vals[6].append(p5)
    
    with open(f'pickles/GFMM_LMR_LRT_{n_iter}_iterations.pkl', 'wb') as f:
        rick.dump(p_vals, f)
    
    # compute average p_val for each n_components value
    grid = {'n_components': [2, 3, 4, 5, 6]}
    lrm_p_vals = []    
    for n_components in grid['n_components']:
        lrm_p_vals.append(np.mean(p_vals[n_components]))

    # plot number of components vs. number of p_vals < 0.05
    num_p_vals = []
    for n_components in grid['n_components']:
        num_p_vals.append(len([p_val for p_val in p_vals[n_components] if p_val < 0.05]))
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    plt.plot([1,2,3,4,5], num_p_vals, color='lightcoral')
    plt.xticks([1,2,3,4,5])
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Number of p-values < 0.05', fontsize=20)
    plt.title('VLMR-LRT', fontsize=20)
    plt.savefig(f'figures/GFMM_LMR_num_p_vals_{n_iter}_iterations.png')
    plt.close()


def posterior_prob_validation():
    datadf = pd.read_csv('../PhenotypeClasses/data/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  # 5279 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
    datadf = datadf.round()
    age = datadf['age_at_eval_years']
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'age_at_eval_years'], 
                    axis=1)  
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    model = StepMix(n_components=4, measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200)

    model.fit(mixed_data, Z_p)
    posterior_probs = model.predict_proba(mixed_data)
    posterior_probs = np.max(posterior_probs, axis=1)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    labels.to_csv('data/SPARK_5392_labels.csv')
    mixed_data['age'] = age
    copydf = datadf.copy()
    scrambled_data = copydf.apply(scramble_column)
    age = scrambled_data['age_at_eval_years']

    Z_p = scrambled_data[['sex', 'age_at_eval_years']]
    X = scrambled_data.drop(['sex', 'age_at_eval_years'], axis=1)
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    scrambled, scrambled_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )
    model_scram = StepMix(n_components=4, measurement=scrambled_descriptor,
                    structural='covariate',
                    n_steps=1, n_init=200)
    model_scram.fit(scrambled, Z_p)
    posterior_probs_scram = model_scram.predict_proba(scrambled)
    posterior_probs_scram = np.max(posterior_probs_scram, axis=1)
    scrambled['mixed_pred'] = model_scram.predict(scrambled)
    labels_scram = scrambled['mixed_pred']
    labels_scram.to_csv('data/SPARK_5392_scrambled_labels.csv') 

    with open(f'pickles/posterior_probs.pkl', 'wb') as f:
        rick.dump(posterior_probs, f)
    with open(f'pickles/posterior_probs_scram.pkl', 'wb') as f:
        rick.dump(posterior_probs_scram, f) 
    
    plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, labels_scram, ncomp=4)


def plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, labels_scram, ncomp, cohort='SPARK_5392'):
    posterior_probs_df = pd.DataFrame()
    posterior_probs_df['posterior_prob'] = posterior_probs
    posterior_probs_df['scrambled'] = False
    posterior_probs_df['class'] = labels.values
    posterior_probs_df_scram = pd.DataFrame()
    posterior_probs_df_scram['posterior_prob'] = posterior_probs_scram
    posterior_probs_df_scram['scrambled'] = True
    posterior_probs_df_scram['class'] = labels_scram.values
    posterior_probs_df = pd.concat([posterior_probs_df, posterior_probs_df_scram])

    pvals = []
    for i in range(ncomp):
        pvals.append(stats.ttest_ind(posterior_probs_df[(posterior_probs_df['class'] == i) & (posterior_probs_df['scrambled'] == False)]['posterior_prob'],
                                    posterior_probs_df[(posterior_probs_df['class'] == i) & (posterior_probs_df['scrambled'] == True)]['posterior_prob'], 
                                    equal_var=False, alternative='greater')[1])
    p_vals_corrected = multipletests(pvals, method='fdr_bh')[1]
    
    plt.figure(figsize=(8, 8))
    sns.boxplot(data=posterior_probs_df, x='class', y='posterior_prob', hue='scrambled', palette='Dark2', showfliers=False, linewidth=2)
    plt.xlabel('Class', fontsize=21)
    plt.ylabel('Posterior Probability', fontsize=21)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    plt.legend(title='Scrambled', title_fontsize='16', fontsize='14')
    plt.savefig(f'figures/GFMM_{cohort}_posterior_probs_boxplot_{ncomp}comp.png', bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    # figure
    compute_LL()
    compute_main_indicators()
    lmr_likelihood_ratio_test()
    plot_main_indicators() 

    get_AWE()
    get_class_sizes()  

    #posterior_prob_validation() 