import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from stepmix.stepmix import StepMix
from stepmix.bootstrap import bootstrap
from sklearn.model_selection import GridSearchCV, ParameterGrid
from stepmix.utils import get_mixed_descriptor
from collections import defaultdict
from scipy import stats
import pickle as rick
from statsmodels.stats.multitest import multipletests

from latent_class_analysis import get_feature_enrichments, split_columns


def get_indicator_stats(ncomp=4, num_iter=200):
    '''Get Val LL over 200 iterations of the model. Save to pkl.'''

    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # Grid-Search CV
    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    val_log_likelihood = defaultdict(list)
    # get the Val LL of each of these hyperparameters
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=random_state_int)
        gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
        gs.fit(mixed_data, Z_p)
        results = pd.DataFrame(gs.cv_results_)
        results["Val_Log_Likelihood"] = results['mean_test_score']
        # for each n_components value, append the val log likelihood to the val_log_likelihood dict
        for n_components, vll in zip(results['param_n_components'], results['Val_Log_Likelihood']): # zipping two values
            val_log_likelihood[n_components].append(vll) # add one value from one iteration, and keep adding values from each iteration
        
    # save the val log likelihood dict to pkl
    with open(f'GFMM_validation_pickles/GFMM_val_log_likelihood_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(val_log_likelihood, f)

def sabic(model, X, Y=None):
    """Sample-Sized Adjusted BIC.

    References
    ----------
    Sclove SL. Application of model-selection criteria to some problems in multivariate analysis. Psychometrika. 1987;52(3):333–343.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    ssa_bic : float
    """
    n = X.shape[0]

    return -2 * model.score(X, Y) * n + model.n_parameters * np.log(
        n * ((n + 2) / 24)
    )

def c_aic(model, X, Y=None):
    """Consistent AIC.

    References
    ----------
    Bozdogan, H. 1987. Model selection and Akaike’s information criterion (AIC):
    The general theory and its analytical extensions. Psychometrika 52: 345–370.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    caic : float
        The lower the better.
    """
    n = X.shape[0]
    return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1)

def awe(model, X, Y=None):
        """Approximate weight of evidence. (Banfield & Raftery (1993))

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        awe : float
        """
        n = X.shape[0]
        return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1.5)

def get_AWE(num_iter=50):
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # Grid-Search CV
    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    awe_dict = defaultdict(list)
    # iterate through num_iter times, and for each n_components value, get the val log likelihood, AIC, BIC, and entropy num_iter times.
    # plot the mean of each of these values for each n_components value and plot the error bars
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        for g in ParameterGrid(grid):
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            awe_dict[g['n_components']].append(awe(model, mixed_data, Z_p))
    
    # save the awe dict to pkl
    with open(f'GFMM_all_figures/LCA_AWE_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(awe_dict, f)
    
    # print the mean of the awe values for each n_components value
    for n_components in grid['n_components']:
        print(f"Mean AWE for {n_components} components: {np.mean(awe_dict[n_components])}")
    
    # plot the mean of the awe values for each n_components value and plot the error bars
    plt.style.use('dark_background')
    awe_mean = [np.mean(awe_dict[n_components]) for n_components in grid['n_components']]
    awe_std = [np.std(awe_dict[n_components]) for n_components in grid['n_components']]
    plt.figure(figsize=(10, 8))
    plt.errorbar(grid['n_components'], awe_mean, yerr=awe_std, fmt='o', color='yellow')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('AWE', fontsize=20)
    plt.savefig(f'GFMM_all_figures/LCA_AWE_{num_iter}_iterations.png')
    plt.close()
        
def get_class_sizes(num_iter=50):
    '''get average class sizes and PP over num_iter iterations.'''
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # Grid-Search CV
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    smallest_n = defaultdict(list)
    smallest_prop = defaultdict(list)
    alcpp = defaultdict(list)
    # iterate through num_iter times, and for each n_components value, get the val log likelihood, AIC, BIC, and entropy num_iter times.
    # plot the mean of each of these values for each n_components value and plot the error bars
    for i in range(num_iter):
        random_state_int = np.random.randint(0, 1000)
        for g in ParameterGrid(grid):
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                    structural='covariate',
                    n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            pred = model.predict(mixed_data)
            # predict probabilities of each class
            class_probs = model.predict_proba(mixed_data)
            max_probs = np.max(class_probs, axis=1)
            alcpp[g['n_components']].append(np.mean(max_probs)) # get the mean of the max probs
            for class_size in np.unique(pred, return_counts=True)[1]:
                smallest_n[g['n_components']].append(class_size)
                smallest_prop[g['n_components']].append(class_size / len(pred))
          
    # save the smallest_n and smallest_prop dicts to pkl
    with open(f'GFMM_all_figures/LCA_smallest_n_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(smallest_n, f)
    with open(f'GFMM_all_figures/LCA_smallest_prop_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(smallest_prop, f)
    with open(f'GFMM_all_figures/LCA_alcpp_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(alcpp, f)
    
    # print the min of the smallest_n and smallest_prop values for each n_components value
    for n_components in grid['n_components']:
        print(f"Min smallest_n for {n_components} components: {np.min(smallest_n[n_components])}")
        print(f"Min smallest_prop for {n_components} components: {np.min(smallest_prop[n_components])}")
        print(f"Mean ALCPP for {n_components} components: {np.mean(alcpp[n_components])}")

def get_main_indicators(num_iter=200):
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_revised_no_bms.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # Grid-Search CV
    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    aic = defaultdict(list)
    bic = defaultdict(list)
    #entropy = defaultdict(list)
    ssabic = defaultdict(list)
    caic = defaultdict(list)
    # iterate through num_iter times, and for each n_components value, get the val log likelihood, AIC, BIC, and entropy num_iter times.
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
            #entropy[g['n_components']].append(model.relative_entropy(mixed_data, Z_p))
            caic[g['n_components']].append(c_aic(model, mixed_data, Z_p))
        
    # save the val log likelihood, AIC, BIC, and entropy dicts to pkl
    with open(f'GFMM_validation_pickles/GFMM_AIC_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(aic, f)
    with open(f'GFMM_validation_pickles/GFMM_BIC_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(bic, f)
    #with open(f'GFMM_validation_pickles/GFMM_entropy_{num_iter}_iterations.pkl', 'wb') as f:
    #    rick.dump(entropy, f)
    with open(f'GFMM_validation_pickles/GFMM_sabic_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(ssabic, f)
    with open(f'GFMM_validation_pickles/GFMM_caic_{num_iter}_iterations.pkl', 'wb') as f:
        rick.dump(caic, f)

def plot_main_indicators(num_iter=200):
    
    with open(f'GFMM_validation_pickles/GFMM_AIC_{num_iter}_iterations.pkl', 'rb') as f:
        aic = rick.load(f)
    with open(f'GFMM_validation_pickles/GFMM_BIC_{num_iter}_iterations.pkl', 'rb') as f:
        bic = rick.load(f)
    with open(f'GFMM_validation_pickles/LCA_entropy_50_iterations.pkl', 'rb') as f:
        entropy = rick.load(f)
    with open(f'GFMM_validation_pickles/GFMM_sabic_{num_iter}_iterations.pkl', 'rb') as f:
        ssabic = rick.load(f)
    with open(f'GFMM_validation_pickles/GFMM_caic_{num_iter}_iterations.pkl', 'rb') as f:
        caic = rick.load(f)
    with open(f'GFMM_validation_pickles/GFMM_val_log_likelihood_{num_iter}_iterations.pkl', 'rb') as f:
        val_log_likelihood = rick.load(f)

    grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    for i in range(1, 11):
        print(f"Mean AIC for {i} components: {np.mean(aic[i])}")
        print(f"Mean BIC for {i} components: {np.mean(bic[i])}")
        print(f"Mean SABIC for {i} components: {np.mean(ssabic[i])}")
        print(f"Mean CAIC for {i} components: {np.mean(caic[i])}")
        print(f"Mean Val Log Likelihood for {i} components: {np.mean(val_log_likelihood[i])}")
        print(f"Mean Entropy for {i} components: {np.mean(entropy[i])}")

    # make subplots to plot LL, AIC, BIC, SABIC, CAIC
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # LL
    val_log_likelihood_mean = [np.mean(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    val_log_likelihood_std = [np.std(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    ax[0, 0].errorbar(grid['n_components'], val_log_likelihood_mean, yerr=val_log_likelihood_std, fmt='o', color='dodgerblue', markersize=12)
    ax[0, 0].set_xlabel('Number of Components', fontsize=20)
    #ax[0, 0].set_ylabel('Val Log Likelihood', fontsize=20)
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
    #ax[0, 1].set_ylabel('AIC', fontsize=20)
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
    #ax[0, 2].set_ylabel('BIC', fontsize=20)
    ax[0, 2].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[0, 2].spines[axis].set_linewidth(1.5)
        ax[0, 2].spines[axis].set_color('black')
    ax[0, 2].axvline(x=4, color='black', linestyle='--')
    ax[0, 2].set_title('BIC', fontsize=20)

    # entropy
    '''
    entropy_mean = [np.mean(entropy[n_components]) for n_components in grid['n_components']]
    entropy_std = [np.std(entropy[n_components]) for n_components in grid['n_components']]
    ax[1, 0].errorbar(grid['n_components'], entropy_mean, yerr=entropy_std, fmt='o', color='tomato', markersize=12)
    ax[1, 0].set_xlabel('Number of Components', fontsize=20)
    ax[1, 0].set_ylabel('Entropy', fontsize=20)
    ax[1, 0].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 0].spines[axis].set_linewidth(1.5)
        ax[1, 0].spines[axis].set_color('black')
    ax[1, 0].set_title('Entropy', fontsize=20)
    '''

    # sabic
    sabic_mean = [np.mean(ssabic[n_components]) for n_components in grid['n_components']]
    sabic_std = [np.std(ssabic[n_components]) for n_components in grid['n_components']]
    ax[1, 0].errorbar(grid['n_components'], sabic_mean, yerr=sabic_std, fmt='o', color='gold', markersize=12)
    ax[1, 0].set_xlabel('Number of Components', fontsize=20)
    #ax[1, 0].set_ylabel('SABIC', fontsize=20)
    ax[1, 0].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 0].spines[axis].set_linewidth(1.5)
        ax[1, 0].spines[axis].set_color('black')
    ax[1, 0].axvline(x=4, color='black', linestyle='--')
    ax[1, 0].set_title('SABIC', fontsize=20)

    # caic
    caic_mean = [np.mean(caic[n_components]) for n_components in grid['n_components']]
    caic_std = [np.std(caic[n_components]) for n_components in grid['n_components']]
    ax[1, 1].errorbar(grid['n_components'], caic_mean, yerr=caic_std, fmt='o', color='tomato', markersize=12)
    ax[1, 1].set_xlabel('Number of Components', fontsize=20)
    #ax[1, 1].set_ylabel('CAIC', fontsize=20)
    ax[1, 1].grid(True)
    for axis in ['top','bottom','left','right']:
        ax[1, 1].spines[axis].set_linewidth(1.5)
        ax[1, 1].spines[axis].set_color('black')
    # vertical line at 4 components
    ax[1, 1].axvline(x=4, color='black', linestyle='--')
    ax[1, 1].set_title('CAIC', fontsize=20)

    # LMR-LRT
    with open(f'GFMM_validation_pickles/LCA_LMR_50_iterations_no_cv.pkl', 'rb') as f:
        p_vals = rick.load(f) # second run of LMR
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
    
    # save 
    plt.tight_layout()
    plt.savefig(f'GFMM_all_figures/GFMM_main_indicators_{num_iter}_iterations.png')
    plt.close()

    # plot val log likelihood
    val_log_likelihood_mean = [np.mean(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    val_log_likelihood_std = [np.std(val_log_likelihood[n_components]) for n_components in grid['n_components']]
    plt.figure(figsize=(8, 5))
    plt.errorbar(grid['n_components'], val_log_likelihood_mean, yerr=val_log_likelihood_std, fmt='o', color='dodgerblue', markersize=12)
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Val Log Likelihood', fontsize=21)
    # add grid
    plt.xticks(fontsize=16)
    for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)
            plt.gca().spines[axis].set_color('black')
    plt.grid(True)
    plt.savefig(f'GFMM_all_figures/GFMM_val_log_likelihood_{num_iter}_iterations.png', bbox_inches='tight')
    plt.close()

    # plot AIC and BIC on the same plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(grid['n_components'], aic_mean, yerr=aic_std, fmt='o', color='darkgreen', label='AIC')
    plt.errorbar(grid['n_components'], bic_mean, yerr=bic_std, fmt='o', color='darkorange', label='BIC')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('AIC/BIC', fontsize=20)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=16)
    for axis in ['top','bottom','left','right']:
            plt.gca().spines[axis].set_linewidth(1.5)
            plt.gca().spines[axis].set_color('black')
    plt.grid(True)
    plt.savefig(f'GFMM_all_figures/GFMM_AIC_BIC_{num_iter}_iterations.png', bbox_inches='tight')
    plt.close()


def scoring_on_bootstrap(estimator, X, y, rng, n_bootstrap=100):
    results_for_prevalence = defaultdict(list)
    for _ in range(n_bootstrap):
        rng = np.random.default_rng(seed=0)
        bootstrap_indices = rng.choice(
            np.arange(X.shape[0]), size=X.shape[0], replace=True
        )
        for key, value in scoring(
            estimator, X[bootstrap_indices], y[bootstrap_indices]
        ).items():
            results_for_prevalence[key].append(value)
    return pd.DataFrame(results_for_prevalence)

def blrt_nocv(bootstrap_draws=50):
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # get observed baseline p_vals
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    observed_ll = []
    for n_comp in grid['n_components']:
        model = StepMix(n_components=n_comp, measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=42)
        model.fit(mixed_data, Z_p)
        ll = model.score(mixed_data, Z_p)
        observed_ll.append(ll)
    ratios_observed = []
    for i in range(len(observed_ll) - 1):
        ratio = -2 * (observed_ll[i] - observed_ll[i + 1])
        ratios_observed.append(ratio)
    print(ratios_observed)
    '''
    rng = np.random.default_rng()
    p_vals = defaultdict(list)
    ratios = defaultdict(list)
    for i in range(bootstrap_draws):
        # bootstrap X: sample with replacement
        bootstrap_indices = rng.choice(
            np.arange(X.shape[0]), size=X.shape[0], replace=True
        )
        X_bootstrap = X.iloc[bootstrap_indices]
        Z_bootstrap = Z_p.iloc[bootstrap_indices]
        log_likelihoods = [] # reset list for each bootstrap sample

        mixed_data_boot, mixed_descriptor_boot = get_mixed_descriptor(
            dataframe=X_bootstrap,
            continuous=continuous_columns,
            binary=binary_columns,
            categorical=categorical_columns
        )
        
        for n_comp in grid['n_components']:
            model = StepMix(n_components=n_comp, measurement=mixed_descriptor_boot,
                        structural='covariate',
                        n_steps=1, random_state=42)
            model.fit(mixed_data_boot, Z_bootstrap)
            ll = model.score(mixed_data_boot, Z_bootstrap)
            log_likelihoods.append(ll)
        
        ratios_bootstrap = []
        for i in range(len(log_likelihoods) - 1):
            ratio = -2 * (log_likelihoods[i] - log_likelihoods[i + 1])
            ratios_bootstrap.append(ratio)
        ratios[2].append(ratios_bootstrap[0])
        ratios[3].append(ratios_bootstrap[1])
        ratios[4].append(ratios_bootstrap[2])
        ratios[5].append(ratios_bootstrap[3])
        ratios[6].append(ratios_bootstrap[4])

    with open(f'GFMM_all_figures/LCA_BLRT_ratios_{bootstrap_draws}_iterations_no_cv.pkl', 'wb') as f:
        rick.dump(ratios, f)
    '''
    with open(f'GFMM_all_figures/LCA_BLRT_ratios_{bootstrap_draws}_iterations_no_cv.pkl', 'rb') as f:
        ratios = rick.load(f)
    
    # compute p_vals per n_components value
    # defined as (s+1)/(B+1), where s is the number of bootstrap samples with a ratio greater than the observed ratio, and B is the number of bootstrap samples
    grid = {'n_components': [2, 3, 4, 5, 6]}
    bootstrap_p_vals = []
    for n_components in grid['n_components']:
        bootstrap_p_vals.append((len([ratio for ratio in ratios[n_components] if ratio > ratios_observed[n_components - 2]]) + 1) / (bootstrap_draws + 1))
    print(bootstrap_p_vals)

    # plot n_components vs. bootstrap p_vals
    plt.style.use('dark_background')
    plt.figure(figsize=(7, 6))
    plt.plot(grid['n_components'], bootstrap_p_vals, color='lightgreen')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('bootstrap p-value', fontsize=20)
    plt.xticks([2, 3, 4, 5, 6])
    plt.title('BLRT', fontsize=20)
    plt.savefig(f'GFMM_all_figures/LCA_BLRT_bootstrap_p_vals_{bootstrap_draws}_iterations_no_cv.png')
    plt.close()


def blrt(bootstrap_draws=50):
    
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'age_at_eval_years'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # get observed baseline p_vals
    model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1)
    
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    
    gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
    gs.fit(mixed_data, Z_p)
    results = pd.DataFrame(gs.cv_results_)
    results["Val_Log_Likelihood"] = results['mean_test_score']
    # access the val log likelihood for n_components=3 and n_components=4
    ll_2 = results['Val_Log_Likelihood'][0]
    ll_3 = results['Val_Log_Likelihood'][1]
    ll_4 = results['Val_Log_Likelihood'][2]
    ll_5 = results['Val_Log_Likelihood'][3]
    ll_6 = results['Val_Log_Likelihood'][4]
    ratio1 = -2 * (ll_2 - ll_3)
    ratio2 = -2 * (ll_3 - ll_4)
    ratio3 = -2 * (ll_4 - ll_5)
    ratio4 = -2 * (ll_5 - ll_6)
    p1 = stats.chi2.sf(ratio1, 1)
    p2 = stats.chi2.sf(ratio2, 1)
    p3 = stats.chi2.sf(ratio3, 1)
    p4 = stats.chi2.sf(ratio4, 1)
    ratios_observed = [ratio1, ratio2, ratio3, ratio4]
    print(ratios_observed)
    '''
    rng = np.random.default_rng()
    p_vals = defaultdict(list)
    ratios = defaultdict(list)
    for i in range(bootstrap_draws):
        # bootstrap X: sample with replacement
        bootstrap_indices = rng.choice(
            np.arange(X.shape[0]), size=X.shape[0], replace=True
        )
        X_bootstrap = X.iloc[bootstrap_indices]
        Z_bootstrap = Z_p.iloc[bootstrap_indices]

        mixed_data_boot, mixed_descriptor_boot = get_mixed_descriptor(
            dataframe=X_bootstrap,
            continuous=continuous_columns,
            binary=binary_columns,
            categorical=categorical_columns
        )
        
        # generate random seed
        random_state_int = np.random.randint(0, 1000)
        model = StepMix(measurement=mixed_descriptor_boot,
                        structural='covariate',
                        n_steps=1, random_state=random_state_int)
        
        gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
        gs.fit(mixed_data_boot, Z_bootstrap)
        results = pd.DataFrame(gs.cv_results_)
        results["Val_Log_Likelihood"] = results['mean_test_score']
        # access the val log likelihood for n_components=3 and n_components=4
        ll_2 = results['Val_Log_Likelihood'][0]
        ll_3 = results['Val_Log_Likelihood'][1]
        ll_4 = results['Val_Log_Likelihood'][2]
        ll_5 = results['Val_Log_Likelihood'][3]
        ll_6 = results['Val_Log_Likelihood'][4]

        ratio1 = -2 * (ll_2 - ll_3)
        ratio2 = -2 * (ll_3 - ll_4)
        ratio3 = -2 * (ll_4 - ll_5)
        ratio4 = -2 * (ll_5 - ll_6)
        p1 = stats.chi2.sf(ratio1, 1)
        p2 = stats.chi2.sf(ratio2, 1)
        p3 = stats.chi2.sf(ratio3, 1)
        p4 = stats.chi2.sf(ratio4, 1)
        p_vals[2].append(p1)
        p_vals[3].append(p2)
        p_vals[4].append(p3)
        p_vals[5].append(p4)
        ratios[2].append(ratio1)
        ratios[3].append(ratio2)
        ratios[4].append(ratio3)
        ratios[5].append(ratio4)
    
    with open(f'GFMM_all_figures/GFMM_BLRT_{bootstrap_draws}_iterations.pkl', 'wb') as f:
        rick.dump(p_vals, f)
    with open(f'GFMM_all_figures/GFMM_BLRT_ratios_{bootstrap_draws}_iterations.pkl', 'wb') as f:
        rick.dump(ratios, f)
    exit()
    '''
    with open(f'GFMM_validation_pickles/LCA_blrt_{bootstrap_draws}_iterations.pkl', 'rb') as f:
        p_vals = rick.load(f)
    with open(f'GFMM_validation_pickles/LCA_blrt_ratios_{bootstrap_draws}_iterations.pkl', 'rb') as f:
        ratios = rick.load(f)
    
    # compute p_vals per n_components value
    # defined as (s+1)/(B+1), where s is the number of bootstrap samples with a ratio greater than the observed ratio, and B is the number of bootstrap samples
    grid = {'n_components': [2, 3, 4, 5, 6]}
    bootstrap_p_vals = []
    for n_components in grid['n_components']:
        bootstrap_p_vals.append((len([ratio for ratio in ratios[n_components] if ratio > ratios_observed[n_components - 2]]) + 1) / (bootstrap_draws + 1))
    print(bootstrap_p_vals)

    # plot n_components vs. bootstrap p_vals
    plt.style.use('dark_background')
    plt.figure(figsize=(7, 6))
    plt.plot(grid['n_components'], bootstrap_p_vals, color='lightgreen')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('bootstrap p-value', fontsize=20)
    plt.xticks([2, 3, 4, 5, 6])
    plt.title('BLRT', fontsize=20)
    plt.savefig(f'GFMM_all_figures/LCA_blrt_bootstrap_p_vals_{bootstrap_draws}_iterations.png')
    plt.close()

    grid = {'n_components': [2, 3, 4, 5, 6]}
    # for each n_components value, plot the mean of the p-values and the error bars
    p_vals_mean = [np.mean(p_vals[n_components]) for n_components in grid['n_components']]
    p_vals_std = [np.std(p_vals[n_components]) for n_components in grid['n_components']]
    plt.figure(figsize=(10, 8))
    plt.errorbar(grid['n_components'], p_vals_mean, yerr=p_vals_std, fmt='o', color='lightgreen')
    plt.xlabel('Number of Components', fontsize=20)
    # cut off x axis at 5.5
    plt.xlim(1.5, 5.5)
    plt.title('BLRT', fontsize=20)
    plt.ylabel('p-value', fontsize=20)
    plt.savefig(f'GFMM_all_figures/GFMM_BLRT_{bootstrap_draws}_iterations.png')
    plt.close()

    # plot number of components vs. number of p_vals < 0.05
    num_p_vals = []
    for n_components in grid['n_components']:
        num_p_vals.append(len([p_val for p_val in p_vals[n_components] if p_val < 0.05]))
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    plt.plot(grid['n_components'], num_p_vals, color='lightcoral')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Number of p-values < 0.05', fontsize=20)
    plt.title('BLRT', fontsize=20)
    plt.savefig(f'GFMM_all_figures/GFMM_BLRT_num_p_vals_{bootstrap_draws}_iterations.png')   
    plt.close()

    # print mean p-values
    for n_components in grid['n_components']:
        print(f"Mean p-value for {n_components} components: {np.mean(p_vals[n_components])}")


def perform_VLMR_LRT_nocv(num_iter=50):
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI
    Z_p = datadf[['sex', 'age_at_eval_years']]
    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))
    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # get observed baseline p_vals
    model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=42)
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    p_vals = defaultdict(list)
    for i in range(n_iter):
        random_state_int = np.random.randint(0, 1000)
        log_likelihoods = []
        for g in ParameterGrid(grid): # iterate through n_components values
            model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                            structural='covariate',
                            n_steps=1, random_state=random_state_int)
            model.fit(mixed_data, Z_p)
            ll = model.score(mixed_data, Z_p)
            log_likelihoods.append(ll)

        # compute ratios using log likelihoods 
        ratio1 = -2 * (log_likelihoods[0] - log_likelihoods[1])
        ratio2 = -2 * (log_likelihoods[1] - log_likelihoods[2])
        ratio3 = -2 * (log_likelihoods[2] - log_likelihoods[3])
        ratio4 = -2 * (log_likelihoods[3] - log_likelihoods[4])
        ratio5 = -2 * (log_likelihoods[4] - log_likelihoods[5])
        p1 = stats.chi2.sf(ratio1, 1)
        p2 = stats.chi2.sf(ratio2, 1)
        p3 = stats.chi2.sf(ratio3, 1)
        p4 = stats.chi2.sf(ratio4, 1)
        p5 = stats.chi2.sf(ratio5, 1)
        p_vals[1].append(p1)
        p_vals[2].append(p2)
        p_vals[3].append(p3)
        p_vals[4].append(p4)
        p_vals[5].append(p5)
    
    with open(f'GFMM_all_figures/LCA_LMR_{n_iter}_iterations_no_cv.pkl', 'wb') as f:
        rick.dump(p_vals, f)

def plot_VLMR_LRT(n_iter=50):
    '''
    Plot results of VLMR-LRT in two ways.
    '''
    # read pickle
    with open(f'GFMM_validation_pickles/LCA_LMR_{n_iter}_iterations_no_cv.pkl', 'rb') as f:
        p_vals = rick.load(f) # second run of LMR

    # compute average p_val per n_components value
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    lrm_p_vals = []    
    for n_components in grid['n_components']:
        if n_components != 6: # skip because nan
            lrm_p_vals.append(np.mean(p_vals[n_components]))
    print(lrm_p_vals)

    # for each n_components value, plot the mean of the p-values and the error bars
    p_vals_mean = lrm_p_vals
    p_vals_std = [np.std(p_vals[n_components]) for n_components in grid['n_components'] if n_components != 6]
    plt.figure(figsize=(7.5, 5))
    plt.errorbar([2,3,4,5,6], p_vals_mean, yerr=p_vals_std, fmt='o', color='magenta', markersize=12)
    plt.xlabel('Number of Components', fontsize=20)
    plt.xticks([2,3,4,5,6])
    plt.xlim([1.5, 6.5])
    plt.title('VLMR-LRT', fontsize=20)
    plt.ylabel('p-value', fontsize=20)
    # make borders thicker
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.savefig(f'GFMM_all_figures/GFMM_LMR_LRT_{n_iter}_iterations.png', bbox_inches='tight')
    plt.close()

    # plot number of components vs. number of p_vals < 0.05
    num_p_vals = []
    for n_components in grid['n_components']:
        if n_components != 6: # skip because nan
            num_p_vals.append(len([p_val for p_val in p_vals[n_components] if p_val < 0.05]))
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    plt.plot([2,3,4,5,6], num_p_vals, color='lightcoral')
    plt.xticks([2,3,4,5,6])
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Number of p-values < 0.05', fontsize=20)
    plt.title('VLMR-LRT', fontsize=20)
    plt.savefig(f'GFMM_all_figures/GFMM_LMR_LRT_num_p_vals_{n_iter}_iterations_no_cv.png', bbox_inches='tight')
    plt.close()


def lmr_likelihood_ratio_test(n_iter=50):
    '''WITH CROSS-VALIDATION.'''
    
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'age_at_eval_years'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    # get observed baseline p_vals
    model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1)
    
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    p_vals = defaultdict(list)
    for i in range(n_iter):
        # get random_state int
        random_state_int = np.random.randint(0, 1000)
        
        model = StepMix(measurement=mixed_descriptor,
                        structural='covariate',
                        n_steps=1, random_state=random_state_int)
        
        gs = GridSearchCV(estimator=model, cv=3, param_grid=grid)
        gs.fit(mixed_data, Z_p)
        results = pd.DataFrame(gs.cv_results_)
        results["Val_Log_Likelihood"] = results['mean_test_score']
        # access the val log likelihood for n_components=3 and n_components=4
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
    
    with open(f'GFMM_validation_pickles/GFMM_LMR_LRT_{n_iter}_iterations.pkl', 'wb') as f:
        rick.dump(p_vals, f)
    
    with open(f'GFMM_validation_pickles/GFMM_LMR_LRT_{n_iter}_iterations.pkl', 'rb') as f:
        p_vals = rick.load(f)

    # compute average p_val per n_components value
    grid = {'n_components': [2, 3, 4, 5, 6]}
    lrm_p_vals = []    
    for n_components in grid['n_components']:
        lrm_p_vals.append(np.mean(p_vals[n_components]))
    print(lrm_p_vals)

    # for each n_components value, plot the mean of the p-values and the error bars
    #plt.style.use('dark_background')
    p_vals_mean = lrm_p_vals
    p_vals_std = [np.std(p_vals[n_components]) for n_components in grid['n_components']]
    plt.figure(figsize=(6, 8))
    plt.errorbar([1,2,3,4,5], p_vals_mean, yerr=p_vals_std, fmt='o', color='lightgreen')
    plt.xlabel('Number of Components', fontsize=20)
    plt.xticks([1,2,3,4,5])
    #plt.xlim([1.5, 5.5])
    plt.title('VLMR-LRT', fontsize=20)
    plt.ylabel('p-value', fontsize=20)
    plt.savefig(f'GFMM_all_figures/GFMM_LMR_{n_iter}_iterations.png')
    plt.close()

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
    plt.savefig(f'GFMM_all_figures/GFMM_LMR_num_p_vals_{n_iter}_iterations.png')
    plt.close()


def bf_approximation(bic0, bic1):
    sic0 = -0.05*bic0
    sic1 = -0.05*bic1
    return np.exp(bic0 - bic1)

def bayes_factor():
    '''
    IN DEVELOPMENT.
    '''
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_pheno_data/spark_data_all_revised_noadi.txt', sep='\t', index_col=0)  # RBSR+CBCL, no vineland, no ADI

    # get covariate data
    Z_p = datadf[['sex', 'age_at_eval_years']]

    X = datadf.drop(['sex', 'asd', 'age_at_eval_years', 'sped_y_n', 'left', 'right', 'ambi'],
                    axis=1)  # drop asd label and convert to np array
    
    continuous_columns, binary_columns, categorical_columns = split_columns(list(X.columns))

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=X,
        continuous=continuous_columns,
        binary=binary_columns,
        categorical=categorical_columns
    )

    bic_list = []
    bf = dict()
    grid = {'n_components': [1, 2, 3, 4, 5, 6]}
    # pick random state using random seed
    for g in ParameterGrid(grid):
        model = StepMix(n_components=g['n_components'], measurement=mixed_descriptor,
                structural='covariate',
                n_steps=1, random_state=42)
        model.fit(mixed_data, Z_p)
        bic_list.append(model.bic(mixed_data, Z_p))
    print(bic_list)
    for i in range(len(bic_list) - 1):
        bf[i] = bf_approximation(bic_list[i], bic_list[i+1])
    print(bf)

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    plt.plot(list(bf.keys()), list(bf.values()), color='blue')
    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Bayes Factor', fontsize=20)
    plt.savefig(f'GFMM_all_figures/LCA_bayes_factor.png')
    plt.close()


def posterior_prob_validation():
    
    datadf = pd.read_csv('/mnt/home/alitman/ceph/SPARK_Phenotype_Dataset/spark_5392_unimputed_cohort.txt', sep='\t', index_col=0)  # 5279 individuals, RBSR+CBCL, no vineland, no ADI, no BMS
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
    # take max posterior probability for each sample
    posterior_probs = np.max(posterior_probs, axis=1)
    mixed_data['mixed_pred'] = model.predict(mixed_data)
    labels = mixed_data['mixed_pred']
    labels.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_labels.csv')
    mixed_data['age'] = age

    # scramble each feature in the unlabeled data, retrain new model, and get posterior probabilities for each class
    copydf = datadf.copy()
    scrambled_data = copydf.apply(scramble_column)
    age = scrambled_data['age_at_eval_years']

    # get covariate data
    Z_p = scrambled_data[['sex', 'age_at_eval_years']]
    X = scrambled_data.drop(['sex', 'age_at_eval_years'],
                    axis=1)  # drop asd label and convert to np array
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
    # take max posterior probability for each sample
    posterior_probs_scram = np.max(posterior_probs_scram, axis=1)
    scrambled['mixed_pred'] = model_scram.predict(scrambled)
    labels_scram = scrambled['mixed_pred']
    labels_scram.to_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_scrambled_labels.csv') 

    # save posterior probabilities 
    with open(f'GFMM_validation_pickles/posterior_probs.pkl', 'wb') as f:
        rick.dump(posterior_probs, f)
    with open(f'GFMM_validation_pickles/posterior_probs_scram.pkl', 'wb') as f:
        rick.dump(posterior_probs_scram, f) 
    
    with open(f'GFMM_validation_pickles/posterior_probs.pkl', 'rb') as f:
        posterior_probs = rick.load(f)
    with open(f'GFMM_validation_pickles/posterior_probs_scram.pkl', 'rb') as f:
        posterior_probs_scram = rick.load(f)
    
    #labels = pd.read_csv('/mnt/home/alitman/ceph/GFMM_Labeled_Data/SPARK_5392_ninit_cohort_GFMM_labeled.csv', index_col=0)
    #labels = labels['mixed_pred']
    
    # plot posterior probabilities for each class for scrambled and unscrambled data
    plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, labels_scram, ncomp=4)


def plot_posterior_probs(posterior_probs, posterior_probs_scram, labels, labels_scram, ncomp, cohort='SPARK_5392_ninit'):
    # for each class, plot posterior probabilities for scrambled and unscrambled data
    posterior_probs_df = pd.DataFrame()
    posterior_probs_df['posterior_prob'] = posterior_probs
    posterior_probs_df['scrambled'] = False
    posterior_probs_df['class'] = labels.values
    posterior_probs_df_scram = pd.DataFrame()
    posterior_probs_df_scram['posterior_prob'] = posterior_probs_scram
    posterior_probs_df_scram['scrambled'] = True
    posterior_probs_df_scram['class'] = labels_scram.values
    posterior_probs_df = pd.concat([posterior_probs_df, posterior_probs_df_scram])

    # hypothesis testing for scrambled vs. unscrambled data
    # t-test for each class
    pvals = []

    for i in range(ncomp):
        pvals.append(stats.ttest_ind(posterior_probs_df[(posterior_probs_df['class'] == i) & (posterior_probs_df['scrambled'] == False)]['posterior_prob'],
                                    posterior_probs_df[(posterior_probs_df['class'] == i) & (posterior_probs_df['scrambled'] == True)]['posterior_prob'], 
                                    equal_var=False, alternative='greater')[1])
    print(pvals)
    p_vals_corrected = multipletests(pvals, method='fdr_bh')[1]
    print(p_vals_corrected)
    
    # plot
    plt.figure(figsize=(8, 8))
    sns.boxplot(data=posterior_probs_df, x='class', y='posterior_prob', hue='scrambled', palette='Dark2', showfliers=False, linewidth=2)
    plt.xlabel('Class', fontsize=21)
    plt.ylabel('Posterior Probability', fontsize=21)
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(1.5)
        plt.gca().spines[axis].set_color('black')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    plt.legend(title='Scrambled', title_fontsize='16', fontsize='14')
    plt.savefig(f'GFMM_all_figures/GFMM_{cohort}_posterior_probs_boxplot_{ncomp}comp.png', bbox_inches='tight')
    plt.clf()

    # plot just scrambled=0
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=posterior_probs_df[posterior_probs_df['scrambled'] == False], x='class', y='posterior_prob', palette='Dark2', showfliers=False)
    plt.savefig(f'GFMM_all_figures/GFMM_{cohort}_posterior_probs_boxplot_{ncomp}comp_no_scram.png')
    plt.clf()


def scramble_column(column):
    return np.random.permutation(column)


if __name__ == "__main__":
    # UNCOMMENT TO RUN DIFFERENT TESTS.
    #posterior_prob_validation(); exit()
    #run_main_LCA_model(ncomp=4)
    #lmr_nocv()
    #blrt_nocv(); exit()
    #plot_VLMR_LRT(); exit()
    #lmr_likelihood_ratio_test(); exit()
    #get_entropy()
    #get_AWE()
    #blrt(); exit()
    #bayes_factor()
    #get_class_sizes()
    #get_indicator_stats()
    #get_main_indicators()
    plot_main_indicators()    