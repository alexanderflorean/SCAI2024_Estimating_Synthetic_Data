"""
All measures used in the experiment are defined in this file.
"""

from math import sqrt
from copy import deepcopy

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sdmetrics.single_table import (BNLogLikelihood,
                                    GMLogLikelihood)
from sdmetrics.single_table.efficacy import (BinaryMLPClassifier,
                                             MulticlassMLPClassifier)
from sdmetrics.single_table.multi_column_pairs import (ContinuousKLDivergence,
                                                       DiscreteKLDivergence)
from sdmetrics.single_table.multi_single_column import CSTest, KSComplement
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils import timefunction
from preprocessing import create_preprocessing_pipeline

### Start - pMSE & S_pMSE
def compute_propensity(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    custom_metadata: dict,
    random_state=None,
    n_splits: int = 10
) -> dict:
    """
    Uses Logistic Regression from sklearn to compute the propensity
    of predicting the synthetic data, i.e. the probability that
    the model predicts the synthetic data.

    The method assumes the original and synthetic data has
    the same features.

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame
        classifier: scikit-learn classifier
        random_state: int, optional, default: None
            Controls the random state for train_test_split
        n_splits: int, default: 10
            The number of K-fold splits used when computing the stratified
            k-fold cross-validation.

    Returns:
        --------

        Dictionary with the following:
            'score': List of probabilities for predicting the samples are synthetic data
            'no': number of original data samples in the test data
            'ns': number of synthetic data samples in the test data
    """
    classifier = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        solver='lbfgs',
        max_iter=300
    )
    original_data = original_data.copy()
    synthetic_data = synthetic_data.copy()

    # Add target label 'S', i.e. if the sample is synthetic
    original_data["S"] = 0
    synthetic_data["S"] = 1

    # Combine original_data and synthetic_data
    combined_data = pd.concat([original_data, synthetic_data], axis=0)
    Z = combined_data.drop(columns="S")  # remove target label
    Z[custom_metadata['target']] = Z[custom_metadata['target']].astype(str)  # Set as string for preprocessor
    Y = combined_data["S"]  # target label

    # copy and set the target label as a categorical feature for the model and preprocessor
    meta = deepcopy(custom_metadata)
    if meta['categorical_features'] != None:
        meta['categorical_features'].append(custom_metadata['target'])
    else:
        meta["categorical_features"]= [meta['target']]

    no_test = []
    ns_test = []
    scores = []

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kfold.split(Z, Y):
        X_train, X_test = Z.iloc[train_index], Z.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        no_test.append(sum(y_test == 0) )  # number of original samples in the test data
        ns_test.append(sum(y_test == 1) )  # number of synthetic samples in the test data

        # create and fit the preprocessor to the training data
        preprocessor = create_preprocessing_pipeline(meta)
        preprocessor.fit(X_train)

        # Transform the data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # fit Logistic Regression model and compute propensity scores
        classifier.fit(X=X_train_transformed, y=y_train)

        # Extract probabilities for class 1 (synthetic) on X_test datapoints
        prob =  classifier.predict_proba(X_test_transformed)[:, 1]

        N = no_test[-1] + ns_test[-1]
        c = ns_test[-1] / N  # proportion of # synthetic samples in the test data

        pmse_score = (1/N) * sum((prob - c) ** 2) 
        scores.append(pmse_score)
    
    return {
        "score": np.mean(pmse_score) 
        ,"no": np.mean(no_test)
        ,"ns": np.mean(ns_test)
        }


@timefunction
def pmse(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    custom_metadata: dict,
) -> float:
    """
    Calculate the propensity mean-squared error.
    Algorithm implemented as described in DOI: 10.29012/jpc.v1i1.568

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame

    Returns:
        --------

        float: pMSE score
    """
    prop_dict = compute_propensity(original_data, synthetic_data, custom_metadata)

    propensity = prop_dict["score"]
    return propensity

    #n_o = prop_dict["no"]  # number of samples from original data
    #n_s = prop_dict["ns"]  # number of samples from synthetic data
    #N = n_o + n_s
    #c = n_s / N  # proportion of # synthetic samples in the test data

    #pmse_score = (1/N) * sum((propensity - c) ** 2) 

    #return pmse_score


@timefunction
def s_pmse(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    custom_metadata: dict,
) -> float:
    """
    Calculate the standardized propensity mean-squared error
        Algorithm implemented as described in DOI: 10.1111/rssa.12358

    Parameters:
        -----------

        original_data: pandas.DataFrame
        synthetic_data: pandas.DataFrame

    Returns:
        --------

        float: standardized pMSE score
    """

    # number of predictors of the combined dataset (i.e. target 'S')
    k = original_data.shape[1] + 1 

    prop_dict = compute_propensity(original_data, synthetic_data, custom_metadata)

    propensity = prop_dict["score"]

    n_o = prop_dict["no"]  # number of samples from original data
    n_s = prop_dict["ns"]  # number of samples from synthetic data
    N = n_o + n_s
    c = n_s / N  # proportion of # synthetic samples in the test data

    pmse_score = (1/N) * sum((propensity - c) ** 2) 

    # simulated results for the expected and standard deviation  pMSE value
    e_pmse = (k - 1) * (1 - c) ** 2 * c / N
    std_pmse = sqrt(2 * (k - 1)) * (1 - c) ** 2 * c / N

    s_pmse_score = (pmse_score - e_pmse) / std_pmse

    return s_pmse_score


### End - pMSE & S_pMSE


### Start - Cluster analysis
def calculate_cluster_weight(
    target_cluster_data_count: int,
    cluster_data_count: int,
    total_data_count: int,
    weight_type: str = "count",
) -> float:
    """
    Calculate the approximate standard error for the percentage of the
    target count in the provided cluster data.

    Parameters:
        -----------

        target_cluster_data_count: (int)
            The number of target data points in the cluster

        cluster_data_count: (int)
            The total number of data points in the cluster

        total_data_count: (int)
            The total number of data points across all clusters

        weight_type: (str), default="count"
            "approx": The approximate standard error for the given cluster over all samples
            "count": The the number of samples in the cluster divided by the total number of samples in the dataset

    Returns:
        --------

        float: the computed weight
    """
    if weight_type == "approx":
        percentage = target_cluster_data_count / cluster_data_count
        return np.sqrt((percentage * (1 - percentage)) / total_data_count)
    elif weight_type == "count":
        return cluster_data_count  # / total_data_count
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")


@timefunction
def cluster_metric(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    num_clusters: int,
    meta: dict,
    random_state=None,
) -> float:
    """
    Calculate the cluster analysis metric for the given original and synthetic datasets.

        Algorthim implemented as described in DOI: 10.29012/jpc.v1i1.568

    Parameters:
        -----------

        original_data:  pandas.DataFrame
            The original dataset.

        synthetic_data: pandas.DataFrame
            The synthetic dataset.

        num_clusters:   integer
            The number of clusters to use in the clustering.

        categorical_columns: list[int]
            List of indices of columns that contain categorical data

    Returns:
        --------

        float: The cluster analysis metric.
    """

    combined_data = pd.concat(
        [original_data, synthetic_data], axis=0, copy=True, ignore_index=True
    )

    dataset_meta = deepcopy(meta)
    cat_features = dataset_meta['categorical_features']

    # Set target label as categorical feature for preprocessing
    if cat_features != None:
        cat_features.append(dataset_meta['target'])
        dataset_meta['categorical_features'].append(dataset_meta['target'])
    else:
        cat_features = [dataset_meta["target"]]
        dataset_meta['categorical_features'] = [dataset_meta['target']]

    if dataset_meta['ordinal_features'] != None:
        cat_features.extend(list(dataset_meta['ordinal_features'].keys()))

    #convert target label to string for preprocessing
    combined_data[dataset_meta['target']] = combined_data[dataset_meta['target']].astype(str)
    #ensure there are no duplicate features
    cat_features = list(set(cat_features))

    if cat_features == None:
        # Contains only numerical cols, standardize the combined data
        scaled_combined_data = StandardScaler().fit_transform(combined_data)
        # Cluster the scaled data with sklearn.KMeans()
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(
            scaled_combined_data
        )  # expects shape (n_samples, n_features)
        cluster_labels = kmeans.labels_  

    elif len(cat_features) == combined_data.shape[1]:
        # TODO: encode features and use k-modes
        raise NotImplementedError(
            "Case: all columns are categorical, solution: use k-modes, however it is not implmented yet."
        )
    else:

        preprocessor = create_preprocessing_pipeline(dataset_meta)
        categorical_indices = [combined_data.columns.get_loc(col) for col in cat_features]
        transformed_data = preprocessor.fit_transform(combined_data)
        # Initialize KPrototypes with the desired number of clusters
        kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=random_state, n_jobs=-1, max_iter=100, n_init=100, verbose=1)
        # Fit the KPrototypes model to the transformed training data
        kproto.fit(transformed_data, categorical=categorical_indices)

        # Assign the cluster labels to the data points
        cluster_labels = kproto.labels_

    original_data_count = original_data.shape[0]  # number of samples in original data
    synthetic_data_count = synthetic_data.shape[0]  # number of samples in synthetic data
    total_data_count = original_data_count + synthetic_data_count

    constant_c = original_data_count / (original_data_count + synthetic_data_count)

    Uc = 0  # initialize Uc

    for cluster_id in range(num_clusters):
        original_cluster_data_count = np.sum(
            cluster_labels[:original_data_count] == cluster_id
        )
        synthetic_cluster_data_count = np.sum(
            cluster_labels[original_data_count:] == cluster_id
        )

        total_cluster_data_count = (
            original_cluster_data_count + synthetic_cluster_data_count
        )

        weight = calculate_cluster_weight(
            target_cluster_data_count=original_cluster_data_count,
            cluster_data_count=total_cluster_data_count,
            total_data_count=total_data_count,
            weight_type="count",
        )

        Uc += ( weight * ((original_cluster_data_count / total_cluster_data_count) - constant_c) ** 2)
    Uc /= num_clusters
    return Uc

### End - Cluster analysis

### Start - Likehood measures:  Looks at the likelihood of the synthetic data belonging to the real data.
@timefunction
def BNLogLikelihood_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    """Returns the average log of BN-Likelihood of all synthetic samples
    Range[-inf, 1]
    NOTE: this metric does not accept missing values.
    NOTE: SDMetrics only evaulates boolean, categorical columns using this measure. 
    
    For more details on the sdmetric implementation, see:
        https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/bnloglikelihood
    """

    return BNLogLikelihood.compute(
        real_data=original_data, 
        synthetic_data=synthetic_data, 
        metadata=sdv_metadata
    )

@timefunction
def GMLogLikelihood_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    """Uses multiple Gaussian mixtures models to learn the distribution of the real data to then returns
    the average likelihood for all samples on wether they belongs to the real data or not.
    Range[-inf,+inf], where -inf means it doesn't belong to the real data and +inf that it does.
    Meaning, the larger the value the more likely the sample/s belong to the real data and
    the smaller means the opposite.

    Meant for continous, numerical data and ignores the other incompatible column types,
    NOTE: this metric does not accept missing values.
    For more details see:
        https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/data-likelihood/gmlikelihood

    """
    return GMLogLikelihood.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=sdv_metadata
    )


### End - Likehood measures


### Start - Difference in Empirical distributions type measures
@timefunction
def ContinousKLDivergence_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    return ContinuousKLDivergence.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=sdv_metadata
    )

@timefunction
def DiscreteKLDivergence_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    return DiscreteKLDivergence.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=sdv_metadata
    )


@timefunction
def KSComplement_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement
    return KSComplement.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=sdv_metadata
    )


@timefunction
def CSTest_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, sdv_metadata: dict
) -> float:
    # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/cstest
    # Might be better to use the TVComplement measure, as CSTest is mentioned to have some issues
    return CSTest.compute(
        real_data=original_data, synthetic_data=synthetic_data, metadata=sdv_metadata
    )


### End - Difference in Empirical distributions type measures


### Start - Cross-Classification measure
def compare_and_filter_dataframes(main_df: pd.DataFrame, second_df: pd.DataFrame, categorical_cols):
    """
    Compares the unique values of specified categorical columns in two dataframes.
    If a value in the second dataframe doesn't exist in the main dataframe, it removes the rows with that value.

    Parameters:
        main_df: The main DataFrame object.
        second_df: The secondary DataFrame to compare and filter.
        categorical_cols: A list of column names that are categorical to which will be filtered in second_df.
    Returns: 
        The original and filtered dataframes.
    """
    second_df = second_df.copy()

    for col in categorical_cols:
        
        # Find unique values for the current column
        main_unique_values = set(main_df[col].unique())
        second_unique_values = set(second_df[col].unique())

        # Find values that exist in the second dataframe but not in the main dataframe
        diff_values = second_unique_values - main_unique_values

        # If there are such values, remove the corresponding rows from the second dataframe
        if diff_values:
            # print(f"To be dropped: {diff_values}")

            # isin(diff_values) will return a Boolean Series where True indicates that 
            # the row in second_df contains a value in diff_values. 
            # The ~ operator is used to negate this Boolean Series, 
            # so we are selecting rows where the value is NOT in diff_values.
            second_df = second_df[~second_df[col].isin(diff_values)]

    # Return the original (main) and filtered (second) dataframes
    return second_df

@timefunction
def CrossClassification_metric(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, dataset_metadata: dict, sdv_metadata: dict
) -> float:

    results = []
    # get categorical columns,
    target_features = dataset_metadata['categorical_features'] if dataset_metadata['categorical_features'] != None else []
    target_features.extend( list(dataset_metadata['ordinal_features'].keys()) if dataset_metadata['ordinal_features'] != None else [])
    target_features.append( dataset_metadata['target'])
    # ensures no duplicates exists
    target_features = list(set(target_features))

    # SDV MLPClassifiers uses onehotencoder to transform categorical data. 
    # However, they do not consider categorical values that exists in the test data but
    # not in the train data, which till inibit the measure.
    synthetic_data = compare_and_filter_dataframes(original_data, synthetic_data, target_features)

    for col in original_data.columns:
        if col in target_features:
            # identify if target label is binary or multiclass
            target_data = original_data[col]
            uniques = target_data.nunique()

            # run respective MLEfficacy algorithm which returns the F1-score
            if uniques == 2:
                # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/ml-efficacy-single-table/binary-classification
                efficacy = BinaryMLPClassifier.compute(
                    test_data=original_data,
                    train_data=synthetic_data,
                    metadata=sdv_metadata,
                    target=col,
                )
            else:
                # Documentation: https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/ml-efficacy-single-table/multiclass-classification
                efficacy = MulticlassMLPClassifier.compute(
                    test_data=original_data,
                    train_data=synthetic_data,
                    metadata=sdv_metadata,
                    target=col,
                )

            results.append(efficacy)

    return np.mean(results)
### End - Cross-Classification measure


### Computes all metrics defined above
def compute_pf_measures(
    original_data: pd.DataFrame
    ,synthetic_data: pd.DataFrame
    ,custom_metadata: dict     # custom metadata
    ,sdv_metadata: dict         # SDV metadata
    ,SD_id: str
) -> pd.DataFrame:

    # get number of clusters, using the original number of samples in the synthetic & original data,
    # and round it to an integer
    g_1= round((original_data.shape[0])    * 0.010)
    # g_2= round((original_data.shape[0])    * 0.020)
    g_values = {
        'G_1': g_1,
        #'G_2': g_2,
        }
    
    measures = {}

    measures["Dataset id"] = SD_id
    measures["SDG"] = SD_id.split('_')[0]

    result = pmse(original_data=original_data, synthetic_data=synthetic_data, custom_metadata=custom_metadata)
    measures["pMSE"] = result["score"]
    measures["pMSE_time"] = result["time"]

    #result = s_pmse(original_data=original_data, synthetic_data=synthetic_data, custom_metadata=custom_metadata)
    #measures["s_pMSE"] = result["score"]
    #measures["s_pMSE_time"] = result["time"]

    #for g in g_values:
    #    result = cluster_metric(
    #        original_data=original_data,
    #        synthetic_data=synthetic_data,
    #        num_clusters= g_values[g],
    #        meta=custom_metadata
    #    )
    #    measures[f"Cluster_{g}"] = result["score"]
    #    measures[f"Cluster_{g}_time"] = result["time"]
    #    #logg
    #    print(f"N_Clusters: {g_values[g]}")
    #    print(f"Value: {result['score']}, Time: {result['time']}")

    ## quickfix for Sdmetrics, some issues arise from specifying dtypes e.g. 'category', 'UInt8', etc.
    ## Mainly for SDMetrics, HyperTransformer, in SDMetrics/sdmetrics/utils.py
    ## seen in git-branch: 7754f13
    #o_data = pd.read_csv(f"../data/real/{custom_metadata['filename']}")
    #s_data = pd.read_csv(f"../data/synthetic/{SD_id}.csv")

    #result = BNLogLikelihood_metric(
    #    original_data=o_data, synthetic_data=s_data, sdv_metadata=sdv_metadata
    #)
    #measures["BNLogLikelihood"] = result["score"]
    #measures["BNLogLikelihood_time"] = result["time"]

    #result = GMLogLikelihood_metric(
    #    original_data=o_data, synthetic_data=s_data, sdv_metadata=sdv_metadata
    #)
    #measures["GMLogLikelihood"] = result["score"]
    #measures["GMLogLikelihood_time"] = result["time"]

    #result = ContinousKLDivergence_metric(o_data, s_data, sdv_metadata)
    #measures["ContinousKLDivergence"] = result["score"]
    #measures["ContinousKLDivergence_time"] = result["time"]

    #result = DiscreteKLDivergence_metric(o_data, s_data, sdv_metadata)
    #measures["DiscreteKLDivergence"] = result["score"]
    #measures["DiscreteKLDivergence_time"] = result["time"]

    #result = KSComplement_metric(o_data, s_data, sdv_metadata)
    #measures["KSComplement"] = result["score"]
    #measures["KSComplement_time"] = result["time"]

    #result = CSTest_metric(o_data, s_data, sdv_metadata)
    #measures["CSTest"] = result["score"]
    #measures["CSTest_time"] = result["time"]

    #result = CrossClassification_metric(o_data, s_data, dataset_metadata=custom_metadata, sdv_metadata=sdv_metadata)
    #measures["CrCl"] = result["score"]
    #measures["CrCl_time"] = result["time"]

    results_df = pd.DataFrame(data=measures, index=[0])
    return results_df
