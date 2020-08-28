
# Import Modules
from borg import *
import pandas as pd
import os
import pickle
import sklearn as sk
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

# Global Vars
pathto_data = '/app_io'
pathto_parambounds = os.path.join(pathto_data, 'phase_1_optimization', 'input', 'param_bounds.csv')
pathto_spacefeats = os.path.join(pathto_data, 'spatial_features_model', 'output')
pathto_damdata = os.path.join(pathto_data, 'phase_1_optimization', 'input', 'MA_U.csv')
pathto_deployidx = os.path.join(pathto_data, 'phase_1_optimization', 'input', 'deploy_idx.pkl')
pathto_output = os.path.join(pathto_data, 'phase_1_optimization', 'output')
parameter_names = ['N_length', 'N_width', 'n_estimators', 'min_samples_split', 'min_samples_leaf',
                   'min_weight_fraction_leaf', 'max_depth', 'max_features', 'max_leaf_nodes']
objective_names = ['accuracy', 'FPR', 'TPR', 'AUROCC']
feature_names = ['Dam Height (ft)', 'Dam Length (ft)', 'Reservoir Size (acre-ft)', 'Maximum Downstream Slope (%)',
                 'Downstream Houses', 'Downstream Population', 'Building Exposure ($1000)',
                 'Building Footprint (1000 sq. ft.)', 'Content Exposure ($1000)']
predicted_name = 'Hazard'
positive_lab = 'NH'


def discretizer(cont_var, disc_min, disc_max, step_size):
    """
    Round the continuous parameters from Borg (defined [0, 1]) to the rounded parameters of the simulation
    :param cont_var: float
                        Continuous variables
    :param disc_min: numeric
                        Minimum of the discretized parameter
    :param disc_max: numeric
                        Maximum of the discretized parameter
    :param step_size: numeric
                        Interval between discretizations
    :return: numeric
                        Discretized values
    """
    # Range Difference
    diff = disc_max - disc_min
    # Proportion of Continious Variable in Range
    dis = diff * cont_var
    # Round to Multiple of Step Size
    if step_size.is_integer():
        # Round to Multiple of Step Size
        disc_var = int(disc_min + step_size * round(dis / step_size))
    else:
        # Round to Multiple of Step Size
        disc_var = disc_min + step_size * round(dis / step_size)
    # Export
    return disc_var


def parameter_converter(params):
    """
    Wrapper function to Round all the continuous parameters from Borg (defined [0, 1]) to the rounded parameters
    of the simulation
    :param params: tuple
                    All the current Borg parameters
    :return: dict
                    All the corresponding discretized simulation parameters
    """
    # Convert to Dictionary
    param_dict = dict(zip(parameter_names, params))
    # Import Parameter Bounds
    pb = pd.read_csv(pathto_parambounds, index_col=0)
    # Parameter To Be Converted
    convert_params = list(pb.index[pb['stepsize'].notnull()])
    # Convert Parameters
    for i in convert_params:
        param_dict[i] = discretizer(param_dict[i], *tuple(pb.loc[i][['min', 'max', 'stepsize']]))
    # Export
    return param_dict


def get_features(param_dict):
    """
    Retrive the corresponding spatial and non-spatial feature values
    :param param_dict: dict
                        All the corresponding discretized simulation parameters
    :return: DataFrame
                        Spatial and non-spatial dam hazard feature values
    """
    # Import Spatial Features
    df_name = 'N_length_' + str(param_dict['N_length']) + '_N_width_' + str(param_dict['N_width'])
    space_feats = pd.read_hdf(os.path.join(pathto_spacefeats, 'spatial_feats.h5'), df_name)
    # Import Non-Spatial Features
    data = pd.read_csv(pathto_damdata)
    # Merge Features
    data = space_feats.join(data)
    data.index = data['RECORDID']
    # Rename Columns
    data = data.rename(index=str, columns={'HAZARD': predicted_name, 'DAM_HEIGHT': feature_names[0],
                                           'DAM_LENGTH': feature_names[1], 'NORMAL_STORAGE': feature_names[2],
                                           'Slope_max': feature_names[3], 'hous_sum': feature_names[4],
                                           'pop_sum': feature_names[5], 'buil_sum': feature_names[6],
                                           'foot_sum': feature_names[7], 'cont_sum': feature_names[8]})
    # Extract Features
    data = data[feature_names+[predicted_name]]
    # Export
    return data


def preprocessor(df):
    """
    Processing the feature values before classification
    :param df: DataFrame
                    Feature values
    :return: DataFrame
                    Processed feature values
    """
    # Combine Categories
    df = df.replace(to_replace=['L', 'S', 'H'], value=['NH', 'NH', 'H'])
    # Replace nans with median
    df = df.fillna(df.median())
    # Specify Objective
    y = df[predicted_name]
    # Shape Data
    X = np.array(df[feature_names])
    y = np.array(y)
    # Export
    return X, y


def objective_calculator(TP, FP, FN, TN):
    """
    Calculate each objective value
    :param TP: float64
                    True positive
    :param FP: float64
                    False positive
    :param FN: float64
                    False negative
    :param TN: float64
                    True negative
    :return: tuple
                    Objective values of accuracy, true positive rate, and false positive rate
    """
    # Calculate Objectives
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    #Export
    return accuracy, FPR, TPR


def machine_learning(X_train, X_test, y_train, y_test, ml_params, random_state):
    """
    Train and predict a random forest with a given set of features and hyperparameters (single fold)
    :param X_train: ndarray
                    Training features
    :param X_test: ndarray
                    Testing features
    :param y_train: ndarray
                    Training true values
    :param y_test: ndarray
                    Testing true values
    :param ml_params: dict
                    Machine learning hyperparameters
    :param random_state: int
                    Random state for the training of the random forest
    :return: tuple
                    Objective values (single fold)

    """
    # Create Model
    clf = sk.ensemble.RandomForestClassifier(n_jobs=-1, random_state=random_state,
                                             n_estimators=ml_params['n_estimators'],
                                             min_samples_split=ml_params['min_samples_split'],
                                             min_samples_leaf=ml_params['min_samples_leaf'],
                                             min_weight_fraction_leaf=ml_params['min_weight_fraction_leaf'],
                                             max_depth=ml_params['max_depth'],
                                             max_features=ml_params['max_features'],
                                             max_leaf_nodes=ml_params['max_leaf_nodes'])
    # Fit model to train data
    clf.fit(X_train, y_train)
    # Predicted Values
    y_pred = clf.predict(X_test)
    # Confusion Matrix
    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    # Extract True-Postive, False-Positive, True-Negative, and False-Negative
    TN, FN, FP, TP = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
    # Area Under ROC Curve
    y_score = clf.predict_proba(X_test)[:, 1]
    false_positive, true_positive, _ = roc_curve(y_test, y_score, pos_label=positive_lab)
    AUROC = auc(false_positive, true_positive)
    # Export
    return TP, FP, FN, TN, AUROC


def eval_params(ml_params, data):
    """
    Evaluate a set of random forest hyperparameters in five-fold cross validated fashion
    :param ml_params: dict
                        Machine learning hyperparameters
    :param data: dataframe
                        Spatial and non-spatial features
    :return: tuple
                        Objective values (cross-validated)
    """
    # Initialized Vars
    random_state = 1008
    folds_df = pd.DataFrame(columns=['TP', 'FP', 'FN', 'TN', 'AUROC'])
    # Process Data
    X, y = preprocessor(data)
    # Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(X, y):
        # Split the data for current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Resample the training data to deal with class imbalance
        method = ADASYN(random_state=random_state)
        X_train_res, y_train_res = method.fit_sample(X_train, y_train)
        # Train and evaluate current fold
        TP, FP, FN, TN, AUROCC = machine_learning(X_train_res, X_test, y_train_res, y_test, ml_params, random_state)
        # Place fold into DataFrame
        folds_df = folds_df.append({'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'AUROCC': AUROCC}, ignore_index=True)
    # Compute Average TP, FP, FN, TN, AUROC
    performance = folds_df.mean()
    # Compute Objectives
    accuracy, FPR, TPR = objective_calculator(performance['TP'], performance['FP'], performance['FN'], performance['TN'])
    # Export
    return accuracy, FPR, TPR, performance['AUROCC']


def simulation(*vars):
    """
    Evaluate a dam hazard potential 'simulation' with a given set of spatial parameters and random forest
    hyperparameters
    :param vars: tuple
                    A set of spatial and nonspatial parameters
    :return: tuple
                    Objective values
    """
    # Disctretize and Converter Parameters
    param_dict = parameter_converter(vars)
    # Get Features
    data = get_features(param_dict)
    # Get Deployment Indexes
    with open(pathto_deployidx, 'rb') as f:
        deploy_idx = pickle.load(f)
    # Omit Deployment Features
    data = data.drop(deploy_idx)
    # Evaluate Current Parameters
    accuracy, FPR, TPR, AUROC = eval_params(param_dict, data)
    # List Objectives
    objectives = [-accuracy, FPR, -TPR, -AUROC]
    # Export
    print(param_dict)
    print(objectives)
    return objectives


def main():
    # Import Parameter Bounds
    pb = pd.read_csv(pathto_parambounds, index_col=0)
    # Setup Optimization
    objectives_ls = []
    parameters_ls = []
    objs=4
    Configuration.seed(1008) #  Note, in paper this is repeated with five different seeds
    borg = Borg(numberOfVariables=pb.__len__(), numberOfObjectives=objs, numberOfConstraints=0, function=simulation)
    borg.setBounds(*tuple(list(pb.loc[i][['borg_min', 'borg_max']]) for i in parameter_names))
    epsi = 0.001
    borg.setEpsilons(*[epsi]*objs)
    print('Success: Setup Optimization')
    # Run Optimization
    result = borg.solve({'maxEvaluations': 100000, 'initialPopulationSize': 100}) 
    print('Success: Run Optimization')
    # Format Results
    for solution in result:
        objectives_ls.append(solution.getObjectives())
        parameters_ls.append(solution.getVariables())
    # Convert to Results to Dataframe
    parameters_df = pd.DataFrame.from_records(parameters_ls, columns=parameter_names)
    objectives_df = pd.DataFrame.from_records(objectives_ls, columns=objective_names)
    result_df = pd.concat([parameters_df, objectives_df], axis=1)
    # Export Data
    result_df.to_csv(os.path.join(pathto_output, 'results_raw.txt'), index=False, header=None, sep=' ', mode='a')
    print('Success: Export Data')
    return 0


if __name__ == '__main__':
    main()
