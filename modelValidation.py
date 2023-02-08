import datetime
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost

from featurization import FeatureEngineering
from performance import PerformanceMetrics
from sharedFunctions import SharedMethod

common = SharedMethod()
feat = FeatureEngineering()
perf = PerformanceMetrics()


class modelSimulator:

    def __init__(self):

        self.performances_df = None
        self.delta_valid = None
        self.input_features = None
        self.output_feature = None
        self.end_date_test = None
        self.start_date_test = None
        self.end_date_training = None
        self.delta_test = None
        self.delta_delay = None
        self.delta_train = None
        self.start_date_training = None
        self.tx_stats = None
        self.transactions_df = None
        self.END_DATE = None
        self.BEGIN_DATE = None
        self.DIR_INPUT = None

    def train_perf_test_perf(self):
        # Load data from the 2022-03-25 to the 2022-04-14

        self.DIR_INPUT = './simulated-data-transformed/'

        self.BEGIN_DATE = "2022-03-25"
        self.END_DATE = "2022-04-25"

        print("Load  files")
        self.transactions_df = common.read_from_files(self.DIR_INPUT, self.BEGIN_DATE, self.END_DATE)

        print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(self.transactions_df),
                                                                                       self.transactions_df.TX_FRAUD.sum()))

        self.tx_stats = feat.get_tx_stats(self.transactions_df, start_date_df="2022-04-01")

        # Training period
        self.start_date_training = datetime.datetime.strptime("2022-03-25", "%Y-%m-%d")
        self.delta_train = self.delta_delay = self.delta_test = 7

        self.end_date_training = self.start_date_training + datetime.timedelta(days=self.delta_train - 1)

        # Test period
        self.start_date_test = self.start_date_training + datetime.timedelta(days=self.delta_train + self.delta_delay)
        self.end_date_test = self.start_date_training + datetime.timedelta(
            days=self.delta_train + self.delta_delay + self.delta_test - 1)

        (self.train_df, self.test_df) = feat.get_train_test_set(self.transactions_df, self.start_date_training,
                                                                delta_train=7, delta_delay=7, delta_test=7)

        self.output_feature = "TX_FRAUD"

        self.input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                               'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                               'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                               'TERMINAL_ID_RISK_30DAY_WINDOW']
        # We first create a decision tree object. We will limit its depth to 2 for interpretability,
        # and set the random state to zero for reproducibility
        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)

        model_and_predictions_dictionary = common.fit_model_and_get_predictions(classifier, self.train_df, self.test_df,
                                                                                self.input_features,
                                                                                self.output_feature,
                                                                                scale=False)

        self.test_df['TX_FRAUD_PREDICTED'] = model_and_predictions_dictionary['predictions_test']

        predictions_df = self.test_df
        predictions_df['predictions'] = model_and_predictions_dictionary['predictions_test']

        perf.performance_assessment(predictions_df, top_k_list=[100])

        predictions_df['predictions'] = 0.5

        perf.performance_assessment(predictions_df, top_k_list=[100])

        classifiers_dictionary = {'Logistic regression': sklearn.linear_model.LogisticRegression(random_state=0),
                                  'Decision tree with depth of two': sklearn.tree.DecisionTreeClassifier(max_depth=2,
                                                                                                         random_state=0),
                                  'Decision tree - unlimited depth': sklearn.tree.DecisionTreeClassifier(
                                      random_state=0),
                                  'Random forest': sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=-1),
                                  'XGBoost': xgboost.XGBClassifier(random_state=0, n_jobs=-1),
                                  }

        fitted_models_and_predictions_dictionary = {}

        for classifier_name in classifiers_dictionary:
            model_and_predictions = common.fit_model_and_get_predictions(classifiers_dictionary[classifier_name],
                                                                         self.train_df,
                                                                         self.test_df,
                                                                         input_features=self.input_features,
                                                                         output_feature=self.output_feature)
            fitted_models_and_predictions_dictionary[classifier_name] = model_and_predictions

        # performances on test set
        df_performances = perf.performance_assessment_model_collection(fitted_models_and_predictions_dictionary,
                                                                       self.test_df,
                                                                       type_set='test',
                                                                       top_k_list=[100])

    def holdOutValidation(self):
        # Note: We load more data than three weeks, as the experiments in the next sections
        # will require up to three months of data

        # Load data from the 2022-02-11 to the 2022-05-14

        self.DIR_INPUT = 'simulated-data-transformed/'

        self.BEGIN_DATE = "2022-02-11"
        self.END_DATE = "2022-05-14"

        print("Load  files")
        self.transactions_df = common.read_from_files(self.DIR_INPUT, self.BEGIN_DATE, self.END_DATE)
        print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(self.transactions_df),
                                                                                       self.transactions_df.TX_FRAUD.sum()))

        self.output_feature = "TX_FRAUD"

        self.input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                               'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                               'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                               'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                               'TERMINAL_ID_RISK_30DAY_WINDOW']

        # Set the starting day for the training period, and the deltas
        self.start_date_training = datetime.datetime.strptime("2022-03-25", "%Y-%m-%d")
        self.delta_train = 7
        self.delta_delay = 7
        self.delta_test = 7

        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)

        self.performances_df = perf.get_performances_train_test_sets(self.transactions_df, classifier,
                                                                     self.input_features, self.output_feature,
                                                                     start_date_training=self.start_date_training,
                                                                     delta_train=self.delta_train,
                                                                     delta_delay=self.delta_delay,
                                                                     delta_test=self.delta_test,
                                                                     parameter_summary=2
                                                                     )

        list_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        self.performances_df = pd.DataFrame()

        for max_depth in list_params:
            classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)

            self.performances_df = self.performances_df.append(
                perf.get_performances_train_test_sets(self.transactions_df, classifier,
                                                      self.input_features, self.output_feature,
                                                      start_date_training=self.start_date_training,
                                                      delta_train=self.delta_train,
                                                      delta_delay=self.delta_delay,
                                                      delta_test=self.delta_test,
                                                      parameter_summary=max_depth
                                                      )
            )

        self.performances_df.reset_index(inplace=True, drop=True)

        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)

        self.delta_valid = self.delta_test

        start_date_training_with_valid = self.start_date_training + datetime.timedelta(days=-(
                self.delta_delay + self.delta_valid))

        performances_df_validation = perf.get_performances_train_test_sets(self.transactions_df,
                                                                           classifier,
                                                                           self.input_features, self.output_feature,
                                                                           start_date_training=start_date_training_with_valid,
                                                                           delta_train=self.delta_train,
                                                                           delta_delay=self.delta_delay,
                                                                           delta_test=self.delta_test,
                                                                           type_test='Validation',
                                                                           parameter_summary='2')

        list_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        performances_df_validation = pd.DataFrame()

        for max_depth in list_params:
            classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)

            performances_df_validation = performances_df_validation.append(
                perf.get_performances_train_test_sets(self.transactions_df,
                                                      classifier,
                                                      self.input_features, self.output_feature,
                                                      start_date_training=start_date_training_with_valid,
                                                      delta_train=self.delta_train,
                                                      delta_delay=self.delta_delay,
                                                      delta_test=self.delta_test,
                                                      type_test='Validation', parameter_summary=max_depth
                                                      )
            )

        performances_df_validation.reset_index(inplace=True, drop=True)

        performances_df_validation['AUC ROC Test'] = self.performances_df['AUC ROC Test']
        performances_df_validation['Average precision Test'] = self.performances_df['Average precision Test']
        performances_df_validation['Card Precision@100 Test'] = self.performances_df['Card Precision@100 Test']

        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)

        performances_df_repeated_holdout_summary, performances_df_repeated_holdout_folds = perf.repeated_holdout_validation(
            self.transactions_df, classifier,
            start_date_training=start_date_training_with_valid,
            delta_train=self.delta_train,
            delta_delay=self.delta_delay,
            delta_test=self.delta_test,
            n_folds=4,
            sampling_ratio=0.7,
            type_test="Validation", parameter_summary='2'
        )

        list_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        performances_df_repeated_holdout = pd.DataFrame()

        start_time = time.time()

        for max_depth in list_params:
            print("Computing performances for a decision tree with max_depth=" + str(max_depth))

            classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)

            performances_df_repeated_holdout = performances_df_repeated_holdout.append(
                perf.repeated_holdout_validation(
                    self.transactions_df, classifier,
                    start_date_training=start_date_training_with_valid,
                    delta_train=self.delta_train,
                    delta_delay=self.delta_delay,
                    delta_test=self.delta_test,
                    n_folds=4,
                    sampling_ratio=0.7,
                    type_test="Validation", parameter_summary=max_depth
                )[0]
            )

        performances_df_repeated_holdout.reset_index(inplace=True, drop=True)

        print("Total execution time: " + str(round(time.time() - start_time, 2)) + "s")

        performances_df_repeated_holdout['AUC ROC Test'] = self.performances_df['AUC ROC Test']
        performances_df_repeated_holdout['Average precision Test'] = self.performances_df['Average precision Test']
        performances_df_repeated_holdout['Card Precision@100 Test'] = self.performances_df['Card Precision@100 Test']

    def prequentialValidation(self):
        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=0)
        start_date_training_with_valid = self.start_date_training + datetime.timedelta(
            days=-(self.delta_delay + self.delta_valid))
        performances_df_prequential_summary, performances_df_prequential_folds = perf.prequential_validation(
            self.transactions_df, classifier,
            start_date_training=start_date_training_with_valid,
            delta_train=self.delta_train,
            delta_delay=self.delta_delay,
            delta_assessment=self.delta_valid,
            n_folds=4,
            type_test="Validation", parameter_summary='2'
        )

        list_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        start_time = time.time()

        performances_df_prequential = pd.DataFrame()

        for max_depth in list_params:
            print("Computing performances for a decision tree with max_depth=" + str(max_depth))

            classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)

            performances_df_prequential = performances_df_prequential.append(
                perf.prequential_validation(
                    self.transactions_df, classifier,
                    start_date_training=start_date_training_with_valid,
                    delta_train=self.delta_train,
                    delta_delay=self.delta_delay,
                    delta_assessment=self.delta_test,
                    n_folds=4,
                    type_test="Validation", parameter_summary=max_depth
                )[0]
            )

        performances_df_prequential.reset_index(inplace=True, drop=True)

        print("Total execution time: " + str(round(time.time() - start_time, 2)) + "s")

        performances_df_prequential['AUC ROC Test'] = self.performances_df['AUC ROC Test']
        performances_df_prequential['Average precision Test'] = self.performances_df['Average precision Test']
        performances_df_prequential['Card Precision@100 Test'] = self.performances_df['Card Precision@100 Test']

        list_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        start_time = time.time()

        n_folds = 4

        performances_df_prequential_test = pd.DataFrame()

        for max_depth in list_params:
            classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)

            performances_df_prequential_test = performances_df_prequential_test.append(
                perf.prequential_validation(
                    self.transactions_df, classifier,
                    start_date_training=self.start_date_training
                                        + datetime.timedelta(days=self.delta_test * (n_folds - 1)),
                    delta_train=self.delta_train,
                    delta_delay=self.delta_delay,
                    delta_assessment=self.delta_test,
                    n_folds=n_folds,
                    type_test="Test", parameter_summary=max_depth
                )[0]
            )

        performances_df_prequential_test.reset_index(inplace=True, drop=True)

        print("Total execution time: " + str(round(time.time() - start_time, 2)) + "s")

        performances_df_prequential['AUC ROC Test'] = performances_df_prequential_test['AUC ROC Test']
        performances_df_prequential['Average precision Test'] = performances_df_prequential_test[
            'Average precision Test']
        performances_df_prequential['Card Precision@100 Test'] = performances_df_prequential_test[
            'Card Precision@100 Test']
        performances_df_prequential['AUC ROC Test Std'] = performances_df_prequential_test['AUC ROC Test Std']
        performances_df_prequential['Average precision Test Std'] = performances_df_prequential_test[
            'Average precision Test Std']
        performances_df_prequential['Card Precision@100 Test Std'] = performances_df_prequential_test[
            'Card Precision@100 Test Std']

    def prequentialSplit(self, transactions_df,
                         start_date_training,
                         n_folds=4,
                         delta_train=7,
                         delta_delay=7,
                         delta_assessment=7):
        prequential_split_indices = []

        # For each fold
        for fold in range(n_folds):
            # Shift back start date for training by the fold index times the assessment period (delta_assessment)
            # (See Fig. 5)
            start_date_training_fold = start_date_training - datetime.timedelta(days=fold * delta_assessment)

            # Get the training and test (assessment) sets
            (train_df, test_df) = common.get_train_test_set(transactions_df,
                                                            start_date_training=start_date_training_fold,
                                                            delta_train=delta_train, delta_delay=delta_delay,
                                                            delta_test=delta_assessment)

            # Get the indices from the two sets, and add them to the list of prequential splits
            indices_train = list(train_df.index)
            indices_test = list(test_df.index)

            prequential_split_indices.append((indices_train, indices_test))

        return prequential_split_indices

    def card_precision_top_k_custom(self, y_true, y_pred, top_k, transactions_df):
        # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
        # (indices of the y_true vector)
        predictions_df = transactions_df.iloc[y_true.index.values].copy()
        predictions_df['predictions'] = y_pred

        # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
        nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k = \
            perf.card_precision_top_k(predictions_df, top_k)

        # Return the mean_card_precision_top_k
        return mean_card_precision_top_k

    def gridSearch(self):

        # Only keep columns that are needed as argument to the custom scoring function
        # (in order to reduce the serialization time of transaction dataset)
        transactions_df_scorer = self.transactions_df[['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']]

        # Make scorer using card_precision_top_k_custom
        card_precision_top_100 = sklearn.metrics.make_scorer(self.card_precision_top_k_custom,
                                                             needs_proba=True,
                                                             top_k=100,
                                                             transactions_df=transactions_df_scorer)

        # Estimator to use
        classifier = sklearn.tree.DecisionTreeClassifier()

        # Hyperparameters to test
        parameters = {'clf__max_depth': [2, 4], 'clf__random_state': [0]}

        # Scoring functions. AUC ROC and Average Precision are readily available from sklearn
        # with `auc_roc` and `average_precision`. Card Precision@100 was implemented with the make_scorer factory function.
        scoring = {'roc_auc': 'roc_auc',
                   'average_precision': 'average_precision',
                   'card_precision@100': card_precision_top_100
                   }

        # A pipeline is created to scale data before fitting a model
        estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', classifier)]
        pipe = sklearn.pipeline.Pipeline(estimators)

        # Indices for the prequential validation are obtained with the prequentialSplit function
        prequential_split_indices = self.prequentialSplit(self.transactions_df,
                                                          self.start_date_training_with_valid,
                                                          n_folds=self.n_folds,
                                                          delta_train=self.delta_train,
                                                          delta_delay=self.delta_delay,
                                                          delta_assessment=self.delta_valid)

        # Let us instantiate the GridSearchCV
        grid_search = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, scoring=scoring, \
                                                           cv=prequential_split_indices, refit=False, n_jobs=-1,
                                                           verbose=0)

        # And select the input features, and output feature
        X = self.transactions_df[self.input_features]
        y = self.transactions_df[self.output_feature]

        grid_search.fit(X, y)

        print("Finished CV fitting")
        grid_search.cv_results_

        performances_df = pd.DataFrame()

        expe_type = "Validation"

        performance_metrics_list_grid = ['roc_auc', 'average_precision', 'card_precision@100']
        performance_metrics_list = ['AUC ROC', 'Average precision', 'Card Precision@100']

        for i in range(len(performance_metrics_list_grid)):
            performances_df[performance_metrics_list[i] + ' ' + expe_type] = self.grid_search.cv_results_[
                'mean_test_' + performance_metrics_list_grid[i]]
            performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = grid_search.cv_results_[
                'std_test_' + performance_metrics_list_grid[i]]

        performances_df['Execution time'] = grid_search.cv_results_['mean_fit_time']

        performances_df['Parameters'] = list(grid_search.cv_results_['params'])

    def integration(self):

        start_time = time.time()

        classifier = sklearn.tree.DecisionTreeClassifier()

        parameters = {'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50], 'clf__random_state': [0]}

        scoring = {'roc_auc': 'roc_auc',
                   'average_precision': 'average_precision',
                   'card_precision@100': self.card_precision_top_100,
                   }

        performances_df_validation = perf.prequential_grid_search(
            self.transactions_df, classifier,
            self.input_features, self.output_feature,
            parameters, scoring,
            start_date_training=self.start_date_training_with_valid,
            n_folds=self.n_folds,
            expe_type='Validation',
            delta_train=self.delta_train,
            delta_delay=self.delta_delay,
            delta_assessment=self.delta_valid,
            performance_metrics_list_grid=self.performance_metrics_list_grid,
            performance_metrics_list=self.performance_metrics_list)

        print("Validation: Total execution time: " + str(round(time.time() - start_time, 2)) + "s")

        start_time = time.time()

        performances_df_test = perf.prequential_grid_search(
            self.transactions_df, classifier,
            self.input_features, self.output_feature,
            parameters, scoring,
            start_date_training=self.start_date_training + datetime.timedelta(
                days=(self.n_folds - 1) * self.delta_test),
            n_folds=self.n_folds,
            expe_type='Test',
            delta_train=self.delta_train,
            delta_delay=self.delta_delay,
            delta_assessment=self.delta_test,
            performance_metrics_list_grid=self.performance_metrics_list_grid,
            performance_metrics_list=self.performance_metrics_list)

        print("Test: Total execution time: " + str(round(time.time() - start_time, 2)) + "s")

        performances_df_validation.drop(columns=['Parameters', 'Execution time'], inplace=True)
        performances_df = pd.concat([performances_df_test, performances_df_validation], axis=1)

        # Use the max_depth as the label for plotting
        parameters_dict = dict(performances_df['Parameters'])
        max_depth = [parameters_dict[i]['clf__max_depth'] for i in range(len(parameters_dict))]
        performances_df['Parameters summary'] = max_depth

    def modelSelection(self):
        # Load data from the 2022-02-11 to the 2022-03-14

        self.DIR_INPUT = 'simulated-data-transformed/'

        self.BEGIN_DATE = "2022-02-11"
        self.END_DATE = "2022-03-14"

        print("Load  files")
        self.transactions_df = common.read_from_files(self.DIR_INPUT, self.BEGIN_DATE, self.END_DATE)
        print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(self.transactions_df),
                                                                                       self.transactions_df.TX_FRAUD.sum()))

        output_feature = "TX_FRAUD"

        input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                          'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                          'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                          'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                          'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                          'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                          'TERMINAL_ID_RISK_30DAY_WINDOW']
        # Number of folds for the prequential validation
        n_folds = 4

        # Set the starting day for the training period, and the deltas
        start_date_training = datetime.datetime.strptime("2022-03-25", "%Y-%m-%d")
        delta_train = delta_delay = delta_test = delta_valid = delta_assessment = 7

        start_date_training_for_valid = start_date_training + datetime.timedelta(days=-(delta_delay + delta_valid))
        start_date_training_for_test = start_date_training + datetime.timedelta(days=(n_folds - 1) * delta_test)

        # Only keep columns that are needed as argument to the custom scoring function
        # (in order to reduce the serialization time of transaction dataset)
        transactions_df_scorer = self.transactions_df[['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']]

        card_precision_top_100 = sklearn.metrics.make_scorer(self.card_precision_top_k_custom,
                                                             needs_proba=True,
                                                             top_k=100,
                                                             transactions_df=transactions_df_scorer)

        performance_metrics_list_grid = ['roc_auc', 'average_precision', 'card_precision@100']
        performance_metrics_list = ['AUC ROC', 'Average precision', 'Card Precision@100']

        scoring = {'roc_auc': 'roc_auc',
                   'average_precision': 'average_precision',
                   'card_precision@100': card_precision_top_100,
                   }

        # Define classifier
        classifier = sklearn.tree.DecisionTreeClassifier()

        # Set of parameters for which to assess model performances
        parameters = {'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50], 'clf__random_state': [0]}

        start_time = time.time()

        # Fit models and assess performances for all parameters
        performances_df = perf.model_selection_wrapper(self.transactions_df, classifier,
                                                       input_features, output_feature,
                                                       parameters, scoring,
                                                       start_date_training_for_valid,
                                                       start_date_training_for_test,
                                                       n_folds=n_folds,
                                                       delta_train=delta_train,
                                                       delta_delay=delta_delay,
                                                       delta_assessment=delta_assessment,
                                                       performance_metrics_list_grid=performance_metrics_list_grid,
                                                       performance_metrics_list=performance_metrics_list,
                                                       n_jobs=1)

        execution_time_dt = time.time() - start_time

        # Select parameter of interest (max_depth)
        parameters_dict = dict(performances_df['Parameters'])
        performances_df['Parameters summary'] = [parameters_dict[i]['clf__max_depth'] for i in
                                                 range(len(parameters_dict))]

        # Rename to performances_df_dt for model performance comparison at the end of this notebook
        performances_df_dt = performances_df

        summary_performances_dt = perf.get_summary_performances(performances_df_dt,
                                                                parameter_column_name="Parameters summary")

    def modelSelLogisticReg(self):
        classifier = sklearn.linear_model.LogisticRegression()

        parameters = {'clf__C': [0.1, 1, 10, 100], 'clf__random_state': [0]}

        start_time = time.time()

        performances_df = perf.model_selection_wrapper(self.transactions_df, classifier,
                                                       self.input_features, self.output_feature,
                                                       parameters, self.scoring,
                                                       self.start_date_training_for_valid,
                                                       self.start_date_training_for_test,
                                                       n_folds=self.n_folds,
                                                       delta_train=self.delta_train,
                                                       delta_delay=self.delta_delay,
                                                       delta_assessment=self.delta_assessment,
                                                       performance_metrics_list_grid=self.performance_metrics_list_grid,
                                                       performance_metrics_list=self.performance_metrics_list,
                                                       n_jobs=1)

        execution_time_lr = time.time() - start_time

        parameters_dict = dict(performances_df['Parameters'])
        performances_df['Parameters summary'] = [parameters_dict[i]['clf__C'] for i in range(len(parameters_dict))]

        # Rename to performances_df_lr for model performance comparison at the end of this notebook
        performances_df_lr = performances_df

        summary_performances_lr = perf.get_summary_performances(performances_df_lr,
                                                                parameter_column_name="Parameters summary")

    def modelSelRanForClass(self):
        classifier = sklearn.ensemble.RandomForestClassifier()

        # Note: n_jobs set to one for getting true execution times
        parameters = {'clf__max_depth': [5, 10, 20, 50], 'clf__n_estimators': [25, 50, 100],
                      'clf__random_state': [0], 'clf__n_jobs': [1]}

        start_time = time.time()

        performances_df = perf.model_selection_wrapper(self.transactions_df, classifier,
                                                       self.input_features, self.output_feature,
                                                       parameters, self.scoring,
                                                       self.start_date_training_for_valid,
                                                       self.start_date_training_for_test,
                                                       n_folds=self.n_folds,
                                                       delta_train=self.delta_train,
                                                       delta_delay=self.delta_delay,
                                                       delta_assessment=self.delta_assessment,
                                                       performance_metrics_list_grid=self.performance_metrics_list_grid,
                                                       performance_metrics_list=self.performance_metrics_list,
                                                       n_jobs=1)

        execution_time_rf = time.time() - start_time

        parameters_dict = dict(performances_df['Parameters'])
        performances_df['Parameters summary'] = [str(parameters_dict[i]['clf__n_estimators']) +
                                                 '/' +
                                                 str(parameters_dict[i]['clf__max_depth'])
                                                 for i in range(len(parameters_dict))]

        # Rename to performances_df_rf for model performance comparison at the end of this notebook
        performances_df_rf = performances_df

        summary_performances_rf = perf.get_summary_performances(performances_df_rf,
                                                                parameter_column_name="Parameters summary")

        performances_df_rf_fixed_number_of_trees = performances_df_rf[
            performances_df_rf["Parameters summary"].str.startswith("100")]

        summary_performances_fixed_number_of_trees = perf.get_summary_performances(
            performances_df_rf_fixed_number_of_trees,
            parameter_column_name="Parameters summary")

        perf.get_performances_plots(performances_df_rf_fixed_number_of_trees,
                                    performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'],
                                    expe_type_list=['Test', 'Validation'], expe_type_color_list=['#008000', '#FF0000'],
                                    parameter_name="Number of trees/Maximum tree depth",
                                    summary_performances=summary_performances_fixed_number_of_trees)

    def modelSelXgboost(self):
        classifier = xgboost.XGBClassifier()

        parameters = {'clf__max_depth': [3, 6, 9], 'clf__n_estimators': [25, 50, 100], 'clf__learning_rate': [0.1, 0.3],
                      'clf__random_state': [0], 'clf__n_jobs': [1], 'clf__verbosity': [0]}

        start_time = time.time()

        performances_df = perf.model_selection_wrapper(self.transactions_df, classifier,
                                                       self.input_features, self.output_feature,
                                                       parameters, self.scoring,
                                                       self.start_date_training_for_valid,
                                                       self.start_date_training_for_test,
                                                       n_folds=self.n_folds,
                                                       delta_train=self.delta_train,
                                                       delta_delay=self.delta_delay,
                                                       delta_assessment=self.delta_assessment,
                                                       performance_metrics_list_grid=self.performance_metrics_list_grid,
                                                       performance_metrics_list=self.performance_metrics_list,
                                                       n_jobs=1)

        execution_time_boosting = time.time() - start_time

        parameters_dict = dict(performances_df['Parameters'])
        performances_df['Parameters summary'] = [str(parameters_dict[i]['clf__n_estimators']) +
                                                 '/' +
                                                 str(parameters_dict[i]['clf__learning_rate']) +
                                                 '/' +
                                                 str(parameters_dict[i]['clf__max_depth'])
                                                 for i in range(len(parameters_dict))]

        # Rename to performances_df_xgboost for model performance comparison at the end of this notebook
        performances_df_xgboost = performances_df

        summary_performances_xgboost = perf.get_summary_performances(performances_df_xgboost,
                                                                     parameter_column_name="Parameters summary")

        performances_df_xgboost_fixed_number_of_trees = performances_df_xgboost[
            performances_df_xgboost["Parameters summary"].str.startswith("100/0.3")]

        summary_performances_fixed_number_of_trees = perf.get_summary_performances(
            performances_df_xgboost_fixed_number_of_trees, parameter_column_name="Parameters summary")

        perf.get_performances_plots(performances_df_xgboost_fixed_number_of_trees,
                                    performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'],
                                    expe_type_list=['Test', 'Validation'], expe_type_color_list=['#008000', '#FF0000'],
                                    parameter_name="Number of trees/Maximum tree depth",
                                    summary_performances=summary_performances_fixed_number_of_trees)

        performances_df_dictionary = {
            "Decision Tree": self.performances_df_dt,
            "Logistic Regression": self.performances_df_lr,
            "Random Forest": self.performances_df_rf,
            "XGBoost": performances_df_xgboost
        }

        perf.model_selection_performances(performances_df_dictionary,
                                          performance_metric='AUC ROC')

        perf.get_model_selection_performances_plots(performances_df_dictionary,
                                                    performance_metrics_list=['AUC ROC', 'Average precision',
                                                                              'Card Precision@100'])

        execution_times = [self.execution_time_dt, self.execution_time_lr,
                           self.execution_time_rf, execution_time_boosting]

        perf.fig_model_selection_execution_times_for_each_model_class

        classifier = xgboost.XGBClassifier()

        parameters = {'clf__max_depth': [3, 6, 9], 'clf__n_estimators': [25, 50, 100], 'clf__learning_rate': [0.1, 0.3],
                      'clf__random_state': [0], 'clf__n_jobs': [1], 'clf__n_verbosity': [0]}

        start_time = time.time()

        performances_df = perf.model_selection_wrapper(self.transactions_df, classifier,
                                                       self.input_features, self.output_feature,
                                                       parameters, self.scoring,
                                                       self.start_date_training_for_valid,
                                                       self.start_date_training_for_test,
                                                       n_folds=self.n_folds,
                                                       delta_train=self.delta_train,
                                                       delta_delay=self.delta_delay,
                                                       delta_assessment=self.delta_assessment,
                                                       performance_metrics_list_grid=self.performance_metrics_list_grid,
                                                       performance_metrics_list=self.performance_metrics_list,
                                                       type_search='random',
                                                       n_iter=10,
                                                       random_state=0,
                                                       n_jobs=1)

        execution_time_boosting_random = time.time() - start_time

        parameters_dict = dict(performances_df['Parameters'])
        performances_df['Parameters summary'] = [str(parameters_dict[i]['clf__n_estimators']) +
                                                 '/' +
                                                 str(parameters_dict[i]['clf__learning_rate']) +
                                                 '/' +
                                                 str(parameters_dict[i]['clf__max_depth'])
                                                 for i in range(len(parameters_dict))]

        # Rename to performances_df_xgboost_random for model performance comparison
        performances_df_xgboost_random = performances_df

        summary_performances_xgboost_random = perf.get_summary_performances(performances_df_xgboost_random,
                                                                            parameter_column_name="Parameters summary")

        performances_df_dictionary = {
            "Decision Tree": self.performances_df_dt,
            "Logistic Regression": self.performances_df_lr,
            "Random Forest": self.performances_df_rf,
            "XGBoost": self.performancself._df_xgboost,
            "XGBoost Random": performances_df_xgboost_random
        }

        execution_times = [self.execution_time_dt,
                           self.execution_time_lr,
                           self.execution_time_rf,
                           execution_time_boosting,
                           execution_time_boosting_random]

        filehandler = open('performances_model_selection.pkl', 'wb')
        pickle.dump((performances_df_dictionary, execution_times), filehandler)
        filehandler.close()
