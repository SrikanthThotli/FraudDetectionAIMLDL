import datetime
import time

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import metrics
from xgboost import sklearn

from sharedFunctions import SharedMethod

common = SharedMethod()


class PerformanceMetrics:
    output_feature = "TX_FRAUD"

    input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                      'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                      'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                      'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                      'TERMINAL_ID_RISK_30DAY_WINDOW']

    def card_precision_top_k_day(self, df_day, top_k):
        # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
        # and sorts by decreasing order of fraudulent prediction
        df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(
            drop=False)

        # Get the top k most suspicious cards
        df_day_top_k = df_day.head(top_k)
        list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)

        # Compute precision top k
        card_precision_top_k = len(list_detected_compromised_cards) / top_k

        return list_detected_compromised_cards, card_precision_top_k

    def card_precision_top_k(self, predictions_df, top_k, remove_detected_compromised_cards=True):
        # Sort days by increasing order
        list_days = list(predictions_df['TX_TIME_DAYS'].unique())
        list_days.sort()

        # At first, the list of detected compromised cards is empty
        list_detected_compromised_cards = []

        card_precision_top_k_per_day_list = []
        nb_compromised_cards_per_day = []

        # For each day, compute precision top k
        for day in list_days:

            df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
            df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]

            # Let us remove detected compromised cards from the set of daily transactions
            df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False]

            nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))

            detected_compromised_cards, card_precision_top_k = self.card_precision_top_k_day(df_day, top_k)

            card_precision_top_k_per_day_list.append(card_precision_top_k)

            # Let us update the list of detected compromised cards
            if remove_detected_compromised_cards:
                list_detected_compromised_cards.extend(detected_compromised_cards)

        # Compute the mean
        mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

        # Returns precision top k per day as a list, and resulting mean
        return nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k

    def performance_assessment(self, predictions_df, output_feature='TX_FRAUD',
                               prediction_feature='predictions', top_k_list=[100],
                               rounded=True):
        AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
        AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])

        performances = pd.DataFrame([[AUC_ROC, AP]],
                                    columns=['AUC ROC', 'Average precision'])

        for top_k in top_k_list:
            _, _, mean_card_precision_top_k = self.card_precision_top_k(predictions_df, top_k)
            performances['Card Precision@' + str(top_k)] = mean_card_precision_top_k

        if rounded:
            performances = performances.round(3)

        return performances

    def performance_assessment_model_collection(self, fitted_models_and_predictions_dictionary,
                                                transactions_df,
                                                type_set='test',
                                                top_k_list=[100]):

        performances = pd.DataFrame()

        for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
            predictions_df = transactions_df

            predictions_df['predictions'] = model_and_predictions['predictions_' + type_set]

            performances_model = self.performance_assessment(predictions_df, output_feature='TX_FRAUD',
                                                             prediction_feature='predictions', top_k_list=top_k_list)
            performances_model.index = [classifier_name]

            performances = performances.append(performances_model)

        return performances

    def get_performances_train_test_sets(self, transactions_df, classifier,
                                         input_features, output_feature,
                                         start_date_training,
                                         delta_train=7, delta_delay=7, delta_test=7,
                                         top_k_list=[100],
                                         type_test="Test", parameter_summary=""):

        # Get the training and test sets
        (train_df, test_df) = common.get_train_test_set(transactions_df, start_date_training,
                                                        delta_train=delta_train,
                                                        delta_delay=delta_delay,
                                                        delta_test=delta_test)

        # Fit model
        start_time = time.time()
        model_and_predictions_dictionary = common.fit_model_and_get_predictions(classifier, train_df, test_df,
                                                                                input_features, output_feature)
        execution_time = time.time() - start_time

        # Compute fraud detection performances
        test_df['predictions'] = model_and_predictions_dictionary['predictions_test']
        performances_df_test = self.performance_assessment(test_df, top_k_list=top_k_list)
        performances_df_test.columns = performances_df_test.columns.values + ' ' + type_test

        train_df['predictions'] = model_and_predictions_dictionary['predictions_train']
        performances_df_train = self.performance_assessment(train_df, top_k_list=top_k_list)
        performances_df_train.columns = performances_df_train.columns.values + ' Train'

        performances_df = pd.concat([performances_df_test, performances_df_train], axis=1)

        performances_df['Execution time'] = execution_time
        performances_df['Parameters summary'] = parameter_summary

        return performances_df

    # Get the performance plot for a single performance metric
    def get_performance_plot(self, performances_df,
                             ax,
                             performance_metric,
                             expe_type_list=['Test', 'Train'],
                             expe_type_color_list=['#008000', '#2F4D7E'],
                             parameter_name="Tree maximum depth",
                             summary_performances=None):

        # expe_type_list is the list of type of experiments, typically containing 'Test', 'Train', or 'Valid'
        # For all types of experiments
        for i in range(len(expe_type_list)):

            # Column in performances_df for which to retrieve the data
            performance_metric_expe_type = performance_metric + ' ' + expe_type_list[i]

            # Plot data on graph
            ax.plot(performances_df['Parameters summary'], performances_df[performance_metric_expe_type],
                    color=expe_type_color_list[i], label=expe_type_list[i])

            # If performances_df contains confidence intervals, add them to the graph
            if performance_metric_expe_type + ' Std' in performances_df.columns:
                conf_min = performances_df[performance_metric_expe_type] \
                           - 2 * performances_df[performance_metric_expe_type + ' Std']
                conf_max = performances_df[performance_metric_expe_type] \
                           + 2 * performances_df[performance_metric_expe_type + ' Std']

                ax.fill_between(performances_df['Parameters summary'], conf_min, conf_max,
                                color=expe_type_color_list[i], alpha=.1)

        # If summary_performances table is present, adds vertical dashed bar for best estimated parameter
        if summary_performances is not None:
            best_estimated_parameter = \
                summary_performances[performance_metric][['Best estimated parameters ($k^*$)']].values[0]
            best_estimated_performance = float(
                summary_performances[performance_metric][['Validation performance']].values[0].split("+/-")[0])
            ymin, ymax = ax.get_ylim()
            ax.vlines(best_estimated_parameter, ymin, best_estimated_performance,
                      linestyles="dashed")

        # Set title, and x and y axes labels
        ax.set_title(performance_metric + '\n', fontsize=14)
        ax.set(xlabel=parameter_name, ylabel=performance_metric)

    # Get the performance plots for a set of performance metric
    def get_performances_plots(self, performances_df,
                               performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'],
                               expe_type_list=['Test', 'Train'], expe_type_color_list=['#008000', '#2F4D7E'],
                               parameter_name="Tree maximum depth",
                               summary_performances=None):

        # Create as many graphs as there are performance metrics to display
        n_performance_metrics = len(performance_metrics_list)
        fig, ax = plt.subplots(1, n_performance_metrics, figsize=(5 * n_performance_metrics, 4))

        # Plot performance metric for each metric in performance_metrics_list
        for i in range(n_performance_metrics):
            self.get_performance_plot(performances_df, ax[i], performance_metric=performance_metrics_list[i],
                                      expe_type_list=expe_type_list,
                                      expe_type_color_list=expe_type_color_list,
                                      parameter_name=parameter_name,
                                      summary_performances=summary_performances)

        ax[n_performance_metrics - 1].legend(loc='upper left',
                                             labels=expe_type_list,
                                             bbox_to_anchor=(1.05, 1),
                                             title="Type set")

        plt.subplots_adjust(wspace=0.5,
                            hspace=0.8)

    def repeated_holdout_validation(self, transactions_df, classifier,
                                    start_date_training,
                                    delta_train=7, delta_delay=7, delta_test=7,
                                    n_folds=4,
                                    sampling_ratio=0.7,
                                    top_k_list=[100],
                                    type_test="Test", parameter_summary=""):

        performances_df_folds = pd.DataFrame()

        start_time = time.time()

        for fold in range(n_folds):
            # Get the training and test sets
            (train_df, test_df) = common.get_train_test_set(transactions_df,
                                                            start_date_training,
                                                            delta_train=delta_train, delta_delay=delta_delay,
                                                            delta_test=delta_test,
                                                            )
            # Fit model
            model_and_predictions_dictionary = common.fit_model_and_get_predictions(classifier, train_df, test_df,
                                                                                    self.input_features,
                                                                                    self.output_feature)

            # Compute fraud detection performances
            test_df['predictions'] = model_and_predictions_dictionary['predictions_test']
            performances_df_test = self.performance_assessment(test_df, top_k_list=top_k_list)
            performances_df_test.columns = performances_df_test.columns.values + ' ' + type_test

            train_df['predictions'] = model_and_predictions_dictionary['predictions_train']
            performances_df_train = self.performance_assessment(train_df, top_k_list=top_k_list)
            performances_df_train.columns = performances_df_train.columns.values + ' Train'

            performances_df_folds = performances_df_folds.append(
                pd.concat([performances_df_test, performances_df_train], axis=1))

        execution_time = time.time() - start_time

        performances_df_folds_mean = performances_df_folds.mean()
        performances_df_folds_std = performances_df_folds.std(ddof=0)

        performances_df_folds_mean = pd.DataFrame(performances_df_folds_mean).transpose()
        performances_df_folds_std = pd.DataFrame(performances_df_folds_std).transpose()
        performances_df_folds_std.columns = performances_df_folds_std.columns.values + " Std"
        performances_df = pd.concat([performances_df_folds_mean, performances_df_folds_std], axis=1)

        performances_df['Execution time'] = execution_time

        performances_df['Parameters summary'] = parameter_summary

        return performances_df, performances_df_folds

    def prequential_validation(self, transactions_df, classifier,
                               start_date_training,
                               delta_train=7,
                               delta_delay=7,
                               delta_assessment=7,
                               n_folds=4,
                               top_k_list=[100],
                               type_test="Test", parameter_summary=""):

        performances_df_folds = pd.DataFrame()

        start_time = time.time()

        for fold in range(n_folds):
            start_date_training_fold = start_date_training - datetime.timedelta(days=fold * delta_assessment)

            # Get the training and test sets
            (train_df, test_df) = common.get_train_test_set(transactions_df,
                                                            start_date_training=start_date_training_fold,
                                                            delta_train=delta_train,
                                                            delta_delay=delta_delay,
                                                            delta_test=delta_assessment)

            # Fit model
            model_and_predictions_dictionary = common.fit_model_and_get_predictions(classifier, train_df, test_df,
                                                                                    self.input_features,
                                                                                    self.output_feature)

            # Compute fraud detection performances
            test_df['predictions'] = model_and_predictions_dictionary['predictions_test']
            performances_df_test = self.performance_assessment(test_df, top_k_list=top_k_list, rounded=False)
            performances_df_test.columns = performances_df_test.columns.values + ' ' + type_test

            train_df['predictions'] = model_and_predictions_dictionary['predictions_train']
            performances_df_train = self.performance_assessment(train_df, top_k_list=top_k_list, rounded=False)
            performances_df_train.columns = performances_df_train.columns.values + ' Train'

            performances_df_folds = performances_df_folds.append(
                pd.concat([performances_df_test, performances_df_train], axis=1))

        execution_time = time.time() - start_time

        performances_df_folds_mean = performances_df_folds.mean()
        performances_df_folds_std = performances_df_folds.std(ddof=0)

        performances_df_folds_mean = pd.DataFrame(performances_df_folds_mean).transpose()
        performances_df_folds_std = pd.DataFrame(performances_df_folds_std).transpose()
        performances_df_folds_std.columns = performances_df_folds_std.columns.values + " Std"
        performances_df = pd.concat([performances_df_folds_mean, performances_df_folds_std], axis=1)

        performances_df['Execution time'] = execution_time

        performances_df['Parameters summary'] = parameter_summary

        return performances_df, performances_df_folds

    def prequential_grid_search(self, transactions_df,
                                classifier,
                                input_features, output_feature,
                                parameters, scoring,
                                start_date_training,
                                n_folds=4,
                                expe_type='Test',
                                delta_train=7,
                                delta_delay=7,
                                delta_assessment=7,
                                performance_metrics_list_grid=['roc_auc'],
                                performance_metrics_list=['AUC ROC'],
                                n_jobs=-1):

        estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', classifier)]
        pipe = sklearn.pipeline.Pipeline(estimators)

        prequential_split_indices = self.prequentialSplit(transactions_df,
                                                          start_date_training=start_date_training,
                                                          n_folds=n_folds,
                                                          delta_train=delta_train,
                                                          delta_delay=delta_delay,
                                                          delta_assessment=delta_assessment)

        grid_search = sklearn.model_selection.GridSearchCV(pipe, parameters, scoring=scoring,
                                                           cv=prequential_split_indices, refit=False, n_jobs=n_jobs)

        X = transactions_df[input_features]
        y = transactions_df[output_feature]

        grid_search.fit(X, y)

        performances_df = pd.DataFrame()

        for i in range(len(performance_metrics_list_grid)):
            performances_df[performance_metrics_list[i] + ' ' + expe_type] = grid_search.cv_results_[
                'mean_test_' + performance_metrics_list_grid[i]]
            performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = grid_search.cv_results_[
                'std_test_' + performance_metrics_list_grid[i]]

        performances_df['Parameters'] = grid_search.cv_results_['params']
        performances_df['Execution time'] = grid_search.cv_results_['mean_fit_time']

        return performances_df

    def model_selection_wrapper(self, transactions_df,
                                classifier,
                                input_features, output_feature,
                                parameters,
                                scoring,
                                start_date_training_for_valid,
                                start_date_training_for_test,
                                n_folds=4,
                                delta_train=7,
                                delta_delay=7,
                                delta_assessment=7,
                                performance_metrics_list_grid=['roc_auc'],
                                performance_metrics_list=['AUC ROC'],
                                n_jobs=-1):
        # Get performances on the validation set using prequential validation
        performances_df_validation = self.prequential_grid_search(transactions_df, classifier,
                                                                  input_features, output_feature,
                                                                  parameters, scoring,
                                                                  start_date_training=start_date_training_for_valid,
                                                                  n_folds=n_folds,
                                                                  expe_type='Validation',
                                                                  delta_train=delta_train,
                                                                  delta_delay=delta_delay,
                                                                  delta_assessment=delta_assessment,
                                                                  performance_metrics_list_grid=performance_metrics_list_grid,
                                                                  performance_metrics_list=performance_metrics_list,
                                                                  n_jobs=n_jobs)

        # Get performances on the test set using prequential validation
        performances_df_test = self.prequential_grid_search(transactions_df, classifier,
                                                            input_features, output_feature,
                                                            parameters, scoring,
                                                            start_date_training=start_date_training_for_test,
                                                            n_folds=n_folds,
                                                            expe_type='Test',
                                                            delta_train=delta_train,
                                                            delta_delay=delta_delay,
                                                            delta_assessment=delta_assessment,
                                                            performance_metrics_list_grid=performance_metrics_list_grid,
                                                            performance_metrics_list=performance_metrics_list,
                                                            n_jobs=n_jobs)

        # Bind the two resulting DataFrames
        performances_df_validation.drop(columns=['Parameters', 'Execution time'], inplace=True)
        performances_df = pd.concat([performances_df_test, performances_df_validation], axis=1)

        # And return as a single DataFrame
        return performances_df

    def get_summary_performances(self,performances_df, parameter_column_name="Parameters summary"):

        # Three performance metrics
        metrics = ['AUC ROC', 'Average precision', 'Card Precision@100']
        performances_results = pd.DataFrame(columns=metrics)

        # Reset indices in case a subset of a performane DataFrame is provided as input
        performances_df.reset_index(drop=True, inplace=True)

        # Lists of parameters/performances that will be retrieved for the best estimated parameters
        best_estimated_parameters = []
        validation_performance = []
        test_performance = []

        # For each performance metric, get the validation and test performance for the best estimated parameter
        for metric in metrics:
            # Find the index which provides the best validation performance
            index_best_validation_performance = performances_df.index[
                np.argmax(performances_df[metric + ' Validation'].values)]

            # Retrieve the corresponding parameters
            best_estimated_parameters.append(
                performances_df[parameter_column_name].iloc[index_best_validation_performance])

            # Add validation performance to the validation_performance list (mean+/-std)
            validation_performance.append(
                str(round(performances_df[metric + ' Validation'].iloc[index_best_validation_performance], 3)) +
                '+/-' +
                str(round(performances_df[metric + ' Validation' + ' Std'].iloc[index_best_validation_performance], 2))
            )

            # Add test performance to the test_performance list (mean+/-std)
            test_performance.append(
                str(round(performances_df[metric + ' Test'].iloc[index_best_validation_performance], 3)) +
                '+/-' +
                str(round(performances_df[metric + ' Test' + ' Std'].iloc[index_best_validation_performance], 2))
            )

        # Add results to the performances_results DataFrame
        performances_results.loc["Best estimated parameters"] = best_estimated_parameters
        performances_results.loc["Validation performance"] = validation_performance
        performances_results.loc["Test performance"] = test_performance

        # Lists of parameters/performances that will be retrieved for the optimal parameters
        optimal_test_performance = []
        optimal_parameters = []

        # For each performance metric, get the performance for the optimal parameter
        for metric in ['AUC ROC Test', 'Average precision Test', 'Card Precision@100 Test']:
            # Find the index which provides the optimal performance
            index_optimal_test_performance = performances_df.index[np.argmax(performances_df[metric].values)]

            # Retrieve the corresponding parameters
            optimal_parameters.append(performances_df[parameter_column_name].iloc[index_optimal_test_performance])

            # Add test performance to the test_performance list (mean+/-std)
            optimal_test_performance.append(
                str(round(performances_df[metric].iloc[index_optimal_test_performance], 3)) +
                '+/-' +
                str(round(performances_df[metric + ' Std'].iloc[index_optimal_test_performance], 2))
            )

        # Add results to the performances_results DataFrame
        performances_results.loc["Optimal parameters"] = optimal_parameters
        performances_results.loc["Optimal test performance"] = optimal_test_performance

        return performances_results

    #Get the performance plot for a single performance metric

    def get_model_selection_performance_plot(self,performances_df_dictionary,
                                             ax,
                                             performance_metric,
                                             ylim=[0, 1],
                                             model_classes=['Decision Tree',
                                                            'Logistic Regression',
                                                            'Random Forest',
                                                            'XGBoost']):
        (mean_performances_dictionary, std_performances_dictionary) = \
            self.model_selection_performances(performances_df_dictionary=performances_df_dictionary,
                                         performance_metric=performance_metric)

        # width of the bars
        barWidth = 0.3
        # The x position of bars
        r1 = np.arange(len(model_classes))
        r2 = r1 + barWidth
        r3 = r1 + 2 * barWidth

        # Create Default parameters bars (Orange)
        ax.bar(r1, mean_performances_dictionary['Default parameters'],
               width=barWidth, color='#CA8035', edgecolor='black',
               yerr=std_performances_dictionary['Default parameters'], capsize=7, label='Default parameters')

        # Create Best validation parameters bars (Red)
        ax.bar(r2, mean_performances_dictionary['Best validation parameters'],
               width=barWidth, color='#008000', edgecolor='black',
               yerr=std_performances_dictionary['Best validation parameters'], capsize=7,
               label='Best validation parameters')

        # Create Optimal parameters bars (Green)
        ax.bar(r3, mean_performances_dictionary['Optimal parameters'],
               width=barWidth, color='#2F4D7E', edgecolor='black',
               yerr=std_performances_dictionary['Optimal parameters'], capsize=7, label='Optimal parameters')

        # Set title, and x and y axes labels
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xticks(r2 + barWidth / 2)
        ax.set_xticklabels(model_classes, rotation=45, ha="right", fontsize=12)
        ax.set_title(performance_metric + '\n', fontsize=18)
        ax.set_xlabel("Model class", fontsize=16)
        ax.set_ylabel(performance_metric, fontsize=15)

    def get_model_selection_performances_plots(self,performances_df_dictionary,
                                               performance_metrics_list=['AUC ROC', 'Average precision',
                                                                         'Card Precision@100'],
                                               ylim_list=[[0.6, 0.9], [0.2, 0.8], [0.2, 0.35]],
                                               model_classes=['Decision Tree',
                                                              'Logistic Regression',
                                                              'Random Forest',
                                                              'XGBoost']):

        # Create as many graphs as there are performance metrics to display
        n_performance_metrics = len(performance_metrics_list)
        fig, ax = plt.subplots(1, n_performance_metrics, figsize=(5 * n_performance_metrics, 4))

        parameter_types = ['Default parameters', 'Best validation parameters', 'Optimal parameters']

        # Plot performance metric for each metric in performance_metrics_list
        for i in range(n_performance_metrics):
            self.get_model_selection_performance_plot(performances_df_dictionary,
                                                 ax[i],
                                                 performance_metrics_list[i],
                                                 ylim=ylim_list[i],
                                                 model_classes=model_classes
                                                 )

        ax[n_performance_metrics - 1].legend(loc='upper left',
                                             labels=parameter_types,
                                             bbox_to_anchor=(1.05, 1),
                                             title="Parameter type",
                                             prop={'size': 12},
                                             title_fontsize=12)

        plt.subplots_adjust(wspace=0.5,
                            hspace=0.8)