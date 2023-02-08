import datetime
import pandas as pd


class FeatureEngineering:

    def is_weekend(self, tx_datetime):
        # Transform date into weekday (0 is Monday, 6 is Sunday)
        weekday = tx_datetime.weekday()
        # Binary value: 0 if weekday, 1 if weekend
        is_weekend = weekday >= 5

        return int(is_weekend)

    def is_night(self, tx_datetime):
        # Get the hour of the transaction
        tx_hour = tx_datetime.hour
        # Binary value: 1 if hour less than 6, and 0 otherwise
        is_night = tx_hour <= 6

        return int(is_night)

    def get_customer_spending_behaviour_features(self, customer_transactions, windows_size_in_days=[1, 7, 30]):
        # Let us first order transactions chronologically
        customer_transactions = customer_transactions.sort_values('TX_DATETIME')

        # The transaction date and time is set as the index, which will allow the use of the rolling function
        customer_transactions.index = customer_transactions.TX_DATETIME

        # For each window size
        for window_size in windows_size_in_days:
            # Compute the sum of the transaction amounts and the number of transactions for the given window size
            SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
            NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').count()

            # Compute the average transaction amount for the given window size
            # NB_TX_WINDOW is always >0 since current transaction is always included
            AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

            # Save feature values
            customer_transactions['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
            customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(
                AVG_AMOUNT_TX_WINDOW)

        # Reindex according to transaction IDs
        customer_transactions.index = customer_transactions.TRANSACTION_ID

        # And return the dataframe with the new features
        return customer_transactions

    def get_count_risk_rolling_window(self, terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30],
                                      feature="TERMINAL_ID"):

        terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')

        terminal_transactions.index = terminal_transactions.TX_DATETIME

        NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
        NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

        for window_size in windows_size_in_days:
            NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(
                str(delay_period + window_size) + 'd').sum()
            NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(
                str(delay_period + window_size) + 'd').count()

            NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
            NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

            RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

            terminal_transactions[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
            terminal_transactions[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)

        terminal_transactions.index = terminal_transactions.TRANSACTION_ID

        # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
        terminal_transactions.fillna(0, inplace=True)

        return terminal_transactions

    # Compute the number of transactions per day, fraudulent transactions per day and fraudulent cards per day

    def get_tx_stats(self, transactions_df, start_date_df="2018-04-01"):
        # Number of transactions per day
        nb_tx_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
        # Number of fraudulent transactions per day
        nb_fraudulent_transactions_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
        # Number of compromised cards per day
        nb_compromised_cards_per_day = transactions_df[transactions_df['TX_FRAUD'] == 1].groupby(
            ['TX_TIME_DAYS']).CUSTOMER_ID.nunique()

        tx_stats = pd.DataFrame({"nb_tx_per_day": nb_tx_per_day,
                                 "nb_fraudulent_transactions_per_day": nb_fraudulent_transactions_per_day,
                                 "nb_compromised_cards_per_day": nb_compromised_cards_per_day})

        tx_stats = tx_stats.reset_index()

        start_date = datetime.datetime.strptime(start_date_df, "%Y-%m-%d")
        tx_date = start_date + tx_stats['TX_TIME_DAYS'].apply(datetime.timedelta)

        tx_stats['tx_date'] = tx_date

        return tx_stats

    def get_train_test_set(self, transactions_df,
                           start_date_training,
                           delta_train=7, delta_delay=7, delta_test=7):
        # Get the training set data
        train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                                   (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                       days=delta_train))]

        # Get the test set data
        test_df = []

        # Note: Cards known to be compromised after the delay period are removed from the test set
        # That is, for each test day, all frauds known at (test_day-delay_period) are removed

        # First, get known defrauded customers from the training set
        known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

        # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
        start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

        # Then, for each day of the test set
        for day in range(delta_test):
            # Get test data for that day
            test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                          delta_train + delta_delay +
                                          day]

            # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded
            # customers
            test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                       delta_train +
                                                       day - 1]

            new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
            known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

            test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

            test_df.append(test_df_day)

        test_df = pd.concat(test_df)

        # Sort data sets by ascending order of transaction ID
        train_df = train_df.sort_values('TRANSACTION_ID')
        test_df = test_df.sort_values('TRANSACTION_ID')

        return train_df, test_df
