# Initialization: Load shared functions and simulated data
import datetime
import os

from featurization import FeatureEngineering

from sharedFunctions import SharedMethod

DIR_INPUT = './simulated-data-raw/'

BEGIN_DATE = "2022-02-01"
END_DATE = "2022-08-02"
common = SharedMethod()
feat = FeatureEngineering()

print("Load  files")
transactions_df = common.read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)
print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),
                                                                               transactions_df.TX_FRAUD.sum()))

print(transactions_df.head())

transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(feat.is_weekend)
transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(feat.is_night)

spending_behaviour_customer_0 = feat.get_customer_spending_behaviour_features(
    transactions_df[transactions_df.CUSTOMER_ID == 0])

transactions_df = transactions_df.groupby('CUSTOMER_ID').apply(
    lambda x: feat.get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30]))
transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)

transactions_df[transactions_df.TX_FRAUD == 0].TERMINAL_ID[0]

feat.get_count_risk_rolling_window(transactions_df[transactions_df.TERMINAL_ID == 3059], delay_period=7,
                                   windows_size_in_days=[1, 7, 30])

transactions_df = transactions_df.groupby('TERMINAL_ID').apply(lambda x: feat.get_count_risk_rolling_window(x,
                                                                                                            delay_period=7,
                                                                                                            windows_size_in_days=[
                                                                                                                1, 7,
                                                                                                                30],
                                                                                                            feature="TERMINAL_ID"))
transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)

DIR_OUTPUT = "./simulated-data-transformed/"

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

start_date = datetime.datetime.strptime("2022-02-01", "%Y-%m-%d")

for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
    transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')

    date = start_date + datetime.timedelta(days=day)
    filename_output = date.strftime("%Y-%m-%d") + '.pkl'

    # Protocol=4 required for Google Colab
    transactions_day.to_pickle(DIR_OUTPUT + filename_output, protocol=4)