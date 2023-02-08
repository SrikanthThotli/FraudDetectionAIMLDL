# Necessary imports for this notebook
import datetime
import os

import seaborn as sns

from components import Components

# For plotting

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

comp = Components();

n_customers = 5
customer_profiles_table = comp.generate_customer_profiles_table(n_customers, random_state=0)
print(customer_profiles_table)

n_terminals = 5
terminal_profiles_table = comp.generate_terminal_profiles_table(n_terminals, random_state=0)
print(terminal_profiles_table)

# We first get the geographical locations of all terminals as a numpy array
x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
# And get the list of terminals within radius of $50$ for the last customer
comp.get_list_terminals_within_radius(customer_profiles_table.iloc[4], x_y_terminals=x_y_terminals, r=50)

customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
    lambda x: comp.get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=50), axis=1)
print(customer_profiles_table)

transaction_table_customer_0 = comp.generate_transactions_table(customer_profiles_table.iloc[1],
                                                                start_date="2022-02-01",
                                                                nb_days=5)
print(transaction_table_customer_0)

transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
    lambda x: comp.generate_transactions_table(x.iloc[0], nb_days=5)).reset_index(drop=True)
print(transactions_df)

(customer_profiles_table, terminal_profiles_table, transactions_df) = \
    comp.generate_dataset(n_customers=5000,
                          n_terminals=10000,
                          nb_days=183,
                          start_date="2022-02-01",
                          r=5)

print(transactions_df.shape)
print(transactions_df)

transactions_df = comp.add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)

DIR_OUTPUT = "./simulated-data-raw/"

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

start_date = datetime.datetime.strptime("2022-02-01", "%Y-%m-%d")

for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
    transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')

    date = start_date + datetime.timedelta(days=day)
    filename_output = date.strftime("%Y-%m-%d") + '.pkl'

    # Protocol=4 required for Google Colab
    transactions_day.to_pickle(DIR_OUTPUT + filename_output, protocol=4)
