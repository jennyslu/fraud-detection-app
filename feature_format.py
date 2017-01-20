from __future__ import division
import pandas as pd
import numpy as np

features_list = []

def total_tickets(data):
    for row in data:
        quantity_total = 0
        for ticket_type in row['ticket_types']:
            quantity_total += ticket_type['quantity_total']
        row['num_tickets'] = quantity_total
    features_list.append('num_tickets')

def email_domains(df):
    email_domains = ['gmail', 'yahoo', 'hotmail']
    features_list.extend(email_domains)
    for domain in email_domains:
        df[domain] = (df['email_domain'].str.startswith(domain)).astype(int)
    df['other'] = np.logical_not(df[email_domains].sum(axis=1).astype(bool)).astype(int)
    features_list.append('other')

def payout_type(df):
    payout_type_col_names = []
    payout_types = ['ACH', 'CHECK', '']
    for i in payout_types:
        name = 'payout_' + i
        df[name] = 0
        payout_type_col_names.append(name)
    dummies = pd.get_dummies(df['payout_type'], prefix='payout')
    df[dummies.columns] = dummies
    features_list.extend(payout_type_col_names)

def sale_duration(df):
    df.loc[:,'sale_duration2'].fillna(0, inplace=True)
    df.loc[df['sale_duration2']<0, 'sale_duration2'] = 0
    # import pdb; pdb.set_trace()
    features_list.append('sale_duration2')

def fill_nans(df, columns):
    df.loc[:,columns] = df[columns].fillna(0)
    features_list.extend(columns)

def user_type(df):
    user_type_list = range(1,6)
    user_type_cols = []
    for i in user_type_list:
        name = 'user_type_'+str(i)
        df[name] = 0
        user_type_cols.append(name)
    df.loc[~df['user_type'].isin(user_type_list), 'user_type'] = 3
    dummies = pd.get_dummies(df['user_type'], prefix='user_type')
    df[dummies.columns] = dummies
    features_list.extend(user_type_cols)

def upper(df):
    df['is_upper'] = (df['name'].str.isupper()).astype(int)
    features_list.append('is_upper')

def currency_dums(df):
    df['dummy_currency'] = (df['currency'].isin(['GBP', 'MXN'])).astype(int)
    features_list.append('dummy_currency')

def payee_checking(df):
    df['payee_check'] = (df['payee_name']=='').astype(int)
    df['payee_check'].fillna(0, inplace=True)
    features_list.append('payee_check')

def payout_ratio(df):
    df['payout_pct'] = df['num_payouts']/(df['num_order'] + 0.01)
    df.loc[df['payout_pct'] > 0,'payout_pct'] = 1
    features_list.append('payout_pct')

def social_media(df, columns):
    df.loc[:,columns].fillna(0, inplace=True)
    df[columns] = df[columns].astype(bool).astype(int)
    features_list.extend(columns)

def prev_payouts(df):
    df['prev_num_payouts'] = df['previous_payouts'].map(len)
    df['prev_pay_no'] = (df['prev_num_payouts']==0).astype(int)
    features_list.append('prev_pay_no')

def cooldown(df):
    df['cd_period'] = df['event_created'] - df['user_created']
    df['cd'] = (df['cd_period'] < 60).astype(int)
    features_list.append('cd')


def feature_engineering(data):
    # calculate num_tickets feature
    total_tickets(data)
    # turn into dataframe
    df = pd.DataFrame(data)
    # create email domain feature
    email_domains(df)
    # create payout type feature
    payout_type(df)
    # truncate negative sales duration
    sale_duration(df)
    # create features for header and address
    fill_nans(df, ['delivery_method'])
    # user types
    user_type(df)
    # is post title all uppercase
    upper(df)
    # GBP and MXN vs others
    currency_dums(df)
    # check if payee name exists?
    payee_checking(df)
    # num_payouts/num_order binned into 0 and not
    payout_ratio(df)
    # social media
    social_media(df, ['org_facebook', 'org_twitter'])
    # check whether this guy has a previous payout
    prev_payouts(df)
    # other features
    features_list.extend(['fb_published'])

    ## extract features we want
    X = df[features_list].values
    return X, np.array(features_list)
