
from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
import argparse
import json
import math
import re
import os
import sys
import csv
import socket # -- ip checks

import seaborn as sns
import matplotlib.pyplot as plt

from jinja2 import Environment, PackageLoader

# --- functions ---
def get_config(config):
    """ convert json config file into a python dict  """
    with open(config, 'r') as f:
        config_dict = json.load(f)[0]
    return config_dict

# -- load data --
def get_dataframe(config):
    """ load csv into python dataframe """
    df = pd.read_csv(config['input_file'], low_memory=False)
    return df

# -- 
def get_overview(config, df):
    """ return details of the dataframe and any issues found  """
    overview_msg = {}
    df = df.copy()
    column_cnt = len(df.columns)
    try:
        df['EVENT_TIMESTAMP'] = pd.to_datetime(df[config['required_features']['EVENT_TIMESTAMP']], infer_datetime_format=True)
        date_range = df['EVENT_TIMESTAMP'].min().strftime('%Y-%m-%d') + ' to ' + df['EVENT_TIMESTAMP'].max().strftime('%Y-%m-%d')
        day_cnt = (df['EVENT_TIMESTAMP'].max() - df['EVENT_TIMESTAMP'].min()).days 
    except:
        overview_msg[config['required_features']['EVENT_TIMESTAMP']] = " Unable to convert" + config['required_features']['EVENT_TIMESTAMP'] + " to timestamp"
        date_range = ""
        day_cnt = 0 
       
    record_cnt  = df.shape[0]
    memory_size = df.memory_usage(index=True).sum()
    record_size = round(float(memory_size) / record_cnt,2)
    n_dupe      = record_cnt - len(df.drop_duplicates())

    if record_cnt <= 10000:
        overview_msg["Record count"] = "A minimum of 10,000 rows are required to train the model, your dataset contains " + str(record_cnt)

    overview_stats = {
        "Record count"      : "{:,}".format(record_cnt) ,
        "Column count"      : "{:,}".format(column_cnt),
        "Duplicate count"   : "{:,}".format(n_dupe),
        "Memory size"       : "{:.2f}".format(memory_size/1024**2) + " MB",
        "Record size"       : "{:,}".format(record_size) + " bytes",
        "Date range"        : date_range,
        "Day count"         : "{:,}".format(day_cnt) + " days",
        "overview_msg"      : overview_msg,
        "overview_cnt"      : len(overview_msg)
    }

    return df, overview_stats

def set_feature(row, config):
    """ sets the feature type of each variable in the file, identifies features with issues 
        as well as the required features. this is the first pass of rules 
    """
    rulehit = 0
    feature = ""
    message = ""
    required_features = config['required_features']
    
    # -- assign numeric -- 
    if ((row._dtype in ['float64', 'int64']) and (row['nunique'] > 1)):
        feature = "numeric"
        message = "(" + "{:,}".format(row['nunique']) + ") unique"
        
    # -- assign categorical -- 
    if ((row._dtype == 'object') and ( row.nunique_pct <= 0.75)):
        feature = "categorical"
        message = "(" + "{:.2f}".format(row.nunique_pct*100) + "%) unique"
        
    # -- assign categorical to numerics  -- 
    if ((row._dtype in ['float64', 'int64']) and ( row['nunique'] <= 1024 )):
        feature = "categorical"
        message = "(" + "{:,}".format(row['nunique']) + ") unique"
    
     # -- assign binary -- 
    if (row['nunique'] == 2 ):
        feature = "categorical"
        message = "(" + "{:}".format(row['nunique']) + ") binary"
    
    # -- single value --
    if (row['nunique'] == 1):
        rulehit = 1
        feature = "exclude"
        message = "(" + "{:}".format(row['nunique']) + ") single value"
    
    # -- null pct --   
    if (row.null_pct >= 0.50 and (rulehit == 0)):
        rulehit = 1
        feature = "exclude"
        message =  "(" + "{:.2f}".format(row.null_pct*100) + "%) missing "

    # -- categorical w. high % unique 
    if ((row._dtype == 'object') and ( row.nunique_pct >= 0.75)) and (rulehit == 0):
        rulehit = 1
        feature = "exclude"
        message = "(" + "{:.2f}".format(row.nunique_pct*100) + "%) unique"
    
     # -- numeric w. extreeme % unique 
    if ((row._dtype in ['float64', 'int64']) and ( row.nunique_pct >= 0.95)) and (rulehit == 0):
        rulehit = 1
        feature = "exclude"
        message = "(" + "{:.2f}".format(row.nunique_pct*100) + "%) unique"
    
    if row._column == required_features['EMAIL_ADDRESS']:
        feature = "EMAIL_ADDRESS"
    if row._column == required_features['IP_ADDRESS']:
        feature = "IP_ADDRESS"
    if row._column == required_features['EVENT_TIMESTAMP']:
        feature = "EVENT_TIMESTAMP"
    if row._column == required_features['EVENT_LABEL']:
        feature = "EVENT_LABEL"
        
    return feature, message

def get_label(config, df):
    """ returns stats on the label and performs intial label checks  """
    message = {}
    label = config['required_features']['EVENT_LABEL']
    label_summary = df[label].value_counts()
    rowcnt = df.shape[0]
    label_dict = {
        "label_field"  : label,
        "label_values" : df[label].unique(),
        "label_dtype"  : label_summary.dtype,
        "fraud_rate"   : "{:.2f}".format((label_summary.min()/label_summary.sum())*100),
        "fraud_label": str(label_summary.idxmin()),
        "fraud_count": label_summary.min(),
        "legit_rate" : "{:.2f}".format((label_summary.max()/label_summary.sum())*100),
        "legit_count": label_summary.max(),
        "legit_label": str(label_summary.idxmax()),
        "null_count" : "{:,}".format(df[label].isnull().sum(axis = 0)),
        "null_rate"  : "{:.2f}".format(df[label].isnull().sum(axis = 0)/rowcnt),
    }
    
    """
    label checks
    """
    if label_dict['fraud_count'] <= 500:
        message['fraud_count'] = "Fraud count " + label_dict['fraud_count'] + " is less than 500\n"
    
    if df[label].isnull().sum(axis = 0)/rowcnt >= 0.01:
        message['label_nulls'] =   "Your LABEL column contains  " + label_dict["null_count"] +" a significant number of null values"
    
    label_dict['warnings'] = len(message)
    
    return label_dict, message

def get_partition(config, df):
    """ evaluates your dataset partitions and checks the distribution of fraud lables """
   
    df = df.copy()
    row_count = df.shape[0]
    required_features = config['required_features']
    message = {}
    stats ={}
    try:
        df['_event_timestamp'] = pd.to_datetime(df[required_features['EVENT_TIMESTAMP']])
        df['_dt'] = pd.to_datetime(df['_event_timestamp'].dt.date)
    except:
        message['_event_timestamp'] = "could not parse " + required_features['EVENT_TIMESTAMP'] + " into a date or timestamp object"
        df['_event_timestamp'] = df[required_features['EVENT_TIMESTAMP']]
        df['_dt'] = df['_event_timestamp']
    
    label_summary = df[required_features['EVENT_LABEL']].value_counts()
     
    legit_label = label_summary.idxmax()
    fraud_label = label_summary.idxmin()
    
    df = df.sort_values(by=['_event_timestamp']).reset_index(drop=True)
    ctab = pd.crosstab(df['_dt'].astype(str), df[required_features['EVENT_LABEL']]).reset_index()
    stats['labels'] = ctab['_dt'].tolist()
    stats['legit_rates'] = ctab[legit_label].tolist()
    stats['fraud_rates'] = ctab[fraud_label].tolist()
    
    # -- set partitions -- 
    df['partition'] = 'training'
    df.loc[math.ceil(row_count*.7):math.ceil(row_count*.85),'partition'] = 'evaluation'
    df.loc[math.ceil(row_count*.85):,'partition'] = 'testing'
    
    message = ""
    
    return stats, message 

def get_stats(config, df):
    """ generates the key column analysis statistics calls set_features function """
    df = df.copy()
    rowcnt = len(df)
    df_s1  = df.agg(['count', 'nunique',]).transpose().reset_index().rename(columns={"index":"_column"})
    df_s1['count'] = df_s1['count'].astype('int64')
    df_s1['nunique'] = df_s1['nunique'].astype('int64')
    df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1['nunique'] / rowcnt
    dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"_column", 0:"_dtype"})
    df_stats = pd.merge(dt, df_s1, on='_column', how='inner')
    df_stats = df_stats.round(4)
    df_stats[['_feature', '_message']] = df_stats.apply(lambda x: set_feature(x,config), axis = 1, result_type="expand")
    
    return df_stats, df_stats.loc[df_stats["_feature"]=="exclude"]

def get_email(config, df):
    """ gets the email statisitcs and performs email checks """
    message = {}
    required_features = config['required_features']
    email = required_features['EMAIL_ADDRESS']
    email_recs = df.shape[0]
    email_null = df[email].isna().sum()
    emails = pd.Series(pd.unique(df[email].values))
    email_unique  = len(emails)
    email_valid = df[email].str.count('\w+\@\w+').sum()
    email_invalid = email_recs - ( email_valid + email_null) 

    df['domain'] = df[email].str.split('@').str[1]
    top_10 = df['domain'].value_counts().head(10)
    top_dict = top_10.to_dict()
    
    label_summary = df[required_features['EVENT_LABEL']].value_counts()
    fraud_label   = label_summary.idxmin()
    legit_label   = label_summary.idxmax()
    
    ctab = pd.crosstab(df['domain'], df[required_features['EVENT_LABEL']],).reset_index()
    ctab['tot'] = ctab[fraud_label] + ctab[legit_label]
    ctab['fraud_rate'] = ctab[fraud_label]/ctab['tot'] 
    ctab = ctab.sort_values(['tot'],ascending=False)
    top_n= ctab.head(10)
    
    domain_count = df['domain'].nunique()
    domain_list = top_n['domain'].tolist()
    domain_fraud = top_n[fraud_label].tolist()
    domain_legit = top_n[legit_label].tolist()
    
    # -- email checks --
    if email_unique <= 100: 
        message['unique_count'] = "Low number of unique emails: " + str(email_unique)
        
    if email_null/len(df) >= 0.20:
        message['null_email'] = "High percentage of null emails: " + '{0: >#016.2f}'.format(email_null/len(df)) + "%"
        
    if email_invalid/len(df) >= 0.5:
        message['invalid_email'] = "High number of invalid emails: " + '{0: >#016.2f}'.format(email_invalid/len(df)) + "%"
    
    domain_list = list(top_dict.keys())
    #domain_value = list(top_dict.values())
    
    email_dict = {
        "email_addr"    : email,
        "email_recs"    : "{:,}".format(email_recs),
        "email_null"    : "{:,}".format(email_null),
        "email_pctnull" : "{:.2f}".format((email_null/email_recs)*100),
        "email_unique"  : "{:,}".format(email_unique),
        "email_pctunq"  : "{:.2f}".format((email_unique/email_recs)*100),
        "email_valid"   : "{:,}".format(email_valid),
        "email_invalid" : "{:,}".format(email_invalid),
        "email_warnings": len(message),
        "domain_count"  : "{:,}".format(domain_count),
        "domain_list"   : domain_list,
        "domain_fraud"  : domain_fraud,
        "domain_legit"  : domain_legit
    }
    
    return email_dict, message
def valid_ip(ip):
    """ checks to insure we have a valid ip address """
    try:
        parts = ip.split('.')
        return len(parts) == 4 and all(0 <= int(part) < 256 for part in parts)
    except ValueError:
        return False # one of the 'parts' not convertible to integer
    except (AttributeError, TypeError):
        return False # `ip` isn't even a string
        
def get_ip_address(config, df):
    """ gets ip address statisitcs and performs ip address checks """
    message = {}
    required_features = config['required_features']
    ip = required_features['IP_ADDRESS']
    ip_recs = df.shape[0] - df[ip].isna().sum()
    ip_null = df[ip].isna().sum()
    ips = pd.Series(pd.unique(df[ip].values))
    ip_unique  = len(ips)
    df['_ip'] = df[ip].apply(valid_ip)
    ip_valid = df['_ip'].sum()
    ip_invalid = ip_recs - ip_valid
    print(ip_null)
    label_summary = df[required_features['EVENT_LABEL']].value_counts()
    fraud_label   = label_summary.idxmin()
    legit_label   = label_summary.idxmax()
    
    ctab = pd.crosstab(df[required_features['IP_ADDRESS']], df[required_features['EVENT_LABEL']],).reset_index()
    
    ctab['tot'] = ctab[fraud_label] + ctab[legit_label]
    ctab['fraud_rate'] = ctab[fraud_label]/ctab['tot'] 
    ctab = ctab.sort_values(['tot'],ascending=False)
    top_n= ctab.head(10)
    
    ip_list = top_n[ip].tolist()
    ip_fraud = top_n[fraud_label].tolist()
    ip_legit = top_n[legit_label].tolist()
    
    # -- ip checks --
    if ip_unique <= 100: 
        message['unique_count'] = "Low number of unique ip addresses: " + str(ip_unique)
        
    if ip_null/len(df) >= 0.20:
        message['null_ip'] = "High percentage of null ip addresses: " + '{0: >#016.2f}'.format(ip_null/len(df)) + "%"
        
    if ip_invalid/len(df) >= 0.5:
        message['invalid_ip'] = "High number of invalid ip addresses: " + '{0: >#016.2f}'.format(ip_invalid/len(df)) + "%"
    
    ip_dict = {
        "ip_addr"    : ip,
        "ip_recs"    : "{:,}".format(ip_recs),
        "ip_null"    : "{:,}".format(ip_null),
        "ip_pctnull" : "{:.2f}".format((ip_null/ip_recs)*100),
        "ip_unique"  : "{:,}".format(ip_unique),
        "ip_pctunq"  : "{:.2f}".format((ip_unique/ip_recs)*100),
        "ip_valid"   : "{:,}".format(ip_valid),
        "ip_invalid" : "{:,}".format(ip_invalid), 
        "ip_warnings": len(message),
        "ip_list"   : ip_list,
        "ip_fraud"  : ip_fraud,
        "ip_legit"  : ip_legit
    }
    
    return ip_dict, message

def col_stats(df, target, column):
    """ generates column statisitcs for categorical columns """
    legit = df[target].value_counts().idxmax()
    fraud = df[target].value_counts().idxmin()
    try:
        cat_summary = pd.crosstab(df[column],df[target]).reset_index().sort_values(fraud, ascending=False).reset_index(drop=True).head(10).rename(columns={legit:"legit", fraud:"fraud"})
        cat_summary['total'] = cat_summary['fraud'] + cat_summary['legit']
        cat_summary['fraud_pct'] = cat_summary['fraud']/(cat_summary['fraud']+ cat_summary['legit'])
        cat_summary['legit_pct'] = 1 - cat_summary['fraud_pct']
        cat_summary = cat_summary.sort_values('fraud_pct', ascending=False).round(4)
    except:
        cat_summary = pd.crosstab(df[column],df[target]).reset_index().sort_values(legit, ascending=True).reset_index(drop=True).head(10).rename(columns={legit:"legit", fraud:"fraud"})
        cat_summary['fraud'] = 0
        cat_summary['total'] = cat_summary['legit']
        cat_summary['fraud_pct'] = 0.0
        cat_summary['legit_pct'] = 1 - cat_summary['fraud_pct']
        cat_summary = cat_summary.sort_values('fraud_pct', ascending=False).round(4)
    return cat_summary  
    
def get_categorical(config, df_stats, df):
    """ gets categorical feature stats: count, nunique, nulls  """
    required_features = config['required_features']
    features = df_stats.loc[df_stats['_feature']=='categorical']._column.tolist()
    target = required_features['EVENT_LABEL']
    df = df[features + [target]].copy()
    rowcnt = len(df)
    df_s1  = df.agg(['count', 'nunique',]).transpose().reset_index().rename(columns={"index":"_column"})
    df_s1['count'] = df_s1['count'].astype('int64')
    df_s1['nunique'] = df_s1['nunique'].astype('int64')
    df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1['nunique'] / rowcnt
    dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"_column", 0:"_dtype"})
    df_stats = pd.merge(dt, df_s1, on='_column', how='inner').round(4)
 
    cat_list = []
    for rec in df_stats.to_dict('records'):
        if rec['_column'] != target:
            cat_summary = col_stats(df, target, rec['_column'])
            rec['top_n'] = cat_summary[rec['_column']].tolist()
            rec['top_n_count'] = cat_summary['total'].tolist()
            rec['fraud_pct'] = cat_summary['fraud_pct'].tolist()
            rec['legit_pct'] = cat_summary['legit_pct'].tolist()
            rec['fraud_count'] = cat_summary['fraud'].tolist()
            rec['legit_count'] = cat_summary['legit'].tolist()
            cat_list.append(rec)
 
    return cat_list

def ncol_stats(df, target, column):
    """ calcuates numeric column statstiics """
    df = df.copy()
    n = df[column].nunique()
    # -- rice rule -- 
    k = int(round(2*(n**(1/3)),0)) 
    # -- bin that mofo -- 
    try:
        df['bin'] = pd.qcut(df[column], q=k, duplicates='drop')
        legit = df[target].value_counts().idxmax()
        fraud = df[target].value_counts().idxmin()
        try:
            num_summary = pd.crosstab(df['bin'],df[target]).reset_index().rename(columns={legit:"legit", fraud:"fraud"})
            num_summary['total'] = num_summary['fraud'] + num_summary['legit']
            num_summary['empty_label'] = [""] * df['bin'].nunique()
            num_summary['bin_label'] = num_summary['bin'].astype(str)
        except:
            num_summary = pd.crosstab(df['bin'],df[target]).reset_index().rename(columns={legit:"legit", fraud:"fraud"})
            num_summary['fraud'] = 0
            num_summary['total'] = num_summary['legit']
            num_summary['empty_label'] = [""] * df['bin'].nunique()
            num_summary['bin_label'] = num_summary['bin'].astype(str)
    except:
        num_summary = pd.DataFrame()
        num_summary['legit'] = 0
        num_summary['fraud'] = 0
        num_summary['total'] = 0
        num_summary['empty_label'] = [""]
        num_summary['bin_label'] = [""]

    return num_summary  
    
def get_numerics( config, df_stats, df):
    """ gets numeric feature descriptive statsitics and graph detalis """
    required_features = config['required_features']
    features = df_stats.loc[df_stats['_feature']=='numeric']._column.tolist()
    target = required_features['EVENT_LABEL']

    df = df[features + [target]].copy()
    rowcnt = len(df)
    df_s1  = df.agg(['count', 'nunique','mean','min','max']).transpose().reset_index().rename(columns={"index":"_column"})
    df_s1['count'] = df_s1['count'].astype('int64')
    df_s1['nunique'] = df_s1['nunique'].astype('int64')
    df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1['nunique'] / rowcnt
    dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"_column", 0:"_dtype"})
    df_stats = pd.merge(dt, df_s1, on='_column', how='inner').round(4)
    num_list = []
    for rec in df_stats.to_dict('records'):
        if rec['_column'] != target and rec['count'] > 1:
            n_summary = ncol_stats(df, target, rec['_column'])
            rec['bin_label'] = n_summary['bin_label'].tolist()
            rec['legit_count'] = n_summary['legit'].tolist()
            rec['fraud_count'] = n_summary['fraud'].tolist()
            rec['total'] = n_summary['total'].tolist()
            rec['empty_label'] = n_summary['empty_label'].tolist()
            num_list.append(rec)
 
    return num_list

def profile_report(config):
    """ main function - generates the profile report of the CSV """
    # -- Jinja2 environment -- 
    env = Environment(loader=PackageLoader('afd_profile', 'templates'))
    profile= env.get_template('profile.html')
    
    # -- all the checks -- 
    df = get_dataframe(config)
    df, overview_stats = get_overview(config, df)
    df_stats, warnings = get_stats(config, df)
    lbl_stats, lbl_warnings = get_label(config, df)
    p_stats, p_warnings = get_partition(config, df)
    e_stats, e_warnings = get_email(config, df)
    i_stats, i_warnings = get_ip_address(config, df)
    cat_rec = get_categorical(config, df_stats, df)
    num_rec = get_numerics( config, df_stats, df)
    
    # -- render the report 
    profile_results = profile.render(file = config['input_file'], 
                                     overview = overview_stats,
                                     warnings = warnings,
                                     df_stats=df_stats.loc[df_stats['_feature'] != 'exclude'],
                                     label  = lbl_stats,
                                     label_msg = lbl_warnings,
                                     p_stats=p_stats,
                                     p_warnings = p_warnings,
                                     e_stats = e_stats,
                                     e_warnings = e_warnings,
                                     i_stats=i_stats,
                                     i_warnings=i_warnings,
                                     cat_rec=cat_rec,
                                     num_rec=num_rec) 
    return profile_results


if __name__ == "__main__":
    """ for command line call: 
        > afd_profile.py config_example.json > afd_profile_report.html 
    """
    config  = sys.argv[1:][0]
    config  = get_config(config)
    profile = profile_report(config)
    print(profile)