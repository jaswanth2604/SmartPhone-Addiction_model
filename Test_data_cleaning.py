# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 02:46:49 2026

@author: Jaswanth
"""

import pandas as pd
df=pd.read_csv("test.csv")
df.isnull().sum()
des=df.describe()
des.columns
cols=[ 'age', 'daily_screen_time_hours', 'social_media_hours','gaming_hours', 'work_study_hours', 'sleep_hours','notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
for c in cols:
  mean=df[c].mean()
  df[c]=df[c].fillna(mean)
df["age"]= df["age"].round(0)
df["app_opens_per_day"]= df["app_opens_per_day"].round(0)
df["notifications_per_day"]= df["notifications_per_day"].round(0)
for c in cols:
  df[c]=df[c].round(2)
cols3=df[["gender","stress_level","academic_work_impact"]]
df["gender"].value_counts().index[0]
for c in cols3:
    high_value= df[c].value_counts().index[0]
    df[c]=df[c].fillna(high_value)
df.to_csv("test_data_cleaned.csv",index=False)