# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:26:02 2018

@author: or
"""



import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("tryout.csv")

df.columns
seed          = df['Seed']
days          = df['Dyas']
sum           = df['Sum(only for Purchase above $X)']
Audience      = df['% Of The Audience']
lal           = df['LAL%']
optimization  = df['Optimization method']
placement     = df['Placement']
gender        = df['Gender']




seed = ['VV','FTD','LVL Achieved','Purchase - FB Organic','Purchase + Not Install' ,'Launch X Times XD','Purchase Above $X',
        'LTV','MA','HV','Purchase','Complete Registration','Page Engagement','Tutorial Completed','Installs',
        'Saved Posts','Initiate Checkout + Purchased Cancelled','MAHV']

days = ['30','60','90','180']

sum = ['10', '25', '50', 'May be different between apps']

Audience = ['5', '10', '25', '50', '75', '95']

lal = ['1', '2', '3', '4', '5']

optimization = ['VO', 'AEO', 'Appinstalls']

placement = ['Facebook', 'Audience Network', 'Instagram', 'Messenger']

gender = ['Male', 'Female', 'All']


print (i,',',j,',',k,',',l,',',m,',',n,',',o,',',p,',')

vol = []



with open('test.csv' , 'w', newline = '' ) as f:
    fieldnames = ['seed', 'days', 'sum', 'Audiendce', 'lal', 'optimization', 'placement', 'gender']
    thewriter = csv.DictWriter(f, fieldnames = fieldnames)


    thewriter.writeheader()
    
    for i in seed:    
            for j in days:
                for k in sum:
                    for l in Audience:
                        for m in lal:
                            for n in optimization:
                                for o in placement:
                                    for p in gender:
                                        thewriter.writerow({'seed': i, 'days': j, 'sum': k, 'Audiendce': l,
                                                            'lal': m, 'optimization': n, 'placement': o, 'gender': p})



