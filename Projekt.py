import pandas as pd
#pip install linearmodels (Run in terminal)
from linearmodels import PanelOLS 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas_datareader
from pandas_datareader import wb
import seaborn as sns
import scipy.stats  as stats
from scipy.stats.stats import pearsonr

###### Downloading inflation and unemployment data from World Bank ######
cntr = ['DK','SE','FR','NL','DE','GB','BE', 'LU', 'AT', 'FI']
cntr1 = ['MX','IN','BR','ID','KE','LY','MY','NG','NO','US']
infl_eu = wb.download(indicator='FP.CPI.TOTL.ZG', country=cntr, start=1991, end=2017)
infl_other = wb.download(indicator='FP.CPI.TOTL.ZG', country=cntr1, start=1991, end=2017)
unem_eu = wb.download(indicator='SL.UEM.TOTL.ZS', country=cntr, start=1991, end=2017)
unem_other = wb.download(indicator='SL.UEM.TOTL.ZS', country=cntr1, start=1991, end=2017)

###### Data manipulation ######

merge_eu = pd.concat([infl_eu, unem_eu], axis=1)
merge_eu = merge_eu.reset_index()
merge_eu.columns = ['country', 'year', 'inflation','unemployment']
merge_eu.year = merge_eu.year.astype(int)
merge_eu

#Making subset for when Quantative Easing was in effect
after_QE = merge_eu[merge_eu['year']>=2015]
after_QE

###### Data description ######
mean_infl_eu = merge_eu.groupby("country")['inflation'].mean()
median_infl_eu = merge_eu.groupby("country")['inflation'].median()
mean_unem_eu = merge_eu.groupby("country")['unemployment'].mean()
median_unem_eu = merge_eu.groupby("country")['unemployment'].median()

tabel = pd.concat([mean_infl_eu, median_infl_eu, mean_unem_eu, median_unem_eu], axis=1)
tabel.columns = ['Average inflation', 'Median inflation', 'Average unemployment', 'Median unemployment']
tabel

Maximum = merge_eu.describe()
Maximum

####### Plots ########
z = merge_eu.set_index('year').groupby('country')['inflation'].plot(legend=True)

y = merge_eu.set_index('year').groupby('country')['unemployment'].plot(legend=True)

#Development in inflation over time
sns.set_style("whitegrid")
g = sns.FacetGrid(merge_eu, col='country', hue='country', col_wrap=4, palette="deep")
g = g.map(plt.plot, 'year', 'inflation')
g = g.map(plt.fill_between, 'year', 'inflation', alpha=0.2).set_titles("{col_name}")  

#Development in unemployment over time
f = sns.FacetGrid(merge_eu, col='country', hue='country', col_wrap=4, palette="deep")
f = f.map(plt.plot, 'year', 'unemployment')
f = f.map(plt.fill_between, 'year', 'unemployment', alpha=0.2).set_titles("{col_name}") 

#Long-run correlation between unemployment and inflation (Long run Phillips curve)
LR = sns.FacetGrid(merge_eu, col='country', hue='country', col_wrap=4, palette="deep")
LR = LR.map(plt.plot, 'unemployment', 'inflation').set_titles("{col_name}") 

#Short-run correlation between unemployment and inflation (SRPC)
SR = sns.FacetGrid(after_QE, col='country', hue='country', col_wrap=4, palette="deep")
SR = SR.map(plt.plot, 'unemployment', 'inflation').set_titles("{col_name}") 


###### Calculating correlation and significance ######

#Defining function calculating p-values
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 10)
    return pvalues

#Calculcating correlation and p-values for full dataset

merge_eu = merge_eu.set_index(['year','country'])
after_QE = after_QE.set_index(['year', 'country'])
corr = merge_eu.corr()
Pval = calculate_pvalues(merge_eu)

#Calculating correlation and p-values for subset during QE
corr_QE = after_QE.corr()
Pval_QE = calculate_pvalues(after_QE)

#Printing results
corr
Pval

corr_QE
Pval_QE

merge_eu = merge_eu.reset_index()
year_full = pd.Categorical(merge_eu.year)
merge_eu = merge_eu.set_index(['country','year'])
merge_eu['year']=year_full
regression1=PanelOLS(merge_eu.inflation, merge_eu.unemployment, entity_effects=True)
res1 = regression1.fit(cov_type='clustered', cluster_entity=True)
print(res1)

after_QE = after_QE.reset_index()
year_QE = pd.Categorical(after_QE.year)
after_QE = after_QE.set_index(['country','year'])
after_QE['year']=year_QE
regression2=PanelOLS(after_QE.inflation, after_QE.unemployment, entity_effects=True)
res2 = regression2.fit(cov_type='clustered', cluster_entity=True)
print(res2)