import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pydst 
Dst = pydst.Dst(lang='da')
Dst.get_data(table_id = 'BY2')
var = Dst.get_variables(table_id='BY2')
var
Data = Dst.get_data(table_id = 'BY2', variables={'KOMK':['*'], 'ALDER':['*'], 'KØN':['*'], 'Tid':['2011','2015']})
Data.drop('BYST', axis=1, inplace=True)
Data.groupby(['TID','KØN']).sum()
Data.groupby('KØN').sum()
