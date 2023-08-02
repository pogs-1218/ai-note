'''
cross tabulation
contingency table(crosstabs)
'''

import pandas as pd
import numpy as np

df = pd.read_excel('survey.xls')
print(df)

df_new = pd.crosstab(df['Sex'], df['Handedness'], margins=True)
print(df_new)

df_new = pd.crosstab(df['Sex'], df['Handedness'], normalize=True)
print(df_new)

df_new = pd.crosstab(df['Sex'], df['Handedness'], values=df['Age'], aggfunc=np.average)
print(df_new)