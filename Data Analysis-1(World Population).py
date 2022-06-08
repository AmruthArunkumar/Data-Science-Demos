import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

World_Data1 = pd.read_csv('Datasets/World Population.csv')

data = dict(type='choropleth',
            colorscale='earth',
            locations=World_Data1['Country Code'],
            z=np.log10(World_Data1['2018']),
            zmin=5,
            zmax=10,
            text=World_Data1['Country Name'],
            colorbar=dict(len=0.75,
                          title='Base 10 logarithmic scale of Population',
                          tickvals=[5, 6, 7, 8, 9, 10],
                          ticktext=['10**5', '10**6', '10**7', '10**8',
                                    '10**9', '10**10']))


layout = dict(title='Population of Countries as of 2018 (from data.worldbank.org)'
              ' **With a Base 10 logarithmic scale',
              geo=dict(showframe=False,
                       projection={'type': 'orthographic'},
                       showocean=True,
                       showlakes=False,
                       oceancolor='lightBlue'))

choromap = go.Figure(data=[data], layout=layout)

iplot(choromap)
