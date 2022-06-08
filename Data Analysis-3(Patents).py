import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot

World_Data3 = pd.read_csv('Datasets/Patents.csv')

data = dict(type='choropleth',
            colorscale=plotly.colors.sequential.YlGnBu,
            locations=World_Data3['Country Code'],
            z=np.log10(World_Data3['2018']),
            text=World_Data3['Country Name'],
            colorbar=dict(len=0.75,
                          title='Base 10 logarithmic scale of Number',
                          tickvals=[0, 1, 2, 3, 4, 5, 6],
                          ticktext=['10**0', '10**1', '10**2', '10**3',
                                    '10**4', '10**5', '10**6']))

layout = dict(title='Number of People who Own Patents as of 2018 (from data.worldbank.org)'
                    ' **With a Base 10 logarithmic scale',
              geo=dict(showframe=True,
                       projection={'type': 'orthographic'},
                       showocean=True,
                       showlakes=True,
                       lakecolor='lightBlue',
                       oceancolor='lightBlue'))

choromap = go.Figure(data=[data], layout=layout)

iplot(choromap)
