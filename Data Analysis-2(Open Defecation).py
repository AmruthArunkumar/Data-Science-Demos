import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot

World_Data2 = pd.read_csv('Datasets/Open Defecation.csv')

data = dict(type='choropleth',
            colorscale=plotly.colors.sequential.Bluyl,
            locations=World_Data2['Country Code'],
            z=World_Data2['2017'],
            text=World_Data2['Country Name'],
            colorbar={'title': 'Percentage of Population'})

layout = dict(title='People Practicing Open Defecation as of 2017 (%) (from data.worldbank.org)',
              geo=dict(showframe=True,
                       projection={'type': 'orthographic'},
                       showocean=True,
                       showlakes=True,
                       lakecolor='lightBlue',
                       oceancolor='lightBlue'))

choromap = go.Figure(data=[data], layout=layout)

iplot(choromap)
