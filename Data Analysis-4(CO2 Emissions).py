import pandas as pd
import plotly.express as px
from plotly.offline import iplot

World_Data4 = pd.read_csv('Datasets/CO2 Emissions.csv')

fig = px.bar(World_Data4, x="year", y="Kilotons", color='Country Name')
iplot(fig)

