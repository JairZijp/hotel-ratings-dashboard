import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pymongo
import plotly
import plotly.graph_objs as go
from plotly.offline import *

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['assignment2']
collection_name = 'reviews'
db_cm = mng_db[collection_name].find()

df = pd.DataFrame(list(db_cm))
df.drop_duplicates(subset ="Hotel_Address", inplace = True) 
print(df)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df['text'] = df['Hotel_Name'] + '<br>Rating ' + (df['Average_Score']/1e6).astype(str)+' '
limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","lightgrey"]
cities = []
scale = 5000

for index, row in df.iterrows():
    city = go.Scattergeo(
        lon = [row['lng']],
        lat = [row['lat']],
        text = row['Hotel_Name'],
        marker = go.scattergeo.Marker(
            size = row['Average_Score'],
            line = go.scattergeo.marker.Line(
                width=0.5, color='rgb(40,40,40)'
            )
        ),
        name = row["Hotel_Name"] )
    cities.append(city)
layout = go.Layout(
        title = go.layout.Title(
            text = 'Hote reviews'
        ),
        showlegend = True,
        geo = go.layout.Geo(
            scope = 'world',
            projection = go.layout.geo.Projection(
                type='mercator'
            ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = go.Figure(data=cities, layout=layout)

app.layout = html.Div([
    #dcc.Graph(fig)
    dcc.Graph(
         id='example-graph',
         figure={
             'data': cities,
             'layout': layout
         }
    )
])


# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),

#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
#                 {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': 'Dash Data Visualization'
#             }
#         }
#     )
# ])

if __name__ == '__main__':
    app.run_server(debug=True)