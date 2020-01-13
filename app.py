import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import pymongo
import plotly
import plotly.graph_objs as go

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['assignment2']
collection_name = 'reviews'
db_cm = mng_db[collection_name].find()

df = pd.DataFrame(list(db_cm))
df.drop_duplicates(subset ="Hotel_Address", inplace = True) 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df['text'] = df['Hotel_Name'] + '<br>Rating ' + df['Average_Score'].astype(str)+' '
cities = []

final_df = df

def fill_map(df):
    print("filling map...")
    cities = []
    for index, row in df.iterrows():
        city = go.Scattergeo(
            lon = [row['lng']],
            lat = [row['lat']],
            text = row['text'],
            marker = go.scattergeo.Marker(
                size = row['Average_Score'],
                line = go.scattergeo.marker.Line(
                    width=0.5, color='rgb(40,40,40)'
                )
            ))
        cities.append(city)
    return cities

fill_map(final_df)

layout = go.Layout(
        title = go.layout.Title(
            text = 'Hotel reviewss'
        ),
        width = 1000,
        height = 750,
        showlegend = False,
        geo = go.layout.Geo(
            scope = 'europe',
            projection = go.layout.geo.Projection(
                type='miller'
            ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

app.layout = html.Div([
    dcc.Graph(
         id='map'
    ),
    dcc.RangeSlider(
        id='my-range-slider',
        min=1,
        max=10,
        step=0.1,
        value=[0, 10]
    ),
    html.Div(id='output-container-range-slider')
])

@app.callback(
    [Output('map', 'figure'),
    Output('output-container-range-slider', 'children')],
    [Input('my-range-slider', 'value')])
def update_output(range):
    # db_cm = mng_db[collection_name].find({ "Average_Score" : { "$gt" :  range[0], "$lt" : range[1]}})
    # df = pd.DataFrame(list(db_cm))
    # df.drop_duplicates(subset ="Hotel_Address", inplace = True) 
    # df['text'] = df['Hotel_Name'] + '<br>Rating ' + (df['Average_Score']/1e6).astype(str)+' '
    
    filtered_df = df[df['Average_Score'].between(range[0], range[1])]
    
    traces = fill_map(filtered_df)
    return {
        'data': traces,
        'layout': go.Layout(
                title = go.layout.Title(
                text = 'Hotel reviews'
            ),
            width = 840,
            height = 650,
            showlegend = False,
            geo = go.layout.Geo(
                scope = 'europe',
                projection = go.layout.geo.Projection(
                    type='miller'
                ),
                showland = True,
                landcolor = 'rgb(217, 217, 217)',
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(255, 255, 255)",
                countrycolor="rgb(255, 255, 255)"
            )
        )
    }, 'Average rating between: "{}"'.format(range)

if __name__ == '__main__':
    app.run_server(debug=True)