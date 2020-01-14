import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import pymongo
import plotly
import plotly.graph_objs as go

ACCESS_TOKEN = 'pk.eyJ1Ijoic2ltb25odmEiLCJhIjoiY2s1OXk2c3BhMTFyaDNwbGprYWowZmZhOSJ9.Alw4kvUDHu7cxG9hB_Ra9Q'

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

    trace = [go.Scattermapbox(lat=list(df['lat']),
                            lon=list(df['lng']),
                            mode='markers',
                            marker=go.scattermapbox.Marker(
                                size=9
                            ),
                            text=list(df['text'])
            )]

    return trace

fill_map(final_df)

layout = go.Layout(
    title='Hotel Reviews in Europe',
    autosize=True,
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        accesstoken=ACCESS_TOKEN,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=50.203745,
            lon=4.335677
        ),
        pitch=0,
        zoom=4,
    ),
    height=650
)

sorted_list = df.sort_values('Average_Score', ascending=False).head(10)

app.layout = html.Div([
    html.H2('Hotel ratings in Europe'),
    dcc.Graph(
         id='map'
    ),
    
    html.Div([
        html.H4('Filter on average rating'),
         dcc.RangeSlider(
            id='my-range-slider',
            min=1,
            max=10,
            step=0.1,
            value=[0, 10]
        ),
        html.Div(id='output-container-range-slider'),
        dcc.Graph(id="rating-plot",
            figure={
                'data': [{'y': sorted_list['Hotel_Name'].iloc[::-1], 'x': sorted_list['Average_Score'].iloc[::-1], 
                    'type': 'bar',
                    'orientation': 'h',
                    'name': 'Hightest Score'
                    }
                ],
                'layout': {
                    'title': 'Highest score'
                }
            }
        ),
    ]),
  
   
])

@app.callback(
    [Output('map', 'figure'),
    Output('output-container-range-slider', 'children')],
    [Input('my-range-slider', 'value')])
def update_output(range):
    filtered_df = df[df['Average_Score'].between(range[0], range[1])]
    traces = fill_map(filtered_df)
    return {
        'data': traces,
        'layout': go.Layout(
            autosize=True,
            hovermode='closest',
            mapbox=go.layout.Mapbox(
                accesstoken=ACCESS_TOKEN,
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=48.203745,
                    lon=4.335677
                ),
                pitch=0,
                zoom=4
            ),
             height=650
        )
    }, 'Average rating between: "{}"'.format(range)

if __name__ == '__main__':
    app.run_server(debug=True)