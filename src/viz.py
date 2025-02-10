import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px


def build_dashboard(df):
    app = Dash(__name__)

    app.layout = html.Div([

        # Title
        html.Div(children='Crab Age Prediction Dataset',
                 style={'textAlign': 'Center', 'color': 'blue', 'fontSize': 30}),
        html.Hr(),
        html.Div(children='Choose a feature to visualize its distribution',
                 style={'textAlign': 'Left', 'color': 'blue', 'fontSize': 18}),
        dcc.Dropdown(df.columns, df.columns[0], id='feature_selector_dropdown'),
        dcc.Graph(id='bar_graph'),

        # html.Div(children=dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]),
        #          style={'textAlign': 'Center', 'color': 'blue', 'fontSize': 24}),
    ])

    @app.callback(
        Output('bar_graph', 'figure'),
        Input('feature_selector_dropdown', 'value'),
    )
    def update_bar_graph(feature):
        data = df[feature]
        fig = px.histogram(data)
        return fig

    app.run()