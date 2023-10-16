import os
from importlib.resources import files

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
from sqlalchemy import create_engine
import plotly.graph_objects as go

# Connect to your local database
db_path = os.path.join(files('src'), 'data/db/dash.db')
db_engine = create_engine(f'sqlite:///{db_path}')

# Load data from the database
description_df = pd.read_sql_table('business_desc', con=db_engine)
classification_df = pd.read_sql_table('bert_1536_prob', con=db_engine) \
    .sort_values(['ticker', 'prob'], ascending=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Multi GICS Dashboard"),

    # Dropdown to select a company ticker
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in description_df['ticker']],
        value=description_df['ticker'].iloc[0],  # Set default value
        style={'width': '50%'}
    ),
    # Bar chart for industry classification
    dcc.Graph(
        id='industry-chart',
        style={'width': '80%', 'margin': 'auto', 'margin-top': '20px'}
    ),

    # Display the company description
    html.Div(
        id='company-description',
        style={'height': '500px', 'overflowY': 'scroll'},
    )
])

# Callback to update description and chart based on selected ticker
@app.callback(
    [Output('company-description', 'children'),
     Output('industry-chart', 'figure')],
    [Input('ticker-dropdown', 'value')]
)
def update_company_info(selected_ticker):
    # Filter data based on selected ticker
    selected_desc = description_df[description_df['ticker'] == selected_ticker]['business'].values[0]
    selected_desc = ' '.join(selected_desc.split(' ')[: 2000]) + '...'
    selected_classification = classification_df[classification_df['ticker'] == selected_ticker]

    # Create a bar chart
    figure = {
        'data': [
            {'y': selected_classification['industry'],
             'x': selected_classification['prob'],
             'type': 'bar',
             'orientation': 'h',
             'name': 'Probability'}
        ],
        'layout': {
            'title': f'Industry Classification for {selected_ticker}',
            'yaxis': {'tickangle': -50, 'tickfont': {'size': 9}},
            'xaxis': {'title': 'Probability', 'tickformat': ',.0%'}
        }
    }

    return selected_desc, figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
