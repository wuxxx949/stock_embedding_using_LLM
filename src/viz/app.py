import os

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
from sqlalchemy import create_engine

# Connect to your local database
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/db/dash.db')
db_engine = create_engine(f'sqlite:///{db_path}')

# Load data from the database
description_df = pd.read_sql_table('business_desc', con=db_engine) \
    .sort_values('ticker')
industry_prob_df = pd.read_sql_table('bert_industry_1536_prob', con=db_engine) \
    .sort_values(['ticker', 'prob'], ascending=True)

sector_prob_df = pd.read_sql_table('bert_sector_1536_prob', con=db_engine) \
    .sort_values(['ticker', 'prob'], ascending=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Multi GICS Dashboard"),
    html.Div([
        html.Label(
            "Select Company Ticker:",
            style={'font-weight': 'bold', 'width': '25%', 'display': 'inline-block'}),

        html.Label(
            "Select Industry or Sector:",
            style={'font-weight': 'bold', 'width': '25%', 'display': 'inline-block'}),
        ]),

    # Dropdown to select a company ticker
    html.Div([
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in description_df['ticker']],
            value=description_df['ticker'].iloc[0],  # Set default value
            # style={'width': '50%'},
            searchable=True,
            style={'width': '25%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='industry-sector-dropdown',
            options=[
                {'label': 'Industry', 'value': 'industry'},
                {'label': 'Sector', 'value': 'sector'},
            ],
            value='industry',  # Default selected option
            style={'width': '25%', 'display': 'inline-block'},
        ),
    ]),
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
    [Input('ticker-dropdown', 'value'),
     Input('industry-sector-dropdown', 'value')]
)
def update_company_info(selected_ticker: str, gics_type: str):
    # Filter data based on selected ticker
    selected_desc = description_df[description_df['ticker'] == selected_ticker]['business'].values[0]
    selected_desc = ' '.join(selected_desc.split(' ')[: 2000]) + '...'

    if gics_type == 'industry':
        selected_classification = industry_prob_df[industry_prob_df['ticker'] == selected_ticker]
    else:
        selected_classification = sector_prob_df[sector_prob_df['ticker'] == selected_ticker]

    # Create a bar chart
    figure = {
        'data': [
            {'y': selected_classification[gics_type],
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
