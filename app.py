import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yahoo_fin.stock_info as si
from dash import Dash, html, dcc, Input, Output
from gmo_stonk_utils import gmostonk

np.seterr(divide='ignore')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

tickers = si.tickers_dow()
tickers.append('SPY')
try:
    stock_obj = gmostonk(tickers, saved_locally=True)
except FileNotFoundError as err:
    stock_obj = gmostonk(tickers, saved_locally=False)

df = stock_obj.tickers_df
df['Date'] = pd.to_datetime(df['Date'])

indicators = 'Open High Low Close Volume'.split()
indicators.sort()

app.layout = html.Div([
    html.H1('GMO DOW Companies Website v1'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                options=tickers,
                value='SPY',
                id='crossfilter-ticker1-column',
            ),
            dcc.Dropdown(
                options=indicators,
                value=indicators[0],
                id='ticker1-indicator'
            ),
            dcc.RadioItems(
                options=['Linear', 'Log'],
                value='Linear',
                id='crossfilter-ticker1-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                options=tickers,
                value='AAPL',
                id='crossfilter-ticker2-column'
            ),
            dcc.Dropdown(
                options=indicators,
                value=indicators[0],
                id='ticker2-indicator'
            ),
            dcc.RadioItems(
                options=['Linear', 'Log'],
                value='Linear',
                id='crossfilter-ticker2-type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'AAPL'}]}
        ),
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.RadioItems(options=['Returns', 'Value'],
                            value='Returns',
                            inline=True,
                            id='value-returns'),
             style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div(dcc.RangeSlider(
        min=df['Date'].dt.year.min(),
        max=df['Date'].dt.year.max(),
        step=None,
        value=[df['Date'].dt.year.min(), df['Date'].dt.year.max()],
        id='year-range-slider',
        marks={str(year): str(year) for year in df['Date'].dt.year.unique()}
    ), style={'width': '49%', 'padding': '0px 0px 20px 20px'}),
     html.Center(dcc.Markdown(id='regression-results', dangerously_allow_html=True),
       style={'textalign': 'center'})
])


@app.callback(
    [Output('crossfilter-indicator-scatter', 'figure'),
     Output('regression-results', 'children')],
    Input('crossfilter-ticker1-column', 'value'),
    Input('crossfilter-ticker2-column', 'value'),
    Input('ticker1-indicator', 'value'),
    Input('ticker2-indicator', 'value'),
    Input('crossfilter-ticker1-type', 'value'),
    Input('crossfilter-ticker2-type', 'value'),
    Input('value-returns', 'value'),
    Input('year-range-slider', 'value'))
def update_graph(ticker1_column_name, ticker2_column_name,
                 ticker1_indicator, ticker2_indicator,
                 ticker1_type, ticker2_type,
                 val_return, year_values):
    dff = df[((df['Date'].dt.year >= year_values[0]) &
              (df['Date'].dt.year <= year_values[1]))]

    xdata = dff[dff['Ticker'] == ticker1_column_name]
    ydata = dff[dff['Ticker'] == ticker2_column_name]

    if val_return == 'Returns':
        ticker1_indicator = ticker1_indicator + '_r'
        ticker2_indicator = ticker2_indicator + '_r'

    ticker1_column_name = ticker1_column_name + '_' + ticker1_indicator
    ticker2_column_name = ticker2_column_name + '_' + ticker2_indicator

    fit_info = stock_obj.OLS_regression(xdata, ydata, ticker1_indicator, ticker2_indicator)
    unleveraged = fit_info['Unleveraged Data']
    high_leverage = fit_info['High Leverage Data (HL)']
    x_fit = fit_info['x_fit']
    y_fit = fit_info['y_fit']
    regres = fit_info['reg_fit_summary']

    fig1 = px.line(
        x=x_fit,
        y=y_fit,
        color_discrete_sequence=['lime']
    )
    fig1.update_xaxes(title=ticker1_column_name, type='linear' if ticker1_type == 'Linear' else 'log')
    fig1.update_yaxes(title=ticker2_column_name, type='linear' if ticker2_type == 'Linear' else 'log')
    fig1.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig2 = px.scatter(
        x=unleveraged[ticker1_column_name],
        y=unleveraged[ticker2_column_name],
        color_discrete_sequence=['blue']
    )
    fig2.update_xaxes(title=ticker1_column_name, type='linear' if ticker1_type == 'Linear' else 'log')
    fig2.update_yaxes(title=ticker2_column_name, type='linear' if ticker2_type == 'Linear' else 'log')
    fig2.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig3 = px.scatter(
        x=high_leverage[ticker1_column_name],
        y=high_leverage[ticker2_column_name],
        color_discrete_sequence=['red']
    )
    fig3.update_xaxes(title=ticker1_column_name, type='linear' if ticker1_type == 'Linear' else 'log')
    fig3.update_yaxes(title=ticker2_column_name, type='linear' if ticker2_type == 'Linear' else 'log')
    fig3.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig = go.Figure(data=fig2.data + fig3.data + fig1.data, layout=fig1.layout)
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name'] = 'Normally Leverage'
    fig['data'][1]['showlegend'] = True
    fig['data'][1]['name'] = 'High Leverage'
    fig['data'][2]['showlegend'] = True
    fig['data'][2]['name'] = 'OLS Regression'
    return fig, regres


def create_time_series(dff, axis_type, indicator, title):
    fig = px.scatter(dff, x='Date', y=indicator)
    fig.update_traces(mode='lines')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-ticker1-column', 'value'),
    Input('ticker1-indicator', 'value'),
    Input('crossfilter-ticker1-type', 'value'),
    Input('year-range-slider', 'value'))
def update_y_timeseries(hoverData, ticker1_column_name, ticker1_indicator, axis_type, year_values):
    # hdata = hoverData['points'][0]['customdata']
    dff = df[df['Ticker'] == ticker1_column_name]
    dff = dff[((dff['Date'].dt.year >= year_values[0]) &
               (dff['Date'].dt.year <= year_values[1]))]
    dff = dff[['Date', ticker1_indicator]]
    title = '<b>{}</b><br>{}'.format(ticker1_column_name, ticker1_column_name)
    return create_time_series(dff, axis_type, ticker1_indicator, ticker1_column_name)


@app.callback(
    Output('y-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-ticker2-column', 'value'),
    Input('ticker2-indicator', 'value'),
    Input('crossfilter-ticker2-type', 'value'),
    Input('year-range-slider', 'value'))
def update_x_timeseries(hoverData, ticker2_column_name, ticker2_indicator, axis_type, year_values):
    # hdata = hoverData['points'][0]['customdata']
    dff = df[df['Ticker'] == ticker2_column_name]
    dff = dff[((dff['Date'].dt.year >= year_values[0]) &
               (dff['Date'].dt.year <= year_values[1]))]
    dff = dff[['Date', ticker2_indicator]]
    return create_time_series(dff, axis_type, ticker2_indicator, ticker2_column_name)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)#, debug=False, dev_tools_silence_routes_logging=True)
