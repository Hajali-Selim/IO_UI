from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import zscore
from plotly.subplots import make_subplots
from copy import deepcopy

external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

H = pd.read_csv('processed_data_collapsed.csv', sep='\t')
Hv = pd.read_csv('processed_data2_collapsed.csv', sep='\t')

NS, NC, NY = 72, 49, 26
N, Ntot = NS*NC, NS*NC*NY
regions, countries, sectors, transition, abate = H.region.iloc[np.arange(0,N,NS)], H.country.unique(), H.sector.unique(), H.transition.unique(), H.abate.unique()
legend_names = {'circle':'uninvolved', 'square':'Transition mineral', 'diamond':'Transition process'}
regions_list, countries_list, sectors_list = list(regions.unique()), list(countries), list(sectors)
region_to_country = {txt: list(np.where(np.array(regions)==txt)[0]) for txt in regions_list}

sector_to_idx, country_to_idx, k1, k2 = {}, {}, 0, 0
for s in sectors:
    sector_to_idx[s], k1 = k1, k1+1

for c in countries:
    country_to_idx[c], k2 = k2, k2+1

units_list = ['region', 'country', 'sector']
vulnerability_list, structural_list, monetary_list = ['coal', 'gas', 'oil', 'all three fossil fuels'], ['in-degree', 'out-degree', 'weighted in-degree', 'weighted out-degree', 'betweenness'], ['forward linkage', 'backward linkage']
metrics_list = vulnerability_list+structural_list+monetary_list
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list], Hv[metrics_list] = H[metrics_list].astype(float), Hv[metrics_list].astype(float)
for s in ['year','transition']+units_list:
    units = list(range(1995,2023))*(s=='year') + regions_list*int(s=='region') + countries_list*(s=='country') + sectors_list*(s=='sector') + ['circle','square','diamond']*(s=='transition')
    H[s], Hv[s] = H[s].astype('category').cat.set_categories(units, ordered=True), Hv[s].astype('category').cat.set_categories(units, ordered=True)

data_cavg = H.groupby(['year','country'])[metrics_list].mean().reset_index().sort_values(['year','country'])
data_s1avg = H.groupby(['year','sector'])[metrics_list].mean().reset_index().sort_values(['year','sector'])
data_s2avg = {c[0]:c[1].groupby(['year','sector'])[metrics_list].mean().reset_index().sort_values(['year','sector']) for c in H.groupby('region')}

sector_groups = ['Agriculture', 'Extraction and mining', 'Manufacture', 'Utilities', 'Services']
sector_groupmap = {'Agriculture':[0], 'Extraction and mining':list(range(1,16)), 'Manufacture':list(range(16,48)), 'Utilities':list(range(48,64)), 'Services':list(range(64,72))}
data_y = {'standard':{int(c[0]):c[1] for c in H.groupby(['year'])[units_list+metrics_list]}, 'yearly variation':{int(c[0]):c[1] for c in Hv.groupby(['year'])[units_list+metrics_list]}}
height, width = {'country':17, 'sector':16.5, 'region':16.5}, {'country':26*25, 'sector':26*30, 'region':26*75}

app.layout = html.Div([
    dcc.Markdown('''# visualizing vulnerability''', style={'textAlign':'center'}),
    html.Hr(),
    dbc.Accordion([
    	dbc.AccordionItem(
    	dbc.Tabs([
    		dbc.Tab([
    		html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(# column of parameter selection
    		dbc.Table(html.Tbody([
    			html.Tr([html.Td(dcc.Markdown('**select type**', style={'textAlign':'right'})), 
    				html.Td(dcc.Dropdown(['standard', 'yearly variation'], 'standard', id='type1v'))]),
    			html.Tr([html.Td(dcc.Markdown('**select energy**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(vulnerability_list, 'coal', id='metric1v'))]),
    			html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'})),
     				html.Td(dcc.Dropdown(units_list, 'country', id='unit1v'))]),
     			html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(list(range(1995,2023)), 2000, id='year1v'))]), ]), borderless=True), width=4),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=2),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1v', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=6)]),
    			]),
    		dcc.Graph(figure={}, id='hist1v'), ], label='Vulnerability', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([
    		html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(# column of parameter selection
    		dbc.Table(html.Tbody([
    			html.Tr([html.Td(dcc.Markdown('**select type**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(['standard', 'yearly variation'], 'standard', id='type1s'))]),
    			html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(structural_list, 'out-degree', id='metric1s'))]),
    			html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'})),
     				html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1s'))]),
     			html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(list(range(1995,2023)), 2000, id='year1s'))]), ]), borderless=True), width=4),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=2),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1s', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=6)]),]),
    		dcc.Graph(figure={}, id='hist1s'), ], label='Structural importance', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([
    		html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(dbc.Table(html.Tbody([
    				html.Tr([html.Td(dcc.Markdown('**select type**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(['standard', 'yearly variation'], 'standard', id='type1m'))]),
    				html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(monetary_list, 'forward linkage', id='metric1m'))]),
    				html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'})),
     					html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1m'))]),
     				html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(list(range(1995,2023)), 2000, id='year1m'))]), ]), borderless=True), width=4),
		dbc.Col([dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=2),
    			dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1m', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=4)]),]),
    		dcc.Graph(figure={}, id='hist1m'), ], label='Monetary importance', activeTabClassName='fw-bold'),
    		]), title='Yearly histogram averages'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([html.Br(),
    				dbc.Offcanvas(dcc.Checklist(countries, countries, id='unit2_c', inputStyle={'margin-right':'10px'}), id='canvas2_c', is_open=False, placement='end', scrollable=True),
    				dbc.Col(dbc.Button('Select countries', id='open_canvas2_c', n_clicks=0), width=2),
    				
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_c'), width=2),]),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=3)]),
    				dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab(dbc.Tabs([dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					dcc.Checklist(sectors, sectors, id='unit2_s1', inputStyle={'margin-right':'10px'})], id='canvas2_s1', is_open=False, placement='end', scrollable=True),
    					dbc.Col(dbc.Button('Select sectors', id='open_canvas2_s1', n_clicks=0)),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_s1'), width=2),]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=3)]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s1')),
    					], label='Worldwide averages', activeTabClassName='fw-bold'),
    					
    					dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					dcc.Checklist(sectors, sectors, id='unit2_s2', inputStyle={'margin-right':'10px'})], id='canvas2_s2', is_open=False, placement='end', scrollable=True),
    					dbc.Col(dbc.Button('Select sectors', id='open_canvas2_s2', n_clicks=0)),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_s2'), width=2),]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=3)]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s2')),
    				], label='Regional averages', activeTabClassName='fw-bold'),])
    			, label='Sectors', activeTabClassName='fw-bold')])
    			, title='Heatmaps'),
    	
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab(dbc.Tabs([
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric3_c1'), width=2),]),
    				dbc.Row([dbc.Col(dcc.Markdown('**select/unselect region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=4)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_c1'))], label='worldwide', activeTabClassName='fw-bold'),
    			
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'oil', id='metric3_c2'), width=2)],),
    				dbc.Row([dbc.Col(dcc.Markdown('**select sector**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(sectors, 'Plastic prod.', id='unit3_c2'), width=2)]),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select/unselect region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=4)]),
	    			dbc.Row(dcc.Graph(figure={}, id='lines3_c2'))], label='for a same sector', activeTabClassName='fw-bold')]
	    		), label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab(dbc.Tabs([
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric3_s1'), width=2)],),
	    			dbc.Row(dcc.Graph(figure={}, id='lines3_s1'))], label='worldwide', activeTabClassName='fw-bold'),
    				
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'gas', id='metric3_s2'), width=2),]),
    				dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(countries, 'USA', id='unit3_s2'), width=2)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_s2'))], label='within a single country', activeTabClassName='fw-bold')]
    			), label='Sectors', activeTabClassName='fw-bold')])
    		, title='Line charts'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select axes**', style={'textAlign':'right'}), width=2),
			dbc.Col(dcc.RadioItems(['Monetary importance', 'Structural importance'], 'Monetary importance', inline=False, inputStyle={'margin-right':'10px', 'margin-bottom':'10px'}, id='metric4t'))]),
		dbc.Row([dbc.Col(dcc.Markdown('**select vulnerability**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric4v'), width=2)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Dropdown(countries, 'Australia', id='unit4'), width=2)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Slider(1995, 2022, 1, marks={y:str(y) for y in range(1995,2023)}, updatemode='drag', value=1996, id='year4'))]),
    		#dbc.Row([dbc.Col(dcc.Markdown('**rescale marker size**', style={'textAlign':'right'}), width=2),
    		#	dbc.Col(dcc.Slider(1,3,.5, updatemode='drag',value=1, id='marker_size4'), width=2)]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4')), ], title='Bubble plots'),
    	
    	dbc.AccordionItem([
    			
    			], title='Waves: national vulnerability trends'),
			], flush=True)
			])


@callback(Output('hist1v', 'figure'), [Input(s, 'value') for s in ['type1v','metric1v','unit1v','year1v','order1v']])
def update_vhistogram(type1v,metric1v,unit1v,year1v,order1v):
    order = (order1v=='original')*'trace' + (order1v=='descending')*'total descending' + (order1v=='ascending')*'total ascending'
    return px.histogram(data_y[type1v][year1v].reset_index(), x=unit1v, y=metric1v, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1v=='sector')*45+(unit1v=='country')*30).update_layout(yaxis_title=metric1v+' vulnerability', font={'size':18}, height=500).update_traces(hovertemplate='%{x}, '+str(year1v)+'<br>%{y}')

@callback(Output('hist1s', 'figure'), [Input(s, 'value') for s in ['type1s','metric1s','unit1s','year1s','order1s']])
def update_shistogram(type1s,metric1s,unit1s,year1s,order1s):
    order = (order1s=='original')*'trace' + (order1s=='descending')*'total descending' + (order1s=='ascending')*'total ascending'
    return px.histogram(data_y[type1s][year1s].reset_index(), x=unit1s, y=metric1s, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1s=='sector')*45+(unit1s=='country')*30).update_layout(yaxis_title=metric1s, font={'size':18}, height=600).update_traces(hovertemplate='%{x}, '+str(year1s)+'<br>%{y}')

@callback(Output('hist1m', 'figure'), [Input(s, 'value') for s in ['type1m','metric1m','unit1m','year1m','order1m']])
def update_mhistogram(type1m,metric1m,unit1m,year1m,order1m):
    order = (order1m=='original')*'trace' + (order1m=='descending')*'total descending' + (order1m=='ascending')*'total ascending'
    return px.histogram(data_y[type1m][year1m].reset_index(), x=unit1m, y=metric1m, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1m=='sector')*45+(unit1m=='country')*30).update_layout(yaxis_title=metric1m+' vulnerability', font={'size':18}, height=600).update_traces(hovertemplate='%{x}, '+str(year1m)+'<br>%{y}')

@callback(Output('heatmap2_c', 'figure'), [Input(s, 'value') for s in ['metric2_c','order2_c','unit2_c']])
def update_cheatmap(metric2_c,order2_c,unit2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    cinds = [country_to_idx[c] for c in unit2_c]
    data_c = deepcopy(data_cavg.iloc[sorted([k*NC+i for k in range(NY) for i in cinds])])
    return px.density_heatmap(data_c, x='year', y='country', z=metric2_c, height=height['country']*len(cinds), width=width['country'], labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=len(unit2_c)).update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)'})

@callback(Output('heatmap2_s1', 'figure'), [Input(s, 'value') for s in ['metric2_s1','group2_s1', 'order2_s1','unit2_s1']])
def update_s1heatmap(metric2_s1,group2_s1,order2_s1,unit2_s1):
    order = (order2_s1=='original')*'trace' + (order2_s1=='descending')*'total descending' + (order2_s1=='ascending')*'total ascending'
    g1inds = sum([sector_groupmap[g] for g in group2_s1],[])
    s1inds = list(set([sector_to_idx[s] for s in unit2_s1]) & set(g1inds))
    data_s1 = deepcopy(data_s1avg.iloc[sorted([k*NS+i for k in range(NY) for i in s1inds])])
    return px.density_heatmap(data_s1, x='year', y='sector', z=metric2_s1, height=height['sector']*len(s1inds), width=width['sector'], labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=len(s1inds)).update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)'})

@callback(Output('heatmap2_s2', 'figure'), [Input(s, 'value') for s in ['metric2_s2','group2_s2','order2_s2','unit2_s2']])
def update_s2heatmap(metric2_s2,group2_s2,order2_s2,unit2_s2):
    order = (order2_s2=='original')*'trace' + (order2_s2=='descending')*'total descending' + (order2_s2=='ascending')*'total ascending'
    g2inds = sum([sector_groupmap[g] for g in group2_s2],[])
    s2inds = list(set([sector_to_idx[s] for s in unit2_s2])&set(g2inds))
    data_s2 = {r:deepcopy(data_s2avg[r].iloc[sorted([k*NS+i for k in range(NY) for i in s2inds])]) for r in regions_list}
    fig = make_subplots(rows=1, cols=5, horizontal_spacing=.005, shared_yaxes=True, subplot_titles=tuple(regions_list))
    for r_idx in range(len(regions_list)):
        r = regions_list[r_idx]
        fig.add_trace(px.density_heatmap(data_s2[r], x='year', y='sector', z=metric2_s2, nbinsx=NY, nbinsy=len(s2inds)).data[0], row=1, col=r_idx+1)
    fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(height=height['region']*len(s2inds), width=width['region'], font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)'})
    return fig

@callback(Output('lines3_c1', 'figure'), [Input(s, 'value') for s in ['metric3_c1','region3_c1']])
def update_c1line(metric3_c1,region3_c1):
    inds = [i for r in region3_c1 for i in region_to_country[r]]
    c1_indices = sorted([k for j in inds for k in np.arange(j,NY*NC,NC)])
    data3_slice = deepcopy(data_cavg.iloc[c1_indices])
    data3_slice['country'] = data3_slice['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    return px.line(data3_slice, x='year', y=metric3_c1, color='country', labels={'x':'year', 'y':metric3_c1+' vulnerability (%)'*int(metric3_c1 in vulnerability_list), 'color':'country'}, markers=True, hover_name='country').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('lines3_c2', 'figure'), [Input(s, 'value') for s in ['metric3_c2','unit3_c2','region3_c2']])
def update_c2line(metric3_c2,unit3_c2,region3_c2):
    return px.line(H[H['sector']==unit3_c2], x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2+' vulnerability (%)'*int(metric3_c2 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='country').update_xaxes(tickvals= np.arange(1995,2023,3) ).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('lines3_s1', 'figure'), Input('metric3_s1', 'value'))
def update_s1line(metric3_s1):
    return px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1+' vulnerability (%)'*int(metric3_s1 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='sector').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('lines3_s2', 'figure'), [Input(s, 'value') for s in ['metric3_s2','unit3_s2']])
def update_s2line(metric3_s2,unit3_s2):
    return px.line(H[H['country']==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2+' vulnerability (%)'*int(metric3_s2 in vulnerability_list), 'color':'country'}, markers=True, hover_name='sector').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font= {'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('bubble4', 'figure'), [Input(s, 'value') for s in ['metric4t','metric4v', 'unit4','year4']])
def update_bubble(metric4t,metric4v,unit4,year4):#,marker_size4):
    if metric4t == 'Monetary importance':
        xlabel, ylabel = 'backward linkage', 'forward linkage'
    elif metric4t == 'Structural importance':
        xlabel, ylabel = 'out-degree', 'betweenness'
    country_dataset, zlabel = H[H['country']==unit4], metric4v
    xmin, xmax, ymin, ymax, zmin, zmax = country_dataset[xlabel].min(), country_dataset[xlabel].max(), country_dataset[ylabel].min(), country_dataset[ylabel].max(), country_dataset[zlabel].min(), country_dataset[zlabel].max()
    dataset = country_dataset[country_dataset['year']==year4]
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='transition', opacity=.8).update_layout(width=1200, height=800, font={'size':20}, coloraxis_colorbar={'title':'vulnerability (%)'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).update_traces(marker_line_width=dataset['abate']*3, marker_line_color='red')#, marker={'size':marker_size4*dataset[zlabel]})
    for i,j in enumerate(legend_names):
        fig.data[i].name=legend_names[j]
    if metric4t == 'Monetary importance':
        fig.add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    return fig

@app.callback(Output('canvas2_c', 'is_open'), Input('open_canvas2_c', 'n_clicks'), State('canvas2_c', 'is_open'))
def toggle_ccanvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas2_s1', 'is_open'), Input('open_canvas2_s1', 'n_clicks'), State('canvas2_s1', 'is_open'))
def toggle_s1canvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas2_s2', 'is_open'), Input('open_canvas2_s2', 'n_clicks'), State('canvas2_s2', 'is_open'))
def toggle_s2canvas(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)




