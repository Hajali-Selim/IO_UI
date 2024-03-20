from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_daq as daq
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
import networkx as nx
from itertools import chain

#pd.options.mode.copy_on_write = True
pd.options.mode.chained_assignment = None
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

H = pd.read_csv('processed_data.csv', compression='gzip', sep=',', index_col=[0])
NS, NC, NY = 163, 49, 26
N, Ntot = NS*NC, NS*NC*NY
regions, countries, sectors = H.region.iloc[np.arange(0,N,NS)], H.country.unique(), H.sector.unique()
regions_list, countries_list, sectors_list, sectors_decarb = list(regions.unique()), list(countries), list(sectors), H[H.sector_color.notna()].sector.unique()
region_to_country = {txt: list(np.where(np.array(regions)==txt)[0]) for txt in regions_list}

sector_to_idx, country_to_idx, k1, k2 = {}, {}, 0, 0
for s in sectors:
    sector_to_idx[s], k1 = k1, k1+1

for c in countries:
    country_to_idx[c], k2 = k2, k2+1

units_list = ['region', 'country', 'sector']
vulnerability_list, structural_list, monetary_list = ['coal', 'gas', 'oil', 'vulnerability_index'], ['out_degree', 'in_degree', 'total_degree', 'betweenness', 'weighted_in_degree', 'weighted_out_degree', 'weighted_total_degree', 'structural_index'], ['forward_linkage', 'backward_linkage', 'monetary_index']

metrics_list = vulnerability_list+structural_list+monetary_list
ordering_list = ['original', 'descending', 'ascending']
#shifts = {'country': {fuel: {y1: np.round(pd.read_csv('shifts/countries/'+fuel+'/from_'+str(y1)+'.csv', sep=','),2) for y1 in range(1995,2020)} for fuel in vulnerability_list}, 'sector': {fuel: {y1: np.round(pd.read_csv('shifts/sectors/'+fuel+'/from_'+str(y1)+'.csv', sep=','),2) for y1 in range(1995,2020)} for fuel in vulnerability_list}}

H[metrics_list] = H[metrics_list].astype(float)
for s in ['year']+units_list+['region_marker']:#, 'sector_color']:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)
    #H[s] = pd.Series(H[s], dtype='category')

# leaving only exogeneous vulnerability
fossil_fuels = {'coal':'Mining of coal and lignite; extraction of peat (10)', 'oil':'Extraction of crude petroleum and services related to crude oil extraction, excluding surveying', 'gas':'Extraction of natural gas and services related to natural gas extraction, excluding surveying'}
for s in fossil_fuels:
    s_ext = fossil_fuels[s]
    H[H.sector==s_ext][s] = np.zeros(NC*NY,float)

data_cavg = H.groupby([('year'),('country')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('country')])
data_s1avg = H.groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')])
data_s2avg = {c[0]:c[1].groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')]) for c in H.groupby('region', observed=False)}

year_NYlist, sector_NYlist = np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil','gas','coal'] for i in range(NY)])
data_cvuln = {}
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['country']==c][['oil','gas','coal']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

sector_groups = ['Agriculture, forestry and fishing', 'Extraction and mining', 'Manufacture and production', 'Utilities', 'Services']
sector_groupmap = {'Agriculture, forestry and fishing':list(range(19)), 'Extraction and mining':list(range(19,34)), 'Manufacture and production':list(range(34,93))+[109,113], 'Utilities':list(range(93,119))+[110,111,112]+list(range(114,120)), 'Services':list(range(120,163))}
data_y = {'standard':{int(c[0][0]):c[1] for c in H.groupby(['year'], observed=False)[units_list+metrics_list]}}
height, width = {'country':17, 'sector':16.5, 'region':16.5}, {'country':27*25, 'sector':27*30, 'region':27*75}

app.layout = html.Div(children=[
    dcc.Markdown('''# visualizing vulnerability v2.0''', style={'textAlign':'center'}),
    html.Hr(),
    dbc.Accordion([
    	dbc.AccordionItem(
    	dbc.Tabs([
    		dbc.Tab([html.Br(),
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
    				html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1v'))]), ]), borderless=True), width=4),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=2),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1v', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=6)]),
    			]),
    		dcc.Graph(figure={}, id='hist1v'), ], label='Vulnerability', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(# column of parameter selection
    		dbc.Table(html.Tbody([
    			html.Tr([html.Td(dcc.Markdown('**select type**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(['standard', 'yearly variation'], 'standard', id='type1s'))]),
    			html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(structural_list, 'out_degree', id='metric1s'))]),
    			html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'})),
     				html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1s'))]),
     			html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'})),
    				html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1s'))]), ]), borderless=True), width=4),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=2),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1s', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=6)]),]),
    		dcc.Graph(figure={}, id='hist1s'), ], label='Structural importance', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(dbc.Table(html.Tbody([
    				html.Tr([html.Td(dcc.Markdown('**select type**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(['standard', 'yearly variation'], 'standard', id='type1m'))]),
    				html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(monetary_list, 'forward_linkage', id='metric1m'))]),
    				html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'})),
     					html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1m'))]),
     				html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'})),
    					html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1m'))]), ]), borderless=True), width=4),
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
    					dbc.Col(dcc.Dropdown(sectors, 'Plastics, basic', id='unit3_c2'), width=2)]),
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
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    dbc.Tab([
    	    html.Br(), dbc.Offcanvas(dcc.Checklist(countries, countries, id='unit4_s1c', inputStyle={'margin-right':'10px'}), id='canvas4_s1c', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Select countries', id='open_canvas4_s1c', n_clicks=0), width=2),
    	    
    	    html.Br(), dbc.Offcanvas(dcc.Checklist(sectors, sectors_decarb, id='unit4_s1', inputStyle={'margin-right':'10px'}), id='canvas4_s1', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Select sectors', id='open_canvas4_s1', n_clicks=0), width=2),
    		
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'forward_linkage', id='metric4x_s1'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'backward_linkage', id='metric4y_s1'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select index (size)**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(['monetary_index', 'structural_index'], 'structural_index', inline=False, inputStyle={'margin-right':'10px', 'margin-bottom':'10px'}, id='metric4i_s1'))]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=1998, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s1')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s1'))], label='worldwide', activeTabClassName='fw-bold'),
    	    
    	    dbc.Tab([html.Br(),
    	    dbc.Offcanvas(dcc.Checklist(sectors, sectors_decarb, id='unit4_s2', inputStyle={'margin-right':'10px'}), id='canvas4_s2', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Select sectors', id='open_canvas4_s2', n_clicks=0), width=2),
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'forward_linkage', id='metric4x_s2'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'backward_linkage', id='metric4y_s2'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select country (unit)**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(countries, 'USA', id='unit4_s2c'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select index (size)**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(['monetary_index', 'structural_index'], 'structural_index', inline=False, inputStyle={'margin-right':'10px', 'margin-bottom':'10px'}, id='metric4i_s2'))]),
		    dbc.Row([dbc.Col(dcc.Markdown('**select vulnerability (color)**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric4v_s2'), width=2)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=1998, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s2')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s2'))], label='within a single country', activeTabClassName='fw-bold')    		
    		]), title='Bubble plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
			dbc.Col(dcc.Dropdown(countries_list, 'Germany', id='unit5'), width=2)]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Waves'),
    	
			], flush=True)
			])


@callback(Output('hist1v', 'figure'), [Input(s, 'value') for s in ['type1v','metric1v','unit1v','year1v','order1v']])
def update_vhistogram(type1v,metric1v,unit1v,year1v,order1v):#reinclude type1v, change data_y[year1v] by data_y[type1v][year1v] after reincluding yearly variations
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
    data_c = deepcopy(data_cavg.iloc[sorted([k*NC+i for k in range(NY) for i in cinds])])# FIX HERE
    return px.density_heatmap(data_c, x='year', y='country', z=metric2_c, height=height['country']*len(cinds), width=width['country'], labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=len(unit2_c), color_continuous_scale='Turbo').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'})

@callback(Output('heatmap2_s1', 'figure'), [Input(s, 'value') for s in ['metric2_s1','group2_s1', 'order2_s1','unit2_s1']])
def update_s1heatmap(metric2_s1,group2_s1,order2_s1,unit2_s1):
    order = (order2_s1=='original')*'trace' + (order2_s1=='descending')*'total descending' + (order2_s1=='ascending')*'total ascending'
    g1inds = sum([sector_groupmap[g] for g in group2_s1],[])
    s1inds = list(set([sector_to_idx[s] for s in unit2_s1]) & set(g1inds))
    data_s1 = deepcopy(data_s1avg.iloc[sorted([k*NS+i for k in range(NY) for i in s1inds])])
    return px.density_heatmap(data_s1, x='year', y='sector', z=metric2_s1, height=height['sector']*len(s1inds), width=width['sector'], labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=len(s1inds), color_continuous_scale='Jet').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'})

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
    fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(height=height['region']*len(s2inds), width=width['region'], font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, coloraxis={'colorscale':'Jet'})
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
    fig = px.line(H[H['sector']==unit3_c2], x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2+' vulnerability (%)'*int(metric3_c2 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='country').update_xaxes(tickvals=np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    try:
        return fig
    except:
        return fig

@callback(Output('lines3_s1', 'figure'), Input('metric3_s1', 'value'))
def update_s1line(metric3_s1):
    fig = px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1+' vulnerability (%)'*int(metric3_s1 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='sector').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    try:
        return fig
    except:
        return fig

@callback(Output('lines3_s2', 'figure'), [Input(s, 'value') for s in ['metric3_s2','unit3_s2']])
def update_s2line(metric3_s2,unit3_s2):
    fig = px.line(H[H.country==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2+' vulnerability (%)'*int(metric3_s2 in vulnerability_list), 'color':'country'}, markers=True, hover_name='sector').update_xaxes(tickvals=np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font= {'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    try:
        return fig
    except:
        return fig

@callback(Output('bubble4_s1', 'figure'), [Input(s, 'value') for s in ['metric4x_s1','metric4y_s1','metric4i_s1','unit4_s1c','unit4_s1','year4_s1']])
def update_s1bubble(metric4x_s1,metric4y_s1,metric4i_s1,unit4_s1c,unit4_s1,year4_s1):
    xlabel, ylabel, zlabel = metric4x_s1, metric4y_s1, metric4i_s1
    cinds, sinds = [country_to_idx[c] for c in unit4_s1c], [sector_to_idx[s] for s in unit4_s1]
    dataset = deepcopy(H.iloc[sorted([c*NS+s for c in cinds for s in sinds])])
    xmin, xmax, ymin, ymax, zmin, zmax = dataset[xlabel].min(), dataset[xlabel].max(), dataset[ylabel].min(), dataset[ylabel].max(), dataset[zlabel].min(), dataset[zlabel].max()
    xmin, xmax, ymin, ymax = .5, 2, .5, 2
    dataset_y = dataset[dataset.year==year4_s1]
    dataset_y = dataset_y[dataset_y.sector_color.notna()]
    fig = px.scatter(dataset_y, x=xlabel, y=ylabel, size=zlabel, color='sector_color', labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region', opacity=.8, color_continuous_scale='Jet').update_layout(width=1200, height=800, font={'size':20}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None})
    fig.add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    try:
        return fig
    except:
        return fig

@callback(Output('bubble4_s2', 'figure'), [Input(s, 'value') for s in ['metric4x_s2','metric4y_s2','unit4_s2c','metric4i_s2','metric4v_s2','unit4_s2','year4_s2']])
def update_s2bubble(metric4x_s2,metric4y_s2,unit4_s2c,metric4i_s2,metric4v_s2,unit4_s2,year4_s2):
    xlabel, ylabel, zlabel = metric4x_s2, metric4y_s2, metric4i_s2
    dataset = H[H.country==unit4_s2c]
    sinds = [sector_to_idx[s] for s in unit4_s2]
    dataset = deepcopy(H.iloc[sorted([country_to_idx[unit4_s2c]*NS+s for s in sinds])])
    xmin, xmax, ymin, ymax, zmin, zmax = dataset[xlabel].min(), dataset[xlabel].max(), dataset[ylabel].min(), dataset[ylabel].max(), dataset[zlabel].min(), dataset[zlabel].max()
    xmin, xmax, ymin, ymax = .5, 2, .5, 2
    dataset_y = dataset[dataset.year==year4_s2]
    dataset_y = dataset_y[dataset_y.sector_color.notna()]
    fig = px.scatter(dataset_y, x=xlabel, y=ylabel, size=zlabel, color='sector_color', labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region', opacity=.8, color_continuous_scale='Jet').update_layout(width=1200, height=800, font={'size':20}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None})
    fig.add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    try:
        return fig
    except:
        return fig

@callback(Output('waves5', 'figure'), Input('unit5', 'value'))
def update_waves(unit5):
    #go.Figure(layout='template':'plotly')
    fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=1000, height=600)
    try:
        return fig
    except:
        return fig

#@app.callback(Output('rows7_c', 'figure'), [Input(s, 'value') for s in ['metric7_c', 'year7_c', 'nb_units7_c']])
#def update_crows(metric7, year7, nb_units7):
#    year7i, year7f = year7
#    dataset = deepcopy(shifts['country'][metric7][year7i])
#    dataset = dataset[dataset['year']==year7f]
#    sorted_c = dataset['country'].unique()
#    dataset['country'] = dataset['country'].astype('category').cat.set_categories(sorted_c, ordered=True)
#    pos_limit, neg_limit = sorted_c[nb_units7], sorted_c[-nb_units7-1]
#    pos_data, neg_data = dataset[dataset['country']<pos_limit], dataset[dataset['country']>neg_limit]
#    return px.bar(neg_data._append(pos_data), y='country', x='length', color='sector', orientation='h', custom_data=['country','sector','length'], text_auto='.2s', height=(nb_units7+1.5)*100).update_layout(font_size=14, hoverlabel={'font_size':22}).update_traces(hovertemplate='<extra></extra>%{customdata[0]}: <b>%{customdata[1]},</b><br>contribution to overall<br><b>'+metric7+'</b> vulnerability: %{customdata[2]}')

@app.callback(Output('canvas2_c', 'is_open'), Input('open_canvas2_c', 'n_clicks'), State('canvas2_c', 'is_open'))
def toggle_2ccanvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas2_s1', 'is_open'), Input('open_canvas2_s1', 'n_clicks'), State('canvas2_s1', 'is_open'))
def toggle_2s1canvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas2_s2', 'is_open'), Input('open_canvas2_s2', 'n_clicks'), State('canvas2_s2', 'is_open'))
def toggle_2s2canvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas4_s1c', 'is_open'), Input('open_canvas4_s1c', 'n_clicks'), State('canvas4_s1c', 'is_open'))
def toggle_4s1ccanvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas4_s1', 'is_open'), Input('open_canvas4_s1', 'n_clicks'), State('canvas4_s1', 'is_open'))
def toggle_4s1canvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas4_s2', 'is_open'), Input('open_canvas4_s2', 'n_clicks'), State('canvas4_s2', 'is_open'))
def toggle_4s2canvas(n, is_open):
    if n:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run_server(debug=True)
