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


external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

H, Hv = pd.read_csv('processed_collapsed_data.csv', sep=','), pd.read_csv('processed_collapsed_data2.csv', sep=',')

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
    units = list(range(1995,1995+NY))*(s=='year') + regions_list*int(s=='region') + countries_list*(s=='country') + sectors_list*(s=='sector') + ['circle','square','diamond']*(s=='transition')
    H[s], Hv[s] = H[s].astype('category').cat.set_categories(units, ordered=True), Hv[s].astype('category').cat.set_categories(units, ordered=True)

data_cavg = H.groupby(['year','country'])[metrics_list].mean().reset_index().sort_values(['year','country'])
data_s1avg = H.groupby(['year','sector'])[metrics_list].mean().reset_index().sort_values(['year','sector'])
data_s2avg = {c[0]:c[1].groupby(['year','sector'])[metrics_list].mean().reset_index().sort_values(['year','sector']) for c in H.groupby('region')}

year_NYlist, sector_NYlist = np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil','gas','coal'] for i in range(NY)])
data_cvuln = {}
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['country']==c][['oil','gas','coal']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

ZCs, ZSs = pickle.load(open('ZC.txt','rb')), pickle.load(open('ZS.txt','rb'))

networktext_NC, networktext_NS = {}, {}
for y in range(1995,2021):
    networktext_NC[y], networktext_NS[y] = {}, {}
    rfl_NC, rbl_NC, rout_NC, rin_NC = np.array(np.round(data_cavg[data_cavg['year']==y]['forward linkage'],2)), np.array(np.round(data_cavg[data_cavg['year']==y]['backward linkage'],2)), np.array(np.round(data_cavg[data_cavg['year']==y]['out-degree'],1)), np.array(np.round(data_cavg[data_cavg['year']==y]['in-degree'],1))
    rfl_NS, rbl_NS, rout_NS, rin_NS = np.array(np.round(data_s1avg[data_s1avg['year']==y]['forward linkage'],2)), np.array(np.round(data_s1avg[data_s1avg['year']==y]['backward linkage'],2)), np.array(np.round(data_s1avg[data_s1avg['year']==y]['out-degree'],1)), np.array(np.round(data_s1avg[data_s1avg['year']==y]['in-degree'],1))
    for fuel in ['coal','gas','oil','all three fossil fuels']:
        rvuln_NC, rvuln_NS = np.array(np.round(data_cavg[data_cavg['year']==y][fuel],2)), np.array(np.round(data_s1avg[data_s1avg['year']==y][fuel],2))
        networktext_NC[y][fuel], networktext_NS[y][fuel] = [], []
        for c in range(NC):
            networktext_NC[y][fuel].append('<b>'+str(countries[c])+'</b><br>'+str(fuel)+' vulnerability = '+str(rvuln_NC[c])+'<br>forward linkage = '+str(rfl_NC[c])+'<br>backward linkage = '+str(rbl_NC[c])+'<br>out-degree = '+str(rout_NC[c])+'<br>in-degree = '+str(rin_NC[c]))
        for s in range(NS):
            networktext_NS[y][fuel].append('<b>'+str(sectors[s])+'</b><br>'+str(fuel)+' vulnerability = '+str(rvuln_NS[s])+'<br>forward linkage = '+str(rfl_NS[s])+'<br>backward linkage = '+str(rbl_NS[s])+'<br>out-degree = '+str(rout_NS[s])+'<br>in-degree = '+str(rin_NS[s]))

sector_groups = ['Agriculture', 'Extraction and mining', 'Manufacture', 'Utilities', 'Services']
sector_groupmap = {'Agriculture':[0], 'Extraction and mining':list(range(1,16)), 'Manufacture':list(range(16,48)), 'Utilities':list(range(48,64)), 'Services':list(range(64,72))}
data_y = {'standard':{int(c[0]):c[1] for c in H.groupby(['year'])[units_list+metrics_list]}, 'yearly variation':{int(c[0]):c[1] for c in Hv.groupby(['year'])[units_list+metrics_list]}}
height, width = {'country':17, 'sector':16.5, 'region':16.5}, {'country':27*25, 'sector':27*30, 'region':27*75}

app.layout = html.Div(children=[
    dcc.Markdown('''# visualizing vulnerability v2.0''', style={'textAlign':'center'}),
    html.Div('11/08/2023 (13:30) update: addition of \'waves\' figure'),
    html.Br(),
    html.Div('11/08/2023 (15:30) update: new colormap and corrected forward linkages'),
    html.Br(),
    html.Div('18/08/2023 (07:00) update: addition of \'network-representation\' figure'),
    html.Div('Comment: I have no idea why does a diagonal appear in the upper-right corner,'),
    html.Div('in the meantime, please select the window you want to observe and discard it.'),
    html.Br(),
    html.Div('18/08/2023 (07:30) update: new year-selection slider for \'bubble plots\''),
    html.Br(),
    html.Div('22/08/2023 (11:30) update: addition of \'decomposed rows\' figure'),
    html.Div('Comment from 23/08/2023 (11:00): the \'decomposed rows\' is not functional online yet... but I promise, it does work on my laptop.'),
    html.Div('I believe this is due to the weak CPU available for free online deployments. That figure, unlike previous ones, requires extensive computations'),
    html.Div('(relative to the chosen year range) which take, on my laptop 3s and apparently >30s online which causes it to crash in the latter case,'),
    html.Div('I will explore two solutions: (1) making these computations a lot faster (if doable), (2) generate all the possible combinations on my laptop'),
    html.Div('and plugging them directly into the app (if importing such a huge new dataset doesn\'t also slow things down).'),
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
    				html.Td(dcc.Dropdown(structural_list, 'out-degree', id='metric1s'))]),
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
    					html.Td(dcc.Dropdown(monetary_list, 'forward linkage', id='metric1m'))]),
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
    			dbc.Col([html.Br(), html.Br(),
    				daq.Slider(min=1995, max=2020, step=1, value=1998, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4'))]
    		, title='Bubble plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
			dbc.Col(dcc.Dropdown(countries_list, 'Germany', id='unit5'), width=2)]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Waves'),
    	
    	dbc.AccordionItem([
    		dbc.Tabs([dbc.Tab([
    			dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Dropdown(list(range(1995,2021)), 2000, id='year6_c'), width=1)]),
    			dbc.Row([dbc.Col(dcc.Markdown('**select vulnerability (color)**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Dropdown(vulnerability_list, 'gas', id='metric6_c'), width=2)]),
    			dbc.Row([dbc.Col(dcc.Markdown('**select linkage (size)**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.RadioItems(monetary_list, 'forward linkage', inline=False, inputStyle={'margin-right':'10px', 'margin-bottom':'10px'}, id='mode6_c'), width=2)]),
    			dbc.Row([dbc.Col(dcc.Markdown('''**select binarization threshold
    					quantile (connectedness)**''', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Input(debounce=True, min=.8, max=.99, value=.9, step=.01, type='number', id='quantile6_c'))]),
    			dbc.Row([dbc.Col(dcc.Markdown('**log2-scaled vulnerability**', style={'textAlign':'right'}), width=3),
    				dbc.Col(daq.BooleanSwitch(on=True, color='#3e7cc8', id='scale6_c'), width=1)
    				]),
    				
    			dbc.Row(dcc.Graph(figure={}, id='network6_c'))]
    			, label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab([
    			dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Dropdown(list(range(1995,2021)), 2000, id='year6_s'), width=1)]),
    			dbc.Row([dbc.Col(dcc.Markdown('**select vulnerability (color)**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Dropdown(vulnerability_list, 'gas', id='metric6_s'), width=2)]),
    			dbc.Row([dbc.Col(dcc.Markdown('**select linkage (size)**', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.RadioItems(monetary_list, 'forward linkage', inline=False, inputStyle={'margin-right':'10px', 'margin-bottom':'10px'}, id='mode6_s'), width=2)]),
    			dbc.Row([dbc.Col(dcc.Markdown('''**select binarization threshold
    					quantile (connectedness)**''', style={'textAlign':'right'}), width=3),
    				dbc.Col(dcc.Input(debounce=True, min=.8, max=.99, value=.9, step=.01, type='number', id='quantile6_s'))]),
    			dbc.Row([dbc.Col(dcc.Markdown('**log2-scaled vulnerability**', style={'textAlign':'right'}), width=3),
    				dbc.Col(daq.BooleanSwitch(on=True, color='#3e7cc8', id='scale6_s'), width=1)
    				]),
    			dbc.Row(dcc.Graph(figure={}, id='network6_s'))]
    			, label='Sectors', activeTabClassName='fw-bold')
    				])
    			]
    		, title='Network representation'),
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select vulnerability**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Dropdown(vulnerability_list, 'gas', id='metric7_c'), width=2)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year range**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.RangeSlider(1995, 2020, 1, None, [1995,2020], tooltip={"placement": "bottom", "always_visible": True}, verticalHeight=100, id='year7_c'), width=5)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select number of limit cases**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Input(debounce=True, min=1, max=10, value=5, step=1, type='number', id='nb_units7_c'))]),
    		dbc.Row(dcc.Graph(figure={}, id='rows7_c'))]
    		, title='Decomposed rows'),
    			
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
    return px.line(H[H['sector']==unit3_c2], x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2+' vulnerability (%)'*int(metric3_c2 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='country').update_xaxes(tickvals= np.arange(1995,2023,3) ).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('lines3_s1', 'figure'), Input('metric3_s1', 'value'))
def update_s1line(metric3_s1):
    return px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1+' vulnerability (%)'*int(metric3_s1 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='sector').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('lines3_s2', 'figure'), [Input(s, 'value') for s in ['metric3_s2','unit3_s2']])
def update_s2line(metric3_s2,unit3_s2):
    return px.line(H[H['country']==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2+' vulnerability (%)'*int(metric3_s2 in vulnerability_list), 'color':'country'}, markers=True, hover_name='sector').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font= {'size':15}, height=700, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})

@callback(Output('bubble4', 'figure'), [Input(s, 'value') for s in ['metric4t','metric4v', 'unit4','year4']])
def update_bubble(metric4t,metric4v,unit4,year4):
    if metric4t == 'Monetary importance':
        xlabel, ylabel = 'backward linkage', 'forward linkage'
    elif metric4t == 'Structural importance':
        xlabel, ylabel = 'out-degree', 'betweenness'
    country_dataset, zlabel = H[H['country']==unit4], metric4v
    xmin, xmax, ymin, ymax, zmin, zmax = country_dataset[xlabel].min(), country_dataset[xlabel].max(), country_dataset[ylabel].min(), country_dataset[ylabel].max(), country_dataset[zlabel].min(), country_dataset[zlabel].max()
    dataset = country_dataset[country_dataset['year']==year4]
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='transition', opacity=.8, color_continuous_scale='Jet').update_layout(width=1200, height=800, font={'size':20}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).update_traces(marker_line_width=dataset['abate']*3, marker_line_color='red')
    for i,j in enumerate(legend_names):
        fig.data[i].name=legend_names[j]
    if metric4t == 'Monetary importance':
        fig.add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    return fig

@callback(Output('waves5', 'figure'), Input('unit5', 'value'))
def update_waves(unit5):
    fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=1000, height=600)
    return fig

@callback(Output('network6_c', 'figure'), [Input(s, 'value') for s in ['year6_c', 'metric6_c', 'mode6_c', 'quantile6_c']]+[Input('scale6_c','on')])
def update_cnetwork(year6, metric6, mode6, quantile6, scale6):
    global ZCs, networktext_NC
    dataset = data_cavg[data_cavg['year']==year6]
    linkage, vulnerability = np.array(dataset[mode6]), np.array(dataset[metric6])
    tickvals = [2**i for i in range(-3,round(np.log2(vulnerability.max())))]
    
    Q = np.quantile(ZCs[year6-1995], quantile6)
    Z = np.where(ZCs[year6-1995]>Q,1,0)
    G = nx.from_numpy_matrix(Z)
    zero_deg = [i for i in G if G.degree(i)==0]
    nonzero_deg = [i for i in G if G.degree(i)]
    G.remove_nodes_from(zero_deg)
    G = nx.relabel_nodes(G, {i:countries[i] for i in range(NC)})
    pos = nx.kamada_kawai_layout(G)
    
    edge_x, edge_y = list(chain.from_iterable([(pos[i][0],pos[j][0],None) for i,j in G.edges()])), list(chain.from_iterable([(pos[i][1],pos[j][1],None) for i,j in G.edges()]))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=.5, color='#888'), mode='lines')
    node_x, node_y = [pos[i][0] for i in G], [pos[i][1] for i in G]
    if scale6:
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(showscale=True, colorscale='Jet', color=np.log2(vulnerability), size=20*linkage**2, colorbar=dict(thickness=15, xanchor='left', titleside='right', tickvals=np.log2(tickvals), ticktext=tickvals), line_width=2), text=networktext_NC[year6][metric6], hovertemplate='<extra></extra>%{text}', hoverlabel={'font_size':20}, line={'width':0.5, 'color':'black'})
    else:
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(showscale=True, colorscale='Jet', color=vulnerability, size=20*linkage**2, colorbar=dict(thickness=15, xanchor='left', titleside='right'), line_width=2), text=networktext_NC[year6][metric6], hovertemplate='<extra></extra>%{text}', hoverlabel={'font_size':20}, line={'width':0.5, 'color':'black'})
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(titlefont_size=16, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), annotations=[dict(showarrow=True)])).update_layout(width=1000, height=800)
    return fig

@callback(Output('network6_s', 'figure'), [Input(s, 'value') for s in ['year6_s', 'metric6_s', 'mode6_s', 'quantile6_s']]+[Input('scale6_s','on')])
def update_snetwork(year6, metric6, mode6, quantile6, scale6):
    global ZSs, networktext_NS
    dataset = data_s1avg[data_s1avg['year']==year6]
    linkage, vulnerability = np.array(dataset[mode6]), np.array(dataset[metric6])
    tickvals = [2**i for i in range(-3,round(np.log2(vulnerability.max())))]
    Q = np.quantile(ZSs[year6-1995],quantile6)
    Z = np.where(ZSs[year6-1995]>Q,1,0)
    G = nx.from_numpy_matrix(Z)
    zero_deg = [i for i in G if G.degree(i)==0]
    nonzero_deg = [i for i in G if G.degree(i)]
    G.remove_nodes_from(zero_deg)
    G = nx.relabel_nodes(G, {i:sectors[i] for i in range(NS)})
    pos = nx.kamada_kawai_layout(G)
    edge_x, edge_y = list(chain.from_iterable([(pos[i][0],pos[j][0],None) for i,j in G.edges()])), list(chain.from_iterable([(pos[i][1],pos[j][1],None) for i,j in G.edges()]))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=.5, color='#888'), mode='lines')
    node_x, node_y = [pos[i][0] for i in G], [pos[i][1] for i in G]
    if scale6:
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(showscale=True, colorscale='Jet', color=np.log2(vulnerability+.01), size=20*linkage**2, colorbar=dict(thickness=15, xanchor='left', titleside='right', tickvals=np.log2(tickvals), ticktext=tickvals), line_width=2), text=networktext_NS[year6][metric6], hovertemplate='<extra></extra>%{text}', hoverlabel={'font_size':20}, line={'width':0.5, 'color':'black'})
    else:
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(showscale=True, colorscale='Jet', color=vulnerability, size=20*linkage**2, colorbar=dict(thickness=15, xanchor='left', titleside='right'), line_width=2), text=networktext_NS[year6][metric6], hovertemplate='<extra></extra>%{text}', hoverlabel={'font_size':20}, line={'width':0.5, 'color':'black'})
    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(titlefont_size=16, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), annotations=[dict(showarrow=True)])).update_layout(width=1000, height=800)

@app.callback(Output('rows7_c', 'figure'), [Input(s, 'value') for s in ['metric7_c', 'year7_c', 'nb_units7_c']])
def update_crows(metric7, year7, nb_units7):
    year7i, year7f = year7
    dataset = (data_cavg[data_cavg['year']==year7f][metric7].reset_index(drop=True).subtract(data_cavg[data_cavg['year']==year7i][metric7].reset_index(drop=True))).sort_values()
    pos_units, neg_units = countries[dataset[-nb_units7:].index], countries[dataset[:nb_units7].index]
    pos_data, neg_data = np.round(all_pos_data_c(pos_units,year7i,year7f,metric7),2), np.round(all_neg_data_c(neg_units,year7i,year7f,metric7),2)
    return px.bar(neg_data._append(pos_data), y='country', x='length', color='sector', orientation='h', custom_data=['country','sector','length'], text_auto='.2s', height=(nb_units7+1.5)*100).update_layout(font_size=14, hoverlabel={'font_size':22}).update_traces(hovertemplate='<extra></extra>%{customdata[0]}: <b>%{customdata[1]},</b><br>contribution to overall<br><b>'+metric7+'</b> vulnerability: %{customdata[2]}')

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

def all_pos_data_c(pos_units,y1_,y2_,fuel_):
    all_pos_units = pd.DataFrame()
    for c in pos_units:
        data_c = deepcopy(H[(H['country']==c)&(H['year']==y2_)][['country','sector',fuel_]].reset_index(drop=True))
        data_c[fuel_] -= H[(H['country']==c)&(H['year']==y1_)][fuel_].reset_index(drop=True)
        avg, std = data_c[fuel_].mean(), data_c[fuel_].std()
        pos_subunits = data_c[data_c[fuel_]>avg+std].reset_index(drop=True).sort_values(fuel_, ascending=False)
        leftover_vuln = NS*avg-pos_subunits[fuel_].sum()
        pos_subunits.loc[len(pos_subunits)] = [c,'leftover',leftover_vuln]
        total_vuln = sum(abs(pos_subunits[fuel_]))
        if pos_subunits.iloc[-1,-1]<0:
            total_vuln_pos = sum([i for i in pos_subunits[fuel_] if i>0])
            pos_subunits['length_ratio'] = [i/total_vuln_pos for i in pos_subunits[fuel_][:-1]] + [total_vuln_pos/total_vuln -1]
        else:
            pos_subunits['length_ratio'] = [i/total_vuln for i in pos_subunits[fuel_]]
        pos_subunits['length'] = pos_subunits['length_ratio']*avg
        all_pos_units = all_pos_units._append(pos_subunits, ignore_index=True)
    return all_pos_units

def all_neg_data_c(neg_units,y1_,y2_,fuel_):
    all_neg_units = pd.DataFrame()
    for c in neg_units:
        data_c = deepcopy(H[(H['country']==c)&(H['year']==y2_)][['country','sector',fuel_]].reset_index(drop=True))
        data_c[fuel_] -= H[(H['country']==c)&(H['year']==y1_)][fuel_].reset_index(drop=True)
        avg, std = data_c[fuel_].mean(), data_c[fuel_].std()
        neg_subunits = data_c[data_c[fuel_]<avg-std].reset_index(drop=True).sort_values(fuel_)
        leftover_vuln = NS*avg-neg_subunits[fuel_].sum()
        neg_subunits.loc[len(neg_subunits)] = [c,'Others',leftover_vuln]
        total_vuln = -sum(abs(neg_subunits[fuel_]))
        if neg_subunits.iloc[-1,-1]>0:
            total_vuln_neg = sum([i for i in neg_subunits[fuel_] if i<0])
            neg_subunits['length_ratio'] = [i/total_vuln_neg for i in neg_subunits[fuel_][:-1]] + [total_vuln_neg/total_vuln -1]
        else:
            neg_subunits['length_ratio'] = [i/total_vuln for i in neg_subunits[fuel_]]
        neg_subunits['length'] = neg_subunits['length_ratio']*avg
        all_neg_units = all_neg_units._append(neg_subunits, ignore_index=True)
    return all_neg_units


if __name__ == '__main__':
    app.run_server(debug=True)
