from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
from random import shuffle
pd.options.mode.chained_assignment = None
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

H = pd.read_csv('processed_data_update.csv', compression='gzip')
worldmap_nodes, worldmap_links = pd.read_csv('worldmap_nodes.csv'), pd.read_csv('worldmap_links_plot.csv')
NS, NC, NY = 163, 49, 26
N, Ntot = NS*NC, NS*NC*NY
regions, countries, sectors = H.region.iloc[np.arange(0,N,NS)], H.country.unique(), H.sector.unique()
regions_list, countries_list, sectors_list, sectors_decarb = list(regions.unique()), list(countries), list(sectors), H[H.sector_color!='grey'].sector.unique()
region_to_country, countries_code = {txt: list(np.where(np.array(regions)==txt)[0]) for txt in regions_list}, [worldmap_nodes.CODE[np.where(worldmap_nodes.EXIOBASE_name==c)[0][0]] for c in countries]
names7, colors7, markers4 = ['Plastics', 'Mining', 'Machinery', 'Nonmetallic products', 'Metallic products', 'Construction', 'Other sectors'], ['pink', 'gold', 'red', 'green', 'blue', 'cyan', 'grey'], H.region_marker.unique()
color_to_sector = {colors7[i]+', '+markers4[j]: names7[i]+', '+regions_list[j] for i in range(7) for j in range(4)}

sector_to_idx, country_to_idx, k1, k2 = {}, {}, 0, 0
for s in sectors:
    sector_to_idx[s], k1 = k1, k1+1

for c in countries:
    country_to_idx[c], k2 = k2, k2+1

units_list = ['region', 'country', 'sector']
vulnerability_list, structural_list, monetary_list = ['coal', 'gas', 'oil', 'vulnerability_index'], ['out_degree', 'in_degree', 'total_degree', 'betweenness', 'weighted_in_degree', 'weighted_out_degree', 'weighted_total_degree', 'structural_index'], ['forward_linkage', 'backward_linkage', 'monetary_index']

metrics_list = vulnerability_list+structural_list+monetary_list
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list] = H[metrics_list].astype(float)
#H[['region_marker', 'sector_color']] = H[['region_marker', 'sector_color']].astype("string")
for s in ['year']+units_list:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)

# leaving only exogeneous vulnerability
for i,j in [('coal','Mining of coal, lignite and peat'), ('oil','Extraction of crude petroleum'), ('gas','Extraction of natural gas')]:
    row, col = H[H.sector==j].index, np.where(H.columns==i)[0][0]
    H.iloc[row,col] = 0

data_cavg = H.groupby([('year'),('country')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('country')])
data_s1avg = H.groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')])
data_s2avg = {c[0]:c[1].groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')]) for c in H.groupby('region', observed=False)}

year_NYlist, sector_NYlist = np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil','gas','coal'] for i in range(NY)])
data_cvuln = {}
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['country']==c][['oil','gas','coal']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

sector_groups, sector_group_colors = ['Agriculture, forestry and fishing', 'Extraction and mining', 'Manufacture and production', 'Utilities', 'Services'], ['lime', 'orange', 'red', 'cyan', 'purple']
sector_groupmap = {'Agriculture, forestry and fishing':list(range(19)), 'Extraction and mining':list(range(19,34)), 'Manufacture and production':list(range(34,93))+[109,113], 'Utilities':list(range(93,119))+[110,111,112]+list(range(114,120)), 'Services':list(range(120,163))}
country_groups = {r:list(np.where(regions==r)[0]) for r in regions_list}
decarb_groups = {names7[i]:list(np.where(H.sector_color[:NS]==colors7[i])[0]) for i in range(7)}
data_y = {int(c[0]):c[1] for c in H.groupby('year', observed=False)[units_list+metrics_list]}

app.layout = html.Div(children=[
    dcc.Markdown('''# visualizing vulnerability v2.1''', style={'textAlign':'center'}),
    html.Hr(),
    html.Div('20/03/2024 (18:00) update: Successful deployment of the post-Durham UI version.'),
    html.Div('16/04/2024 (12:30) update: Heatmaps\' cell sizes now (almost perfectly) adjust to selecting/unselecting sectors/countries + multiple minor dataset and UI layout fixes'),
    html.Div('16/04/2024 (20:30) update: Added 2 types of bubbles figures: (i) Worldwide decarbonization sectors. (ii) All sectors of a single country.'),
    html.Div('24/04/2024 (14:30) update: Worldmap figure, added options on bubble plots.'),
    html.Hr(),
    html.Div('Incoming minor updates: more consistent layout across figure types, easier-to-read hover data, fixing sector-color associations in bubble1 & region-marker associations in bubble2, explaining worldmap figure (sector groups and RoW regions).'),
    #html.Div('Future major updates: rows figure (Fig1c in the manuscript), if I figure out a lightened version of it.'),
    dbc.Accordion([
    	dbc.AccordionItem(
    	dbc.Tabs([
    		dbc.Tab([html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(# column of parameter selection
    		dbc.Table(html.Tbody([
    			html.Tr([html.Td(dcc.Markdown('**select energy**', style={'textAlign':'right'}), ),
    				html.Td(dcc.Dropdown(vulnerability_list, 'coal', id='metric1v'), )]),
    			html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'}), ),
     				html.Td(dcc.Dropdown(units_list, 'country', id='unit1v'), )]),
     			html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'}), ),
    				html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1v'))]), ]), borderless=True), ),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=3),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1v', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=4)]),
    			]),
    		dcc.Graph(figure={}, id='hist1v'), ], label='Vulnerability', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(# column of parameter selection
    		dbc.Table(html.Tbody([
    			html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'}), ),
    				html.Td(dcc.Dropdown(structural_list, 'out_degree', id='metric1s'), )]),
    			html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'}), ),
     				html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1s'), )]),
     			html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'}), ),
    				html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1s'))]), ]), borderless=True), ),
    		dbc.Col(#column of ordering selection
    		[dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=3),
    		dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1s', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=4)]),]),
    		dcc.Graph(figure={}, id='hist1s'), ], label='Structural importance', activeTabClassName='fw-bold'),
    		
    		dbc.Tab([html.Br(),
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col(dbc.Table(html.Tbody([
    				html.Tr([html.Td(dcc.Markdown('**select metric**', style={'textAlign':'right'}), ),
    					html.Td(dcc.Dropdown(monetary_list, 'forward_linkage', id='metric1m'), )]),
    				html.Tr([html.Td(dcc.Markdown('**select unit**', style={'textAlign':'right'}), ),
     					html.Td(dcc.Dropdown([units_list[2]], 'sector', id='unit1m'), )]),
     				html.Tr([html.Td(dcc.Markdown('**select year**', style={'textAlign':'right'}), ),
    					html.Td(dcc.Dropdown(list(range(1995,2021)), 2000, id='year1m'))]), ]), borderless=True), ),
		dbc.Col([dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'center'}), width=3),
    			dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1m', inputStyle={'margin-right':'10px', 'margin-left':'30px'}), width=4)]),]),
    		dcc.Graph(figure={}, id='hist1m'), ], label='Monetary importance', activeTabClassName='fw-bold'),
    		]), title='Yearly histogram averages'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([html.Br(),
    				dbc.Offcanvas([
    				dcc.Checklist(countries, countries, id='unit2_c', inputStyle={'margin-right':'10px'})], id='canvas2_c', is_open=False, placement='end', scrollable=True),
    				
    				dbc.Col(dbc.Button('Select countries', id='open_canvas2_c', n_clicks=0), width=2),
    				html.Br(),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_c'), width=2),]),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=4)]),
    				dbc.Row([
    				    dbc.Col(dcc.Markdown('**select regions**', style={'textAlign':'right'}), width=2),
    				    dbc.Col(dcc.Checklist(regions_list, regions_list, id='group2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    				dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab(dbc.Tabs([dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s1', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					html.Br(),
    					dcc.Checklist(sectors, sectors, id='unit2_s1', inputStyle={'margin-right':'10px'})], id='canvas2_s1', is_open=False, placement='end', scrollable=True),
    					dbc.Col(dbc.Button('Select sectors', id='open_canvas2_s1', n_clicks=0)),
    					html.Br(),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_s1'), width=2), ]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s1')),
    					], label='Worldwide averages', activeTabClassName='fw-bold'),
    				    
    					dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s2', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					html.Br(),
    					dcc.Checklist(sectors, sectors, id='unit2_s2', inputStyle={'margin-right':'10px'})], id='canvas2_s2', is_open=False, placement='end', scrollable=True),
    					dbc.Col(dbc.Button('Select sectors', id='open_canvas2_s2', n_clicks=0)),
    					html.Br(),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric2_s2'), width=2),]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s2')),
    				], label='Regional averages', activeTabClassName='fw-bold'),])
    			, label='Sectors', activeTabClassName='fw-bold')])
    			, title='Heatmaps'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab(dbc.Tabs([
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal', id='metric3_c1'), width=2), ]),
    				dbc.Row([dbc.Col(dcc.Markdown('**select/unselect region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_c1'))], label='worldwide', activeTabClassName='fw-bold'),
    			
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'oil', id='metric3_c2'), width=2)],),
    				dbc.Row([dbc.Col(dcc.Markdown('**select sector**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(sectors, 'Plastics', id='unit3_c2'), width=3)]),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select/unselect region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
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
    					dbc.Col(dcc.Dropdown(countries, 'USA', id='unit3_s2'), width=3)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_s2'))], label='Single country', activeTabClassName='fw-bold')]
    			), label='Sectors', activeTabClassName='fw-bold')])
    		, title='Line charts'),
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    dbc.Tab([
    	    html.Br(),
    	    dbc.Offcanvas(dcc.Checklist(countries, countries, id='unit4_s1c', inputStyle={'margin-right':'10px'}), id='canvas4_s1c', is_open=False, placement='end', scrollable=True),
    	    dbc.Row([dbc.Col(dbc.Button('Select countries', id='open_canvas4_s1c', n_clicks=0), width=1.5)]),
    	    html.Br(),
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'forward_linkage', id='metric4x_s1'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'backward_linkage', id='metric4y_s1'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(vulnerability_list+ ['structural_index', 'monetary_index'], 'gas', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4i_s1'))]),
			dbc.Row([dbc.Col(dcc.Markdown('**select regions to display (marker style)**', style={'textAlign':'right'}), width=2),
    		    dbc.Col(dcc.Checklist(regions_list, ['Europe'], id='group4_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    		   
    		dbc.Row([dbc.Col(dcc.Markdown('**select sector groups to display**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Checklist(names7, names7[:-1], id='group4_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    		
    		dbc.Row([dbc.Col(dcc.Markdown('**select color according to**', style={'textAlign':'right'}), width=2),
    		    dbc.Col([
    		        dbc.Row([
    		        dcc.RadioItems(['sector group', 'vulnerability'], 'sector group', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px', 'margin-bottom':'5px'}, id='color4'),
    		        dbc.Row([dcc.Markdown('if vulnerability, select which fossil fuel:', style={'textAlign':'left'}),
    		                dcc.RadioItems(vulnerability_list, 'oil', inline=True, inputStyle={'margin-top':'0px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4v_s1'), ])
    		        
    		            ])], width=8),]),
    		    
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2020, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s1')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s1'))], label='Worldwide (decarbonization-related sectors)', activeTabClassName='fw-bold'),
    	    
    	    dbc.Tab([html.Br(),
    	    dbc.Offcanvas([dcc.Checklist(sector_groups, sector_groups, id='group4_s2', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    	    html.Br(),
    		dcc.Checklist(sectors, sectors, id='unit4_s2', inputStyle={'margin-right':'10px'})], id='canvas4_s2', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Select sectors', id='open_canvas4_s2', n_clicks=0), width=2),
    		html.Br(),
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'forward_linkage', id='metric4x_s2'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+monetary_list, 'backward_linkage', id='metric4y_s2'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select country (unit)**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(countries, 'USA', id='unit4_s2c'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(['structural_index', 'vulnerability_index', 'monetary_index'], 'structural_index', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4i_s2'))]),
		    dbc.Row([dbc.Col(dcc.Markdown('**select color**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.RadioItems(vulnerability_list, 'coal', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4v_s2'), width=4)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=1998, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s2')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s2'))], label='Single country (all sectors)', activeTabClassName='fw-bold')    		
    		]), title='Bubble plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(countries_list, 'Brazil', id='unit5'), width=3)]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Waves'),
    	
    	dbc.AccordionItem([
    		# insert number of edges (max, per sector group)
    		# add details on what are Rest of World regions
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			    dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2020, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year6')])]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select energy**', style={'textAlign':'right'}), width=2),
    		        dbc.Col(dcc.RadioItems(vulnerability_list, 'oil', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric6'), width=4),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select number of links per sector group**', style={'textAlign':'right'}), width=2),
    			    dbc.Col([html.Br(), html.Br(), daq.Slider(min=10, max=100, step=10, value=50, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='nbedges6')])]),
                		
    		dbc.Row(dcc.Graph(figure={}, id='map6'))]
    		, title='Worldmap'),
    	
			], flush=True)
			])


@callback(Output('hist1v', 'figure'), [Input(s, 'value') for s in ['metric1v','unit1v','year1v','order1v']])
def update_vhistogram(metric1v,unit1v,year1v,order1v):#reinclude type1v, change data_y[year1v] by data_y[type1v][year1v] after reincluding yearly variations
    order = (order1v=='original')*'trace' + (order1v=='descending')*'total descending' + (order1v=='ascending')*'total ascending'
    return px.histogram(data_y[year1v].reset_index(), x=unit1v, y=metric1v, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1v=='sector')*45+(unit1v=='country')*30).update_layout(yaxis_title=metric1v+' vulnerability', font={'size':18}, height=500).update_traces(hovertemplate='%{x}, '+str(year1v)+'<br>%{y}')

@callback(Output('hist1s', 'figure'), [Input(s, 'value') for s in ['metric1s','unit1s','year1s','order1s']])
def update_shistogram(metric1s,unit1s,year1s,order1s):
    order = (order1s=='original')*'trace' + (order1s=='descending')*'total descending' + (order1s=='ascending')*'total ascending'
    return px.histogram(data_y[year1s].reset_index(), x=unit1s, y=metric1s, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1s=='sector')*45+(unit1s=='country')*30).update_layout(yaxis_title=metric1s, font={'size':18}, height=600).update_traces(hovertemplate='%{x}, '+str(year1s)+'<br>%{y}')

@callback(Output('hist1m', 'figure'), [Input(s, 'value') for s in ['metric1m','unit1m','year1m','order1m']])
def update_mhistogram(metric1m,unit1m,year1m,order1m):
    order = (order1m=='original')*'trace' + (order1m=='descending')*'total descending' + (order1m=='ascending')*'total ascending'
    return px.histogram(data_y[year1m].reset_index(), x=unit1m, y=metric1m, histfunc='avg').update_xaxes(categoryorder=order, autorange='reversed', tickangle=(unit1m=='sector')*45+(unit1m=='country')*30).update_layout(yaxis_title=metric1m+' vulnerability', font={'size':18}, height=600).update_traces(hovertemplate='%{x}, '+str(year1m)+'<br>%{y}')

@callback(Output('heatmap2_c', 'figure'), [Input(s, 'value') for s in ['metric2_c','group2_c','order2_c','unit2_c']])
def update_cheatmap(metric2_c,group2_c,order2_c,unit2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    gcinds = sum([country_groups[c] for c in group2_c],[])
    cinds = list(set([country_to_idx[c] for c in unit2_c]) & set(gcinds))
    data_c = deepcopy(data_cavg.iloc[sorted([k*NC+i for k in range(NY) for i in cinds])])
    return px.density_heatmap(data_c, x='year', y='country', z=metric2_c, height=70*len(cinds)**.63, width=450+260, labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=len(unit2_c), color_continuous_scale='Turbo').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'})

@callback(Output('heatmap2_s1', 'figure'), [Input(s, 'value') for s in ['metric2_s1','group2_s1', 'order2_s1','unit2_s1']])
def update_s1heatmap(metric2_s1,group2_s1,order2_s1,unit2_s1):
    order = (order2_s1=='original')*'trace' + (order2_s1=='descending')*'total descending' + (order2_s1=='ascending')*'total ascending'
    gs1inds = sum([sector_groupmap[g] for g in group2_s1],[])
    s1inds = list(set([sector_to_idx[s] for s in unit2_s1]) & set(gs1inds))
    data_s1 = deepcopy(data_s1avg.iloc[sorted([k*NS+i for k in range(NY) for i in s1inds])])
    return px.density_heatmap(data_s1, x='year', y='sector', z=metric2_s1, height=45*len(s1inds)**.78, width=680+260, labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=len(s1inds), color_continuous_scale='Jet').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'})

@callback(Output('heatmap2_s2', 'figure'), [Input(s, 'value') for s in ['metric2_s2','group2_s2','order2_s2','unit2_s2']])
def update_s2heatmap(metric2_s2,group2_s2,order2_s2,unit2_s2):
    order = (order2_s2=='original')*'trace' + (order2_s2=='descending')*'total descending' + (order2_s2=='ascending')*'total ascending'
    gs2inds = sum([sector_groupmap[g] for g in group2_s2],[])
    s2inds = list(set([sector_to_idx[s] for s in unit2_s2])&set(gs2inds))
    data_s2 = {r:deepcopy(data_s2avg[r].iloc[sorted([k*NS+i for k in range(NY) for i in s2inds])]) for r in regions_list}
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=.005, shared_yaxes=True, subplot_titles=['<b>'+str(i)+'</b>' for i in regions_list])
    for r_idx in range(len(regions_list)):
        r = regions_list[r_idx]
        fig.add_trace(px.density_heatmap(data_s2[r], x='year', y='sector', z=metric2_s2, nbinsx=NY, nbinsy=len(s2inds)).data[0], row=1, col=r_idx+1)
    fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(height=15+45*len(s2inds)**.78, width=720+4*260, font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, coloraxis={'colorscale':'Jet'})
    return fig

@callback(Output('lines3_c1', 'figure'), [Input(s, 'value') for s in ['metric3_c1','region3_c1']])
def update_c1line(metric3_c1,region3_c1):
    inds = [i for r in region3_c1 for i in region_to_country[r]]
    c1_indices = sorted([k for j in inds for k in np.arange(j,NY*NC,NC)])
    data3_slice = deepcopy(data_cavg.iloc[c1_indices])
    data3_slice['country'] = data3_slice['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    try:
        fig = px.line(data3_slice, x='year', y=metric3_c1, color='country', labels={'x':'year', 'y':metric3_c1+' vulnerability (%)'*int(metric3_c1 in vulnerability_list), 'color':'country'}, markers=True, hover_name='country').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    except:
        fig = px.line(data3_slice, x='year', y=metric3_c1, color='country', labels={'x':'year', 'y':metric3_c1+' vulnerability (%)'*int(metric3_c1 in vulnerability_list), 'color':'country'}, markers=True, hover_name='country').update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    return fig

@callback(Output('lines3_c2', 'figure'), [Input(s, 'value') for s in ['metric3_c2','unit3_c2','region3_c2']])
def update_c2line(metric3_c2,unit3_c2,region3_c2):
    try:
        fig = px.line(H[H['sector']==unit3_c2], x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2+' vulnerability (%)'*int(metric3_c2 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='country')
    except:
        fig = px.line(H[H['sector']==unit3_c2], x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2+' vulnerability (%)'*int(metric3_c2 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='country')
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    return fig

@callback(Output('lines3_s1', 'figure'), Input('metric3_s1', 'value'))
def update_s1line(metric3_s1):
    try:
        fig = px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1+' vulnerability (%)'*int(metric3_s1 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='sector')
    except:
        fig = px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1+' vulnerability (%)'*int(metric3_s1 in vulnerability_list), 'color':'sector'}, markers=True, hover_name='sector')
    fig.update_xaxes(tickvals= np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    return fig

@callback(Output('lines3_s2', 'figure'), [Input(s, 'value') for s in ['metric3_s2','unit3_s2']])
def update_s2line(metric3_s2,unit3_s2):
    try:
        fig = px.line(H[H.country==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2+' vulnerability (%)'*int(metric3_s2 in vulnerability_list), 'color':'country'}, markers=True, hover_name='sector')
    except:
        fig = px.line(H[H.country==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2+' vulnerability (%)'*int(metric3_s2 in vulnerability_list), 'color':'country'}, markers=True, hover_name='sector')
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_yaxes(tickmode= 'linear').update_layout(font= {'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10})
    return fig

@callback(Output('bubble4_s1', 'figure'), [Input(s, 'value') for s in ['metric4x_s1','metric4y_s1','metric4i_s1', 'group4_c','unit4_s1c', 'group4_s','year4_s1', 'color4', 'metric4v_s1']])
def update_s1bubble(metric4x_s1,metric4y_s1,metric4i_s1,group4_c,unit4_s1c,group4_s,year4_s1,color4,metric4v_s1):
    xlabel, ylabel, zlabel = metric4x_s1, metric4y_s1, metric4i_s1
    gcinds, sinds = sum([country_groups[c] for c in group4_c],[]), sum([decarb_groups[c] for c in group4_s],[])
    cinds = list(set([country_to_idx[c] for c in unit4_s1c]) & set(gcinds))
    dataset = deepcopy(H[H.year==year4_s1].reset_index().iloc[sorted([c*NS+s for c in cinds for s in sinds]),1:])
    xmin, xmax, ymin, ymax, zmin, zmax = .5, 2, .5, 2, dataset[zlabel].min(), dataset[zlabel].max()
    color4 = 'sector_color'*(color4=='sector group') + metric4v_s1*(color4=='vulnerability')
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet')
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet')
    fig.update_layout(width=900, height=700, xaxis_range=[xmin,xmax], yaxis_range=[ymin,ymax], font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    if color4=='sector_color':
        fig.update_layout(legend={'title':'sector group, region'})
        fig.for_each_trace(lambda t: t.update(name=color_to_sector[t.name], legendgroup=color_to_sector[t.name], hovertemplate=t.hovertemplate.replace(t.name, color_to_sector[t.name])))
    else:
        fig.update_layout(showlegend=False)
    return fig

@callback(Output('bubble4_s2', 'figure'), [Input(s, 'value') for s in ['metric4x_s2','metric4y_s2','group4_s2', 'unit4_s2c','metric4i_s2', 'metric4v_s2', 'unit4_s2','year4_s2']])
def update_s2bubble(metric4x_s2,metric4y_s2,group4_s2,unit4_s2c,metric4i_s2,metric4v_s2,unit4_s2,year4_s2):
    xlabel, ylabel, zlabel = metric4x_s2, metric4y_s2, metric4i_s2
    gsinds = sum([sector_groupmap[g] for g in group4_s2],[])
    sinds = list(set([sector_to_idx[s] for s in unit4_s2]) & set(gsinds))
    dataset = deepcopy(H[H.year==year4_s2].reset_index().iloc[sorted([country_to_idx[unit4_s2c]*NS+s for s in sinds]),1:])
    xmin, xmax, ymin, ymax, zmin, zmax = .5, 2, .5, 2, dataset[zlabel].min(), dataset[zlabel].max()
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet')
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet')
    fig.update_layout(width=900, height=700, font={'size':16}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7)
    return fig

@callback(Output('waves5', 'figure'), Input('unit5', 'value'))
def update_waves(unit5):
    try:
        fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'vulnerability (%)'})
    except:
        fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'vulnerability (%)'})
    return fig

@callback(Output('map6', 'figure'), [Input(s, 'value') for s in ['year6', 'metric6', 'nbedges6']])
def update_maps(year6, metric6, nbedges6):
    dataset, colorbar_legend = worldmap_links[worldmap_links.year==year6].reset_index(), 'vulnerability index'*(metric6=='vulnerability_index') + (metric6+' vulnerability (%)')*(metric6!='vulnerability_index')
    fig = go.Figure(go.Choropleth(locations=countries_code, z=data_cavg[data_cavg.year==year6][metric6], colorscale='Reds', colorbar={'title': colorbar_legend}))
    for s in range(5):
        fig.add_scattergeo(lat=worldmap_nodes['lat'+str(s)], lon=worldmap_nodes['lon'+str(s)], marker={'size':7, 'color':sector_group_colors[s]}, showlegend=False)
    #fig.add_scattergeo(lat=worldmap_nodes.lat, lon=worldmap_nodes.lon, marker={'size':18, 'symbol':'circle-open', 'color':'black'})
    fig.add_scattergeo(lat=worldmap_nodes.lat, lon=worldmap_nodes.lon, marker={'size':7, 'symbol':'circle-open', 'color':'black'}, showlegend=False)
    for idx in range(nbedges6//10):
        for s in range(5):
            batch, color = deepcopy(dataset.iloc[3*(100*s+10*idx): 3*(100*s+10*(idx+1))]), sector_group_colors[s]
            fig.add_scattergeo(lat=batch.latitude, lon=batch.longitude, mode='lines', line={'width':1.5, 'color':color}, showlegend=False)
    _ = fig.update_geos(lataxis_range=[-55, 90], showocean=True, oceancolor='LightBlue').update_layout(width=1400, height=600)
    return fig

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

@app.callback(Output('canvas4_s2', 'is_open'), Input('open_canvas4_s2', 'n_clicks'), State('canvas4_s2', 'is_open'))
def toggle_4s2canvas(n, is_open):
    if n:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run_server(debug=True, port=8052)

#pip install --upgrade pip && pip install -r requirements.txt
