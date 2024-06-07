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
vulnerability_list, structural_list, economic_list = ['coal vulnerability', 'gas vulnerability', 'oil vulnerability', 'vulnerability index'], ['out-degree', 'in-degree', 'total degree', 'betweenness', 'weighted in-degree', 'weighted out-degree', 'weighted total degree', 'structural index'], ['forward linkage', 'backward linkage', 'linkage index']

metrics_list = vulnerability_list+structural_list+economic_list
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list] = H[metrics_list].astype(float)
#H[['region_marker', 'sector_color']] = H[['region_marker', 'sector_color']].astype("string")
for s in ['year']+units_list:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)

# leaving only exogeneous vulnerability
for i,j in [('coal vulnerability','Mining of coal, lignite and peat'), ('oil vulnerability','Extraction of crude petroleum'), ('gas vulnerability','Extraction of natural gas')]:
    row, col = H[H.sector==j].index, np.where(H.columns==i)[0][0]
    H.iloc[row,col] = 0

data_cavg = H.groupby([('year'),('country')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('country')])
data_s1avg = H.groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')])
data_s2avg = {c[0]:c[1].groupby([('year'),('sector')], observed=False)[metrics_list].mean().reset_index().sort_values([('year'),('sector')]) for c in H.groupby('region', observed=False)}

data_cvuln, year_NYlist, sector_NYlist = {}, np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil vulnerability','gas vulnerability','coal vulnerability'] for i in range(NY)])
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['country']==c][['oil vulnerability','gas vulnerability','coal vulnerability']].unstack().values)).T, columns=['year','sector','vulnerability'])
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
    html.Div('Next minor updates: more consistent layout across figure types, easier-to-read hover data, fixing sector-color associations in bubble1 & region-marker associations in bubble2, explaining worldmap figure (sector groups and RoW regions).'),
    dbc.Accordion([
    	dbc.AccordionItem([
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col([# column of parameter selection
    		dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Dropdown(vulnerability_list+structural_list+economic_list, 'coal vulnerability', id='metric1'), width=2), 
    		        dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order1', inline=True, inputStyle={'margin-right':'10px', 'margin-left':'20px'}), width=5), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select unit**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Dropdown(units_list, 'country', id='unit1'), width=2), 
    		        dbc.Col(dcc.Markdown('**decompose metric**', style={'textAlign':'right'}), width=2), dbc.Col(dbc.Checklist(id='type1', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}), width=4), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2), dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2000, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year1')], width=5), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**display regions**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Checklist(regions_list, regions_list, id='group1_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**display sector groups**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Checklist(sector_groups, sector_groups, id='group1_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)
    		        ])], ), ]),
    		dcc.Graph(figure={}, id='hist1')], title='Yearly histogram averages'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([html.Br(),
    				dbc.Offcanvas([
    				dcc.Checklist(countries, countries, id='unit2_c', inputStyle={'margin-right':'10px'})], id='canvas2_c', is_open=False, placement='end', scrollable=True),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_c'), width=2),]),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=4)]),
    				dbc.Row([
    				    dbc.Col(dcc.Markdown('**display regions**', style={'textAlign':'right'}), width=2),
    				    dbc.Col(dcc.Checklist(regions_list, regions_list, id='group2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    				dbc.Row([dbc.Col(width=1), dbc.Col(dbc.Button('display countries', id='open_canvas2_c', n_clicks=0), width=2),]),
    				dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab(dbc.Tabs([dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s1', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					dcc.Checklist(sectors, sectors, id='unit2_s1', inputStyle={'margin-right':'10px'})], id='canvas2_s1', is_open=False, placement='end', scrollable=True),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_s1'), width=2), ]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    					dbc.Row([dbc.Col(width=1), dbc.Col(dbc.Button('display sectors', id='open_canvas2_s1', n_clicks=0), width=2),]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s1')),
    					], label='Worldwide averages', activeTabClassName='fw-bold'),
    				    
    					dbc.Tab([html.Br(),
    					dbc.Offcanvas([
    					dcc.Checklist(sector_groups, sector_groups, id='group2_s2', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    					dcc.Checklist(sectors, sectors, id='unit2_s2', inputStyle={'margin-right':'10px'})], id='canvas2_s2', is_open=False, placement='end', scrollable=True),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_s2'), width=2),]),
    					dbc.Row([
    						dbc.Col(dcc.Markdown('**select ordering**', style={'textAlign':'right'}), width=2),
    						dbc.Col(dcc.RadioItems(ordering_list, 'original', id='order2_s2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    					dbc.Row([dbc.Col(width=1), dbc.Col(dbc.Button('display sectors', id='open_canvas2_s2', n_clicks=0), width=2),]),
    					dbc.Row(dcc.Graph(figure={}, id='heatmap2_s2')),
    				], label='Regional averages', activeTabClassName='fw-bold'),])
    			, label='Sectors', activeTabClassName='fw-bold')])
    			, title='Heatmaps'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab(dbc.Tabs([
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric3_c1'), width=2), ]),
    				dbc.Row([dbc.Col(dcc.Markdown('**display region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c1', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_c1'))], label='worldwide', activeTabClassName='fw-bold'),
    			
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'oil vulnerability', id='metric3_c2'), width=2)],),
    				dbc.Row([dbc.Col(dcc.Markdown('**select sector**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(sectors, 'Plastics', id='unit3_c2'), width=5)]),
	    			dbc.Row([dbc.Col(dcc.Markdown('**display region**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
	    			dbc.Row(dcc.Graph(figure={}, id='lines3_c2'))], label='for a same sector', activeTabClassName='fw-bold')]
	    		), label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab(dbc.Tabs([
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric3_s1'), width=2)],),
	    			dbc.Row(dcc.Graph(figure={}, id='lines3_s1'))], label='worldwide', activeTabClassName='fw-bold'),
    				
    				dbc.Tab([html.Br(),
	    			dbc.Row([dbc.Col(dcc.Markdown('**select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'gas vulnerability', id='metric3_s2'), width=2),]),
    				dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(countries, 'USA', id='unit3_s2'), width=2)]),
    				dbc.Row(dcc.Graph(figure={}, id='lines3_s2'))], label='Single country', activeTabClassName='fw-bold')]
    			), label='Sectors', activeTabClassName='fw-bold')])
    		, title='Line charts'),
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    dbc.Tab([
    	    html.Br(),
    	    dbc.Offcanvas(dcc.Checklist(countries, countries, id='unit4_s1c', inputStyle={'margin-right':'10px'}), id='canvas4_s1c', is_open=False, placement='end', scrollable=True),
    	    dbc.Row([dbc.Col(dbc.Button('Display countries', id='open_canvas4_s1c', n_clicks=0), width=1.5)]),
    	    html.Br(),
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+economic_list, 'forward linkage', id='metric4x_s1'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+economic_list, 'backward linkage', id='metric4y_s1'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(vulnerability_list+ ['structural index', 'linkage index'], 'oil vulnerability', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4i_s1'), width=5)]),
			dbc.Row([dbc.Col(dcc.Markdown('**display regions\n(marker style)**', style={'textAlign':'right', 'white-space':'pre'}), width=2),
    		    dbc.Col(dcc.Checklist(regions_list, ['Europe'], id='group4_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    		   
    		dbc.Row([dbc.Col(dcc.Markdown('**display sector groups (marker color)**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Checklist(names7, names7[:-1], id='group4_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    		
    		dbc.Row([dbc.Col(dcc.Markdown('**select color according to**', style={'textAlign':'right'}), width=2),
    		        dbc.Col([
    		        dbc.Row([
    		        dcc.RadioItems(['sector group', 'vulnerability'], 'sector group', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px', 'margin-bottom':'15px'}, id='color4'),
    		        dbc.Row([dcc.Markdown('if vulnerability, select which one:', style={'textAlign':'left'}),
    		                dcc.RadioItems(vulnerability_list, 'oil vulnerability', inline=True, inputStyle={'margin-top':'0px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4v_s1'), ]) ])], width=8),]),
    		    
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2000, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s1')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s1'))], label='Worldwide (decarbonization-related sectors)', activeTabClassName='fw-bold'),
    	    
    	    dbc.Tab([html.Br(),
    	    dbc.Offcanvas([dcc.Checklist(sector_groups, sector_groups, id='group4_s2', inline=False, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}),
    	    html.Br(),
    		dcc.Checklist(sectors, sectors, id='unit4_s2', inputStyle={'margin-right':'10px'})], id='canvas4_s2', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Select sectors', id='open_canvas4_s2', n_clicks=0), width=2),
    		html.Br(),
    		dbc.Row([dbc.Col(dcc.Markdown('**select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+economic_list, 'forward linkage', id='metric4x_s2'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structural_list+economic_list, 'backward linkage', id='metric4y_s2'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(countries, 'USA', id='unit4_s2c'), width=2)]),
			dbc.Row([dbc.Col(dcc.Markdown('**select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.RadioItems(['structural index', 'vulnerability index', 'linkage index'], 'structural index', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4i_s2'))]),
		    dbc.Row([dbc.Col(dcc.Markdown('**select color**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.RadioItems(vulnerability_list, 'coal vulnerability', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric4v_s2'), width=6)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2000, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year4_s2')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s2'))], label='Single country (all sectors)', activeTabClassName='fw-bold')    		
    		]), title='Bubble plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**select country**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(countries_list, 'Brazil', id='unit5'), width=2)]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Waves'),
    	
    	dbc.AccordionItem([
    		# insert number of edges (max, per sector group)
    		# add details on what are Rest of World regions
    		dbc.Row([dbc.Col(dcc.Markdown('**select year**', style={'textAlign':'right'}), width=2),
    			    dbc.Col([html.Br(), html.Br(), daq.Slider(min=1995, max=2020, step=1, value=2020, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='year6')])]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select energy**', style={'textAlign':'right'}), width=2),
    		        dbc.Col(dcc.RadioItems(vulnerability_list, 'oil vulnerability', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}, id='metric6'), width=5),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**select number of links per sector group**', style={'textAlign':'right'}), width=2),
    			    dbc.Col([html.Br(), html.Br(), daq.Slider(min=10, max=100, step=10, value=50, labelPosition='bottom', handleLabel={'showCurrentValue':True, 'label':' ', 'color':'#3e7cc8'}, size=500, id='nbedges6')])]),
                		
    		dbc.Row(dcc.Graph(figure={}, id='map6'))]
    		, title='Worldmap'),
			], flush=True)
			])

@callback(Output('hist1', 'figure'), [Input(s, 'value') for s in ['metric1','unit1','group1_s','group1_c','year1','order1','type1']])
def update_histogram(metric1,unit1,group1_s,group1_c,year1,order1,type1):
    order, dataset, color = (order1=='original')*'trace' + (order1=='descending')*'total descending' + (order1=='ascending')*'total ascending', data_y[year1].reset_index(), 'sector'*(unit1 in ['country','region']) + 'country'*(unit1=='sector')
    gsinds, gcinds = sum([sector_groupmap[g] for g in group1_s],[]), sum([country_groups[c] for c in group1_c],[])
    dataset = deepcopy(dataset.iloc[sorted([NS*c+s for c in gcinds for s in gsinds])])
    if type1:
        return px.bar(dataset, x=metric1, y=unit1, orientation='h', color=color).update_yaxes(categoryorder=order, autorange='reversed').update_layout(xaxis_title='cumulated '+metric1, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector')).update_traces(hovertemplate='<b>%{y} ('+str(year1)+')</b><br>'+str(metric1)+': %{x:.2f}%')
    else:
        return px.histogram(dataset, x=metric1, y=unit1, histfunc='avg', orientation='h').update_yaxes(categoryorder=order, autorange='reversed').update_layout(xaxis_title=metric1, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector')).update_traces(hovertemplate='<b>%{y} ('+str(year1)+')</b><br>average '+str(metric1)+': %{x:.2f}%')

@callback(Output('heatmap2_c', 'figure'), [Input(s, 'value') for s in ['metric2_c','group2_c','order2_c','unit2_c']])
def update_cheatmap(metric2_c,group2_c,order2_c,unit2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    gcinds = sum([country_groups[c] for c in group2_c],[])
    cinds = list(set([country_to_idx[c] for c in unit2_c]) & set(gcinds))
    data_c = deepcopy(data_cavg.iloc[sorted([k*NC+i for k in range(NY) for i in cinds])])
    return px.density_heatmap(data_c, x='year', y='country', z=metric2_c, height=70*len(cinds)**.63, width=450+260, labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=len(unit2_c), color_continuous_scale='Turbo').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_c)+': %{z:.2f}%')

@callback(Output('heatmap2_s1', 'figure'), [Input(s, 'value') for s in ['metric2_s1','group2_s1', 'order2_s1','unit2_s1']])
def update_s1heatmap(metric2_s1,group2_s1,order2_s1,unit2_s1):
    order = (order2_s1=='original')*'trace' + (order2_s1=='descending')*'total descending' + (order2_s1=='ascending')*'total ascending'
    gs1inds = sum([sector_groupmap[g] for g in group2_s1],[])
    s1inds = list(set([sector_to_idx[s] for s in unit2_s1]) & set(gs1inds))
    data_s1 = deepcopy(data_s1avg.iloc[sorted([k*NS+i for k in range(NY) for i in s1inds])])
    return px.density_heatmap(data_s1, x='year', y='sector', z=metric2_s1, height=45*len(s1inds)**.78, width=680+260, labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=len(s1inds), color_continuous_scale='Jet').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_s1)+': %{z:.2f}%')

@callback(Output('heatmap2_s2', 'figure'), [Input(s, 'value') for s in ['metric2_s2','group2_s2','order2_s2','unit2_s2']])
def update_s2heatmap(metric2_s2,group2_s2,order2_s2,unit2_s2):
    order = (order2_s2=='original')*'trace' + (order2_s2=='descending')*'total descending' + (order2_s2=='ascending')*'total ascending'
    gs2inds = sum([sector_groupmap[g] for g in group2_s2],[])
    s2inds = list(set([sector_to_idx[s] for s in unit2_s2])&set(gs2inds))
    data_s2 = {r:deepcopy(data_s2avg[r].iloc[sorted([k*NS+i for k in range(NY) for i in s2inds])]) for r in regions_list}
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=.005, shared_yaxes=True, subplot_titles=['<b>'+str(i)+'</b>' for i in regions_list])
    for r_idx in range(len(regions_list)):
        r = regions_list[r_idx]
        fig.add_trace(px.density_heatmap(data_s2[r], x='year', y='sector', z=metric2_s2, nbinsx=NY, nbinsy=len(s2inds)).data[0], row=1, col=r_idx+1).update_traces(hovertemplate='<b>%{y} ('+r+', %{x})</b><br>'+str(metric2_s2)+': %{z:.2f}%')
    fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(height=15+45*len(s2inds)**.78, width=720+4*260, font={'size':15}, showlegend=True, coloraxis_colorbar={'title':metric2_s2, 'orientation':'v'}, coloraxis={'colorscale':'Jet'})
    return fig

@callback(Output('lines3_c1', 'figure'), [Input(s, 'value') for s in ['metric3_c1','region3_c1']])
def update_c1line(metric3_c1,region3_c1):
    inds = [i for r in region3_c1 for i in region_to_country[r]]
    c1_indices = sorted([k for j in inds for k in np.arange(j,NY*NC,NC)])
    dataset = deepcopy(data_cavg.iloc[c1_indices])
    dataset['country'] = dataset['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    try:
        fig = px.line(dataset, x='year', y=metric3_c1, color='country', labels={'x':'year', 'y':metric3_c1, 'color':'country'}, markers=True, hover_name='country')
    except:
        fig = px.line(dataset, x='year', y=metric3_c1, color='country', labels={'x':'year', 'y':metric3_c1, 'color':'country'}, markers=True, hover_name='country')
    fig.update_xaxes(tickvals= np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<br>year: %{x}<br>'+str(metric3_c1)+': %{y:.2f}%')
    return fig

@callback(Output('lines3_c2', 'figure'), [Input(s, 'value') for s in ['metric3_c2','unit3_c2','region3_c2']])
def update_c2line(metric3_c2,unit3_c2,region3_c2):
    inds = [i for r in region3_c2 for i in region_to_country[r]]
    c2_indices = sorted([k for j in inds for k in np.arange(j,NY*NC,NC)])
    dataset = deepcopy(H[H['sector']==unit3_c2].iloc[c2_indices])
    dataset['country'] = dataset['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    try:
        fig = px.line(dataset, x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2, 'color':'sector'}, markers=True, hover_name='country', hovertemplate='<b>%{color}</b><br>year: %{x}<br>'+str(metric3_c2)+': %{y:.2f}%')
    except:
        fig = px.line(dataset, x='year', y=metric3_c2, color='country', labels={'x':'year', 'y':metric3_c2, 'color':'sector'}, markers=True, hover_name='country')
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='year: %{x}<br>'+str(metric3_c2)+': %{y:.2f}%')
    return fig

@callback(Output('lines3_s1', 'figure'), Input('metric3_s1', 'value'))
def update_s1line(metric3_s1):
    try:
        fig = px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1, 'color':'sector'}, markers=True, hover_name='sector')
    except:
        fig = px.line(data_s1avg, x='year', y=metric3_s1, color='sector', labels={'x':'year', 'y':metric3_s1, 'color':'sector'}, markers=True, hover_name='sector')
    fig.update_xaxes(tickvals= np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='year: %{x}<br>'+str(metric3_s1)+': %{y:.2f}%')
    return fig

@callback(Output('lines3_s2', 'figure'), [Input(s, 'value') for s in ['metric3_s2','unit3_s2']])
def update_s2line(metric3_s2,unit3_s2):
    try:
        fig = px.line(H[H.country==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2, 'color':'country'}, markers=True, hover_name='sector')
    except:
        fig = px.line(H[H.country==unit3_s2], x='year', y=metric3_s2, color='sector', labels={'x':'year', 'y':metric3_s2, 'color':'country'}, markers=True, hover_name='sector')
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':16}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='year: %{x}<br>'+str(metric3_s2)+': %{y:.2f}%')
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
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet', custom_data= ['sector','country',zlabel, xlabel,ylabel])
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet', custom_data= ['sector','country',zlabel, xlabel,ylabel])
    fig.update_layout(width=900, height=700, xaxis_range=[xmin,xmax], yaxis_range=[ymin,ymax], font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]}, %{customdata[1]}</b><br>', str(zlabel)+': %{customdata[2]:.2f}', str(xlabel)+': %{customdata[3]:.2f})', str(ylabel)+': %{customdata[4]:.2f}']))
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
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet', custom_data=['sector',zlabel, xlabel,ylabel])
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet', custom_data=['sector',zlabel, xlabel,ylabel])
    fig.update_layout(width=900, height=700, font={'size':16}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]}</b><br>', str(zlabel)+': %{customdata[1]}', str(xlabel)+': %{customdata[2]:.2f})', str(ylabel)+': %{customdata[3]:.2f}']))
    return fig

@callback(Output('waves5', 'figure'), Input('unit5', 'value'))
def update_waves(unit5):
    try:
        fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'vulnerability (%)'})
    except:
        fig = px.area(data_cvuln[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'vulnerability (%)'})
    fig.update_traces(hovertemplate='%{y:.2f}%')
    return fig

@callback(Output('map6', 'figure'), [Input(s, 'value') for s in ['year6', 'metric6', 'nbedges6']])
def update_maps(year6, metric6, nbedges6):
    dataset, colorbar_legend = worldmap_links[worldmap_links.year==year6].reset_index(), metric6
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
    app.run_server(debug=True, port=8053)
