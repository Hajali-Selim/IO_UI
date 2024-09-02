from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_daq as daq
import dash_mantine_components as dmc
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
from PIL import Image

pd.options.mode.chained_assignment = None
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

H = pd.read_csv('processed_data.csv', compression='bz2')
worldmap_nodes, worldmap_table, worldmap_plot, sector_group_scheme = pd.read_csv('worldmap_nodes.csv'), pd.read_csv('worldmap_table.csv'), pd.read_csv('worldmap_plot.csv'), Image.open('worldmap_scheme.png')

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
structuralnorm_list = ['normalised '+s for s in structural_list]

metrics_list = vulnerability_list+structural_list+economic_list
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list] = H[metrics_list].astype(float)
for s in ['year']+units_list:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)

# leaving only exogeneous vulnerability
for i,j in [('coal vulnerability','Mining of coal, lignite and peat'), ('oil vulnerability','Extraction of crude petroleum'), ('gas vulnerability','Extraction of natural gas')]:
    row, col = H[H.sector==j].index, np.where(H.columns==i)[0][0]
    H.iloc[row,col] = 0

data_cavg = H.groupby(['year','country'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','country'])
data_cavg.insert(1, 'region', list(regions)*NY)
data_s1avg = H.groupby(['year','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','sector'])
data_s2avg = {c[0]:c[1].groupby(['year','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','sector']) for c in H.groupby('region', observed=False)}

data_cvuln, data_rvuln, year_NYlist, sector_NYlist = {}, {}, np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil vulnerability','gas vulnerability','coal vulnerability'] for i in range(NY)])
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['country']==c][['oil vulnerability','gas vulnerability','coal vulnerability']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

for r in regions_list:
    data_rvuln[r] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg['region']==r].groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_rvuln[r]['year'], data_rvuln[r]['sector'], data_rvuln[r]['vulnerability'] = data_rvuln[r]['year'].astype(int), data_rvuln[r]['sector'].astype(str), data_rvuln[r]['vulnerability'].astype(float)

data_rvuln['Worldwide'] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg.groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
data_rvuln['Worldwide']['year'], data_rvuln['Worldwide']['sector'], data_rvuln['Worldwide']['vulnerability'] = data_rvuln['Worldwide']['year'].astype(int), data_rvuln['Worldwide']['sector'].astype(str), data_rvuln['Worldwide']['vulnerability'].astype(float)

sector_groups, sector_group_colors = ['Agriculture, forestry and fishing', 'Extraction and mining', 'Manufacture and production', 'Utilities', 'Services'], ['lime', 'orange', 'red', 'cyan', 'purple']
sector_groupmap = {'Agriculture, forestry and fishing':list(range(19)), 'Extraction and mining':list(range(19,34)), 'Manufacture and production':list(range(34,93))+[109,113], 'Utilities':list(range(93,119))+[110,111,112]+list(range(114,120)), 'Services':list(range(120,163))}
country_groups = {r:list(np.where(regions==r)[0]) for r in regions_list}
decarb_groups = {names7[i]:list(np.where(H.sector_color[:NS]==colors7[i])[0]) for i in range(7)}
data_y = {int(c[0]):c[1] for c in H.groupby('year', observed=False)[units_list+metrics_list]}

app.layout = html.Div(children=[
    dcc.Markdown('''# Global Economic Vulnerability''', style={'textAlign':'center'}),
    html.Hr(),
    dbc.Accordion([
    	dbc.AccordionItem([
    		dbc.Row([# row of data selection (parameter+ordering)
    		dbc.Col([# column of parameter selection
    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Dropdown(vulnerability_list+ structural_list+economic_list, 'coal vulnerability', id='metric1'), width=3),
    		        dbc.Col(dcc.Markdown('**Decompose metric**', style={'textAlign':'right'}), width=2), dbc.Col(dbc.Checklist(id='type1', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}), width=4),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select unit**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Dropdown(units_list, 'country', id='unit1'), width=3),
    		        dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2), dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order1'), width=5), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select type of data**', style={'textAlign':'right'}), width=2),
    		        dbc.Col(
    		        dbc.Row([
        		        dbc.Col(
        		        dmc.SegmentedControl(['single year', 'range of years'], 'single year', id='changes1'), width=3),
        		        dcc.Markdown('If \'single year\', select which one:'), html.Br(),
        		        dcc.Slider(min=1995, max=2020, step=1, value=2000, marks={1995:'1995', 2020:'2020'}, tooltip={'placement':'top', 'always_visible':True}, id='year1'),
        		        dcc.Markdown('If \'range of years\', select a range:'), html.Br(),
        		        dcc.RangeSlider(min=1995, max=2020, step=1, value=[2015,2020], marks={1995:'1995', 2020:'2020'}, tooltip={'placement':'top', 'always_visible':True}, id='range1'),
        		        ]), width=8)
    		        ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Display regions**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Checklist(regions_list, regions_list, id='group1_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6), ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Display sector groups**', style={'textAlign':'right'}), width=2), dbc.Col(dcc.Checklist(sector_groups, sector_groups, id='group1_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)
    		        ])], ), ]),
    		dcc.Graph(figure={}, id='hist1')], title='Histograms'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([html.Br(),
    				dbc.Offcanvas([
    				dcc.Checklist(countries, countries, id='unit2_c', inputStyle={'margin-right':'10px'})], id='canvas2_c', is_open=False, placement='end', scrollable=True),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_c'), width=3),]),
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_c',), width=4)]),
    				dbc.Row([
    				    dbc.Col(dcc.Markdown('**Display regions**', style={'textAlign':'right'}), width=2),
    				    dbc.Col(dcc.Checklist(regions_list, regions_list, id='group2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    				dbc.Row([dbc.Col(width=1), dbc.Col(dbc.Button('Display countries', id='open_canvas2_c', n_clicks=0), width=2),]),
    				dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab([dbc.Row(dbc.Col(dmc.SegmentedControl(id='segment2_s', value='Worldwide averages', data=['Worldwide averages', 'Regional averages'], radius=5, size='md'), width=3),),
    			html.Br(),
    			dbc.Offcanvas([
                dcc.Checklist(sectors, sectors, id='unit2_s', inputStyle={'margin-right':'10px'})], id='canvas2_s', is_open=False, placement='end', scrollable=True),
    			dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_s'), width=3),]),
    			dbc.Row([dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_s'), width=6)]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Checklist(sector_groups, sector_groups, id='group2_s', inline=True, inputStyle={'margin-top':'10px','margin-right':'5px','margin-left':'30px'}), width=6)]),
    			dbc.Row([dbc.Col(width=1), dbc.Col(dbc.Button('Display sectors', id='open_canvas2_s', n_clicks=0), width=2),]),
    			dbc.Row(dcc.Graph(figure={}, id='heatmap2_s')),]
    			, label='Sectors', activeTabClassName='fw-bold')]),
    			
    			title='Heatmaps'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([
    		    
	    		dbc.Tab([html.Br(),
    			dbc.Row([
    				dbc.Col(dcc.Markdown('**Select sectors**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(['All sectors (average)']+list(sectors), 'All sectors (average)', id='unit3_c'), width=4),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric3_c'), width=3), ]),
    			dbc.Row([dbc.Col(dcc.Markdown('**Display region**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_c')),],
	    		label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab([html.Br(),
    			dbc.Row([dbc.Col(dcc.Markdown('**Select countries**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(['All countries (average)']+list(countries), 'All countries (average)', id='unit3_s'), width=3),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(metrics_list, 'gas vulnerability', id='metric3_s'), width=3), ]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_s')),],
    			label='Sectors', activeTabClassName='fw-bold')])
    		
    		, title='Line charts'),
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    dbc.Tab([
    	    html.Br(),
    	    dbc.Offcanvas(dcc.Checklist(countries, countries, id='unit4_s1c', inputStyle={'margin-right':'10px'}), id='canvas4_s1c', is_open=False, placement='end', scrollable=True),
    	    dbc.Row([dbc.Col(dbc.Button('Display countries', id='open_canvas4_s1c', n_clicks=0), width=1.5)]),
    	    html.Br(),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structuralnorm_list+economic_list, 'forward linkage', id='metric4x_s1'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structuralnorm_list+economic_list, 'backward linkage', id='metric4y_s1'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dmc.SegmentedControl(vulnerability_list+ ['structural index', 'linkage index'], 'oil vulnerability', id='metric4i_s1'), width=5)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Display regions\n(marker style)**', style={'textAlign':'right', 'white-space':'pre'}), width=2),
    		    dbc.Col(dcc.Checklist(regions_list, ['Europe'], id='group4_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6), ]),
    		   
    		dbc.Row([dbc.Col(dcc.Markdown('**Display sector groups (marker color)**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dcc.Checklist(names7, names7[:-1], id='group4_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5), ]),
    		
    		dbc.Row([dbc.Col(dcc.Markdown('**Select color according to**', style={'textAlign':'right'}), width=2),
    		        dbc.Col([
    		        dbc.Row([dbc.Col(dmc.SegmentedControl(['sector group', 'vulnerability'], 'sector group', id='color4'), width=2),
    		        dbc.Row([dcc.Markdown('If vulnerability, select:', style={'textAlign':'left'}),
    		                dbc.Col(dmc.SegmentedControl(vulnerability_list, 'oil vulnerability', id='metric4v_s1'), width=4)]) ])], width=2),]),
    		    
    		dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), dcc.Slider(min=1995, max=2020, step=1, value=2000, marks={1995:'1995', 2020:'2020'}, tooltip={'placement':'top', 'always_visible':True}, id='year4_s1')], width=8)]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s1'))], label='Worldwide (decarbonization-related sectors)', activeTabClassName='fw-bold'),
    	    
    	    dbc.Tab([html.Br(),
    	    dbc.Offcanvas([
    		dcc.Checklist(sectors, sectors, id='unit4_s2', inputStyle={'margin-right':'10px'})], id='canvas4_s2', is_open=False, placement='end', scrollable=True),
    		dbc.Col(dbc.Button('Display sectors', id='open_canvas4_s2', n_clicks=0), width=2),
    		html.Br(),
            dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Checklist(sector_groups, sector_groups, id='group4_s2', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structuralnorm_list+economic_list, 'forward linkage', id='metric4x_s2'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(structuralnorm_list+economic_list, 'backward linkage', id='metric4y_s2'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select country**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(countries, 'USA', id='unit4_s2c'), width=3)]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select marker size**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dmc.SegmentedControl(['structural index', 'vulnerability index', 'linkage index'], 'structural index', id='metric4i_s2'))]),
		    dbc.Row([dbc.Col(dcc.Markdown('**Select color**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dmc.SegmentedControl(vulnerability_list, 'coal vulnerability', id='metric4v_s2'), width=6)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
    			dbc.Col([html.Br(), html.Br(), dcc.Slider(min=1995, max=2020, step=1, value=2000, marks={1995:'1995', 2020:'2020'}, tooltip={'placement':'top', 'always_visible':True}, id='year4_s2')])]),
    		dbc.Row(dcc.Graph(figure={}, id='bubble4_s2'))], label='Single country (all sectors)', activeTabClassName='fw-bold')    		
    		]), title='Scatter plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**Select region or country**', style={'textAlign':'right'}), width=3),
    		        dbc.Col(dcc.Dropdown(['Worldwide (average)']+[r+' (average)' for r in regions_list]+countries_list, 'Brazil', id='unit5'), width=3)]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Waves'),
    	
    	dbc.AccordionItem([
    		dbc.Row([
    		dbc.Col([dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=3),
    			    dbc.Col(dcc.Slider(min=1995, max=2020, step=1, value=2000, marks={1995:'1995', 2020:'2020'}, tooltip={'placement':'top', 'always_visible':True}, id='year6'))]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (country color)**', style={'textAlign':'right'}), width=3),
    		        dbc.Col(dmc.SegmentedControl(vulnerability_list, 'oil vulnerability', id='metric6'), width=5),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select number of links per sector group**', style={'textAlign':'right'}), width=3),
    			    dbc.Col(dcc.Slider(min=10, max=100, step=10, value=40, marks={10:'10', 100:'100'}, tooltip={'placement':'top', 'always_visible':True}, id='nbedges6'))]),], width=8),
    		dbc.Col([html.Img(src=sector_group_scheme),], width=4),]),
    		dbc.Row([
    		dbc.Col(dcc.Markdown('Ndlr: All 163 sectors are aggregated into 5 main sector groups. Transactions are represented by links whose colours correspond to the exporting sector group, as shown in the colour coding in the image opposite. Click on a country (node) to view a table below detailing its major imports and exports (links), as links may overlap on the world map figure. Amounts are in millions of USD.', style={'font-size':13}), width=8),
    		dbc.Row(dcc.Graph(figure={}, id='map6')),
            dbc.Row([html.Pre(id='title6_exports'),
                    dash_table.DataTable(id='table6_exports', css=[{'selector': '.dash-spreadsheet td div', 'rule':'''max-height: 10px'''}], style_data={'whiteSpace':'normal'}),
                    html.Br(),
                    html.Pre(id='title6_imports'),
                    dash_table.DataTable(id='table6_imports', css=[{'selector': '.dash-spreadsheet td div', 'rule':'''max-height: 10px'''}], style_data={'whiteSpace':'normal'})                    ,
                    ]),]),]
    		, title='World map'),
			], flush=True)
			])

@callback(Output('hist1', 'figure'), [Input(s, 'value') for s in ['metric1','unit1','group1_s','group1_c','year1','order1','type1','changes1', 'range1']])
def update_histogram(metric1,unit1,group1_s,group1_c,year1,order1,type1,changes1,range1):
    order, color = (order1=='original')*'trace' + (order1=='descending')*'total descending' + (order1=='ascending')*'total ascending', 'sector'*(unit1 in ['country','region']) + 'country'*(unit1=='sector')
    gsinds, gcinds = sum([sector_groupmap[g] for g in group1_s],[]), sum([country_groups[c] for c in group1_c],[])
    if changes1 == 'range of years': # if looking at variations
        year0, year1 = range1
        dataset_previous, dataset = data_y[year0][metric1].reset_index(), data_y[year1][['region','country','sector',metric1]].reset_index()
        dataset[metric1] = dataset[metric1].subtract(dataset_previous[metric1])
    else:
        dataset = data_y[year1][['region','country','sector',metric1]]
    dataset = deepcopy(dataset.iloc[sorted([NS*c+s for c in gcinds for s in gsinds])]).reset_index()
    if type1: # if decomp
        decomp = 'country'*(unit1 == 'sector') + 'sector'*(unit1 in ['country','region'])
        return px.bar(dataset, x=metric1, y=unit1, orientation='h', color=color, custom_data=[decomp]).update_yaxes(categoryorder=order, autorange='reversed').update_layout(xaxis_title='cumulative '+metric1, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector')).update_traces(hovertemplate='<b>%{customdata[0]} (%{y}, '+str(year1)+')</b><br>'+str(metric1)+': %{x:.2f}%')
    else:
        return px.histogram(dataset, x=metric1, y=unit1, histfunc='avg', orientation='h').update_yaxes(categoryorder=order, autorange='reversed').update_layout(xaxis_title=metric1, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector')).update_traces(hovertemplate='<b>%{y}, '+str(year1)+'</b><br>average '+str(metric1)+': %{x:.2f}%')

@callback(Output('heatmap2_c', 'figure'), [Input(s, 'value') for s in ['metric2_c','group2_c','order2_c','unit2_c']])
def update_cheatmap(metric2_c,group2_c,order2_c,unit2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    gcinds = sum([country_groups[c] for c in group2_c],[])
    cinds = list(set([country_to_idx[c] for c in unit2_c]) & set(gcinds))
    data_c = deepcopy(data_cavg.iloc[sorted([k*NC+i for k in range(NY) for i in cinds])])
    return px.density_heatmap(data_c, x='year', y='country', z=metric2_c, height=70*len(cinds)**.63, width=450+260, labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=len(unit2_c), color_continuous_scale='Turbo').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_c)+': %{z:.2f}%')

@callback(Output('heatmap2_s', 'figure'), [Input(s, 'value') for s in ['segment2_s','metric2_s','group2_s', 'order2_s','unit2_s']])
def update_sheatmap(segment2_s,metric2_s,group2_s,order2_s,unit2_s):
    order = (order2_s=='original')*'trace' + (order2_s=='descending')*'total descending' + (order2_s=='ascending')*'total ascending'
    gsinds = sum([sector_groupmap[g] for g in group2_s],[])
    sinds = list(set([sector_to_idx[s] for s in unit2_s]) & set(gsinds))
    if segment2_s == 'Worldwide averages':
        return px.density_heatmap(data_s1avg.iloc[sorted([k*NS+i for k in range(NY) for i in sinds])], x='year', y='sector', z=metric2_s, height=45*len(sinds)**.78, width=680+260, labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=len(sinds), color_continuous_scale='Jet').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':15}, showlegend=True, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_s)+': %{z:.2f}%')
    else:
        fig = make_subplots(rows=1, cols=4, horizontal_spacing=.005, shared_yaxes=True, subplot_titles=['<b>'+str(i)+'</b>' for i in regions_list])
        for r_idx in range(len(regions_list)):
            r = regions_list[r_idx]
            fig.add_trace(px.density_heatmap(data_s2avg[r].iloc[sorted([k*NS+i for k in range(NY) for i in sinds])], x='year', y='sector', z=metric2_s, nbinsx=NY, nbinsy=len(sinds)).data[0], row=1, col=r_idx+1).update_traces(hovertemplate='<b>%{y} ('+r+', %{x})</b><br>'+str(metric2_s)+': %{z:.2f}%')
        fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(height=15+45*len(sinds)**.78, width=720+4*260, font={'size':15}, showlegend=True, coloraxis_colorbar={'title':metric2_s, 'orientation':'v'}, coloraxis={'colorscale':'Jet'})
        return fig

@callback(Output('lines3_c', 'figure'), [Input(s, 'value') for s in ['metric3_c','unit3_c','region3_c']])
def update_cline(metric3_c,unit3_c,region3_c):
    inds = [i for r in region3_c for i in region_to_country[r]]
    c_indices = sorted([k for j in inds for k in np.arange(j,NY*NC,NC)])
    if unit3_c == 'All sectors (average)':
        dataset = data_cavg.iloc[c_indices]
        dataset['country'] = deepcopy(dataset)['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    else:
        dataset = H[H['sector']==unit3_c].iloc[c_indices]
        dataset['country'] = deepcopy(dataset)['country'].cat.set_categories(np.array(countries)[inds], ordered=True)
    try:
        fig = px.line(dataset, x='year', y=metric3_c, color='country', labels={'x':'year', 'y':metric3_c, 'color':'country'}, markers=True, custom_data=['country'])
    except:
        fig = px.line(dataset, x='year', y=metric3_c, color='country', labels={'x':'year', 'y':metric3_c, 'color':'country'}, markers=True, custom_data=['country'])
    fig.update_xaxes(tickvals= np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':14}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_c)+': %{y:.2f}<extra></extra>')
    return fig

@callback(Output('lines3_s', 'figure'), [Input(s, 'value') for s in ['metric3_s','unit3_s']])
def update_sline(metric3_s,unit3_s):
    if unit3_s == 'All countries (average)':
        dataset = data_s1avg
    else:
        dataset = H[H.country==unit3_s]
    try:
        fig = px.line(dataset, x='year', y=metric3_s, color='sector', labels={'x':'year', 'y':metric3_s, 'color':'country'}, markers=True, custom_data=['sector'])
    except:
        fig = px.line(dataset, x='year', y=metric3_s, color='sector', labels={'x':'year', 'y':metric3_s, 'color':'country'}, markers=True, custom_data=['sector'])
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_layout(font={'size':15}, height=700, width=1300, hoverlabel={'font_size':14}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_s)+': %{y:.2f}<extra></extra>')
    return fig

@callback(Output('bubble4_s1', 'figure'), [Input(s, 'value') for s in ['metric4x_s1','metric4y_s1','metric4i_s1', 'group4_c','unit4_s1c', 'group4_s','year4_s1', 'color4', 'metric4v_s1']])
def update_s1bubble(metric4x_s1,metric4y_s1,metric4i_s1,group4_c,unit4_s1c,group4_s,year4_s1,color4,metric4v_s1):
    xlabel, ylabel, zlabel = metric4x_s1, metric4y_s1, metric4i_s1
    gcinds, sinds = sum([country_groups[c] for c in group4_c],[]), sum([decarb_groups[c] for c in group4_s],[])
    cinds = list(set([country_to_idx[c] for c in unit4_s1c]) & set(gcinds))
    dataset = deepcopy(H[H.year==year4_s1].reset_index().iloc[sorted([c*NS+s for c in cinds for s in sinds]),1:])
    size = deepcopy(dataset[zlabel]*1000)
    if xlabel in structuralnorm_list:
        xmin, xmax = .01, .5
    else:
        xmin, xmax = .5, 2
    if ylabel in structuralnorm_list:
        ymin, ymax = .01, .5
    else:
        ymin, ymax = .5, 2
    zmin, zmax = dataset[zlabel].min(), dataset[zlabel].max()
    color4 = 'sector_color'*(color4=='sector group') + metric4v_s1*(color4=='vulnerability')
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=size, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet', custom_data= ['country','sector',zlabel,xlabel,ylabel])
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=size, color=color4, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', symbol='region_marker', opacity=.7, color_continuous_scale='Jet', custom_data= ['country','sector',zlabel,xlabel,ylabel])
    fig.update_layout(width=1150, height=750, xaxis_range=[xmin,xmax], yaxis_range=[ymin,ymax], font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]},<br>%{customdata[1]}</b><br>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}']))
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
    if xlabel in structuralnorm_list:
        xmin, xmax = .01, .5
    else:
        xmin, xmax = .5, 2
    if ylabel in structuralnorm_list:
        ymin, ymax = .01, .5
    else:
        ymin, ymax = .5, 2
    zmin, zmax = dataset[zlabel].min(), dataset[zlabel].max()
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet', custom_data=['sector',zlabel, xlabel,ylabel])
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=metric4v_s2, labels={'x':xlabel, 'y':ylabel}, range_x=[xmin-.1,xmax+.1], range_y=[ymin-.1,ymax+.1], range_color=[zmin-.02, zmax+.02], hover_name='sector', opacity=.7, color_continuous_scale='Jet', custom_data=['sector',zlabel, xlabel,ylabel])
    fig.update_layout(width=1050, height=750, font={'size':16}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v'}, hoverlabel={'font_size':18}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).add_hline(y=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).add_vline(x=1, line_width=3, line_dash='dash', line_color='red', opacity=.7).update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]}</b><br>', str(zlabel)+': %{customdata[1]:.2f}', str(xlabel)+': %{customdata[2]:.2f}', str(ylabel)+': %{customdata[3]:.2f}']))
    return fig

@callback(Output('waves5', 'figure'), Input('unit5', 'value'))
def update_waves(unit5):
    if unit5[:-10] in regions_list+['Worldwide']:
        dataset, unit5 = data_rvuln, unit5[:-10]
    else:
        dataset = data_cvuln
    try:
        fig = px.area(dataset[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'cumulative vulnerability (%)'}).update_layout(legend={'font_size':16})
    except:
        fig = px.area(dataset[unit5], x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'cumulative vulnerability (%)'}).update_layout(legend={'font_size':16})
    fig.update_traces(hovertemplate='%{x}: %{y:.2f}%')
    return fig

@callback(Output('map6', 'figure'), [Input(s, 'value') for s in ['year6', 'metric6', 'nbedges6']])
def update_maps(year6, metric6, nbedges6):
    dataset, colorbar_legend = worldmap_plot[worldmap_plot.year==year6].reset_index(), metric6
    fig = go.Figure(go.Choropleth(locations=countries_code, z=data_cavg[data_cavg.year==year6][metric6], colorscale='Reds', colorbar={'title': colorbar_legend}, hoverinfo='skip'))
    for idx in range(nbedges6//10):
        for s in range(5):
            batch, color = deepcopy(dataset.iloc[3*(100*s+10*idx): 3*(100*s+10*(idx+1))]), sector_group_colors[s] 
            fig.add_scattergeo(lat=batch.latitude, lon=batch.longitude, mode='lines', line={'width':1.5, 'color':color}, showlegend=False, hoverinfo='skip')
    fig.add_scattergeo(lat=worldmap_nodes.lat, lon=worldmap_nodes.lon, marker={'size':5, 'symbol':'circle-open', 'color':'black', 'opacity':0.8}, showlegend=False, customdata=worldmap_nodes.EXIOBASE_name, hovertemplate='%{customdata}<extra></extra>').update_layout(clickmode='event+select')
    _ = fig.update_geos(lataxis_range=[-55, 90], showocean=True, oceancolor='LightBlue').update_layout(width=1400, height=500, margin=dict(l=20, r=20, t=0, b=0))
    return fig

@callback(Output('title6_exports', 'children'), [Input('map6', 'clickData'), Input('year6', 'value')])
def update_export_title(clickData, year6):
    if clickData:
        return 'Largest exports from '+str(clickData['points'][0]['customdata'])+' (source country) in '+str(year6)+'.'

@callback(Output('table6_exports', 'data'), [Input('map6', 'clickData')]+[Input(s, 'value') for s in ['year6', 'nbedges6']])
def update_export_table(click_data, year6, nbedges6):
    if click_data:
        dataset = worldmap_table[worldmap_table.year==year6]
        dataset = dataset.iloc[[100*g+s for g in range(5) for s in range(nbedges6)],:-1]
        dataset = dataset[dataset['source country']==click_data['points'][0]['customdata']]
        return dataset.to_dict('records')

@callback(Output('title6_imports', 'children'), [Input('map6', 'clickData'), Input('year6', 'value')])
def update_import_title(clickData, year6):
    if clickData:
        return 'Largest imports to '+str(clickData['points'][0]['customdata'])+' (target country) in '+str(year6)+'.'

@callback(Output('table6_imports', 'data'), [Input('map6', 'clickData')]+[Input(s, 'value') for s in ['year6', 'nbedges6']])
def update_import_table(click_data, year6, nbedges6):
    if click_data:
        dataset = worldmap_table[(worldmap_table.year==year6)]
        dataset = dataset.iloc[[100*g+s for g in range(5) for s in range(nbedges6)],:-1]
        dataset = dataset[dataset['target country']==click_data['points'][0]['customdata']]
        return dataset.to_dict('records')

@app.callback(Output('canvas2_c', 'is_open'), Input('open_canvas2_c', 'n_clicks'), State('canvas2_c', 'is_open'))
def toggle_2ccanvas(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(Output('canvas2_s', 'is_open'), Input('open_canvas2_s', 'n_clicks'), State('canvas2_s', 'is_open'))
def toggle_2s1canvas(n, is_open):
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
    app.run_server(debug=True, port=8050)
