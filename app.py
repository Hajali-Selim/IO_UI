from dash import Dash, html, dash_table, dcc, callback, Output, Input, no_update
import dash_mantine_components as dmc
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
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

H = pd.read_csv('processed_data_non_normalised.csv', compression='bz2')

#worldmap_nodes, worldmap_table, worldmap_plot, sector_group_scheme = pd.read_csv('worldmap_nodes.csv'), pd.read_csv('worldmap_table.csv'), pd.read_csv('worldmap_plot.csv'), Image.open('worldmap_scheme.png')
country_network_dynamics = pd.read_csv('country_network_dynamics.csv')

NS, NC, NY = 163, 49, 26
H = H[H.year<=1994+NY]
N = NS*NC
regions, countries, sectors = H.region.iloc[np.arange(0,N,NS)], H.country.unique(), H.sector.unique()
regions_list, countries_list, sectors_list = list(regions.unique()), list(countries), list(sectors)
#countries_code = [worldmap_nodes.CODE[np.where(worldmap_nodes.EXIOBASE_name==c)[0][0]] for c in countries]

sector_to_idx, country_to_idx, k1, k2 = {}, {}, 0, 0
for s in sectors:
    sector_to_idx[s], k1 = k1, k1+1, 

for c in countries:
    country_to_idx[c], k2 = k2, k2+1

units_list = ['region', 'country', 'sector', 'group']
vulnerability_list, importance_list = ['coal vulnerability', 'gas vulnerability', 'oil vulnerability', 'fossil fuel vulnerability'], ['forward linkage', 'backward linkage', 'weighted in-degree', 'weighted out-degree', 'out-degree', 'in-degree', 'betweenness']
metrics_list = vulnerability_list+importance_list
country_metrics_list = ['modularity', 'in-conductance', 'out-conductance']
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list] = H[metrics_list].astype(float)
for s in ['year']+units_list:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)

for s in ['year','region','country']:
    country_network_dynamics[s] = country_network_dynamics[s].astype('category').cat.set_categories(country_network_dynamics[s].unique(), ordered=True)

data_cavg = H.groupby(['year','country'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','country'])
data_cavg.insert(1, 'region', list(regions)*NY)
data_s1avg = H.groupby(['year','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','sector'])
data_s1avg.insert(4, 'group', list(H.group.iloc[:NS])*NY)
data_s2avg = {c[0]:c[1].groupby(['year','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','sector']) for c in H.groupby('region', observed=False)}
_ = [data_s2avg[r].insert(4, 'group', list(H.group.iloc[:NS])*NY) for r in regions_list]

data_cvuln, data_rvuln, year_NYlist, sector_NYlist = {}, {}, np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil vulnerability','gas vulnerability','coal vulnerability'] for i in range(NY)])
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg.country==c][['oil vulnerability','gas vulnerability','coal vulnerability']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

for r in regions_list:
    data_rvuln[r] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[data_cavg.region==r].groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_rvuln[r]['year'], data_rvuln[r]['sector'], data_rvuln[r]['vulnerability'] = data_rvuln[r]['year'].astype(int), data_rvuln[r]['sector'].astype(str), data_rvuln[r]['vulnerability'].astype(float)

data_rvuln['Worldwide'] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg.groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
data_rvuln['Worldwide'] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg.groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
data_rvuln['Worldwide']['year'], data_rvuln['Worldwide']['sector'], data_rvuln['Worldwide']['vulnerability'] = data_rvuln['Worldwide']['year'].astype(int), data_rvuln['Worldwide']['sector'].astype(str), data_rvuln['Worldwide']['vulnerability'].astype(float)

groups_list, sector_group_colors = ['Agriculture, forestry and fishing', 'Extraction and mining', 'Manufacture and production', 'Utilities', 'Services'], ['lime', 'orange', 'red', 'cyan', 'purple']
data_y = {int(c[0]):c[1] for c in H.groupby('year', observed=False)[units_list+metrics_list]}
clicked_countries, clicked_sectors = [], []

app.layout = html.Div(children=[
    dcc.Markdown('''# Global Economic Vulnerability''', style={'textAlign':'center'}),
    html.Hr(),
    dbc.Accordion([
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric1'), width=3),
                    dbc.Col(dcc.Markdown('**decompose metric**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dbc.Checklist(id='type1', switch=True, value=['On'], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select unit**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(units_list[:-1], 'country', id='unit1'), width=3),
                    dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order1'), width=5),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select type of data**', style={'textAlign':'right'}), width=2),
    		        dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2000, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year1'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                            dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range1'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes1', active_tab='single year'), width=5)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(regions_list, regions_list, id='group1_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv1', n_clicks=0, outline=True, color='dark'), width=3), dcc.Download(id='data1')]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, id='group1_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), )]),
       		dcc.Graph(figure={}, id='hist1')], title='Bar plots'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list+country_metrics_list, 'coal vulnerability', id='metric2_c'), width=3),
    					dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_c',), width=4)]),
    				dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
    				    dbc.Col(dcc.Checklist(regions_list, regions_list, id='region2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6), ]),                    
                    dbc.Row([dbc.Col(dcc.Markdown('**Hide countries**', style={'textAlign':'right'}), width=2),
                        dbc.Col(dcc.Dropdown(countries, [], id='unselect2_c', multi=True), width=3),
                        dbc.Col(dbc.Button('Reset selection', id='reset2_c', outline=True, color='dark'), width=2),
                        dbc.Col(dbc.Button('Download spreadsheet', id='csv2_c', outline=True, color='dark')), dcc.Download(id='data2_c'),]),
                    dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab([dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_s'), width=3),
                        dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_s'), width=4)]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Checklist(groups_list, groups_list, id='group2_s', inline=True, inputStyle={'margin-top':'10px','margin-right':'5px','margin-left':'30px'}), )]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select regional average**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(['Worldwide']+regions_list, 'Worldwide', id='regions2_s'), width=2)]),
                dbc.Row([dbc.Col(dcc.Markdown('**Hide sectors**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(sectors, [], id='unselect2_s', multi=True), width=4),
                    dbc.Col(dbc.Button('Reset selection', id='reset2_s', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv2_s', outline=True, color='dark')), dcc.Download(id='data2_s'),]),
                dbc.Row(dcc.Graph(figure={}, id='heatmap2_s')),]
    			, label='Sectors', activeTabClassName='fw-bold')]),
    			
    			title='Heatmaps'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([
	    		dbc.Tab([
    			dbc.Row([
    				dbc.Col(dcc.Markdown('**Select sectors**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(['All sectors (average)']+list(sectors), 'All sectors (average)', id='unit3_c'), width=4),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(metrics_list+country_metrics_list, 'coal vulnerability', id='metric3_c'), width=3),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv3_c', outline=True, color='dark')), dcc.Download(id='data3_c')]),
    			dbc.Row([dbc.Col(dcc.Markdown('**Select region**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_c')),],
	    		label='Countries', activeTabClassName='fw-bold'),
    			dbc.Tab([
    			dbc.Row([dbc.Col(dcc.Markdown('**Select countries**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(['All countries (average)']+list(countries), 'All countries (average)', id='unit3_s'), width=3),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
                        dbc.Col(dcc.Dropdown(metrics_list, 'gas vulnerability', id='metric3_s'), width=3),
                        dbc.Col(dbc.Button('Download spreadsheet', id='csv3_s', outline=True, color='dark')), dcc.Download(id='data3_s')]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, id='group3_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), ) ]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_s')),],
    			label='Sectors', activeTabClassName='fw-bold')])
    		
    		, title='Line chart'),
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    
    	    dbc.Tab([
			dbc.Row([dbc.Col(dcc.Markdown('**Select country**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(countries_list+['average country'], ['average country'], id='country4_c', multi=True), width=2),
                dbc.Col(dcc.Markdown('different marker symbols', style={'textAlign':'right'}), width=3),
                dbc.Col(dbc.Checklist(id='symbol4_c', switch=True, value=[], options=[{'label':'', 'value':'country'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_c'), width=3),
                #dbc.Col(dcc.Markdown('log-scaled x-axis', style={'textAlign':'right'}), width=4),
                #dbc.Col(dbc.Checklist(id='log4x_s', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_c'), width=3),
                #dbc.Col(dcc.Markdown('log-scaled x-axis', style={'textAlign':'right'}), width=4),
                #dbc.Col(dbc.Checklist(id='log4y_s', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select marker size\n (vulnerability)**', style={'textAlign':'right'}), width=2),
			    dbc.Col(dmc.SegmentedControl(vulnerability_list, 'gas vulnerability', id='metric4i_c'), width=4),
                dbc.Col(dcc.Markdown('size scaling', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Input(value=20, type='number', step=1, inputMode='numeric', style={'width':60}, id='size4_c'), width=1),]),
		    dbc.Row([dbc.Col(dcc.Markdown('**Select marker color**', style={'textAlign':'right'}), width=2),
    			dbc.Col(dmc.SegmentedControl(['vulnerability', 'sector groups'], 'vulnerability', id='color4_c'), width=4),
                #dbc.Col(dcc.Markdown('log2-scaled color', style={'textAlign':'right'}), width=3),
                #dbc.Col(dbc.Checklist(id='log4i_s', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                ]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Checklist(groups_list, groups_list, inline=True, id='group4_c', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), )]),
            dbc.Row([dbc.Col(dcc.Markdown('**Filter-out**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Dropdown(metrics_list, value='out-degree', id='filtermetric4_c'), width=3),
                dbc.Col(dcc.Markdown('**under**'), width=1),
                dbc.Col(dcc.Input(value=0, type='number', min=0, inputMode='numeric', style={'width':60}, id='filternb4_c'), width=1)
                ]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                
                dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2000, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_c'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                            dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_c'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_c', active_tab='single year'), width=5),
                
                dbc.Col(dcc.Markdown('**Fixed color-range**', style={'textAlign':'right'}), width=2),
                dbc.Col(dbc.Checklist(id='cmax4_c', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                ]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a sector data point to hide it and rescale the color-coding and marker sizes.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_c', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_c', outline=True, color='dark')), dcc.Store(id='click4_c'), dcc.Download(id='data4_c')]),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_c')),], label='Countries', activeTabClassName='fw-bold'),
            
            dbc.Tab([
    	    dbc.Row([dbc.Col(dcc.Markdown('**Select sector**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Dropdown(sectors_list+['average sector'], ['average sector'], id='sector4_s', multi=True), width=4),
                dbc.Col(dcc.Markdown('different marker symbols', style={'textAlign':'right'}), width=3),
                dbc.Col(dbc.Checklist(id='symbol4_s', switch=True, value=[], options=[{'label':'', 'value':'sector'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_s'), width=3),
                    #dbc.Col(dcc.Markdown('log-scaled x-axis', style={'textAlign':'right'}), width=4),
                    #dbc.Col(dbc.Checklist(id='log4x_c', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_s'), width=3),
                    #dbc.Col(dcc.Markdown('log-scaled y-axis', style={'textAlign':'right'}), width=4),
                    #dbc.Col(dbc.Checklist(id='log4y_c', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select marker size\n (vulnerability)**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(vulnerability_list, 'oil vulnerability', id='metric4i_s'), width=4),
    		        dbc.Col(dcc.Markdown('size scaling', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Input(value=20, type='number', step=1, inputMode='numeric', style={'width':60}, id='size4_s'), width=1),]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select marker color**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(['vulnerability', 'regions'], 'vulnerability', id='color4_s'), width=4),
                    #dbc.Col(dcc.Markdown('log2-scaled color', style={'textAlign':'right'}), width=3),
                    #dbc.Col(dbc.Checklist(id='log4i_c', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))
                ]),    		
            dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Checklist(regions_list, regions_list, inline=True, id='group4_s', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Filter-out**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Dropdown(metrics_list, value='out-degree', id='filtermetric4_s'), width=3),
                dbc.Col(dcc.Markdown('**under**'), width=1),
                dbc.Col(dcc.Input(value=0, type='number', min=0, inputMode='numeric', style={'width':60}, id='filternb4_s'), width=1)
                ]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                
                dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2000, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_s'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                            dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_s'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_s', active_tab='single year'), width=5),
                
                dbc.Col(dcc.Markdown('**Fixed color-range**', style={'textAlign':'right'}), width=2),
                dbc.Col(dbc.Checklist(id='cmax4_s', switch=True, value=[], options=[{'label':'', 'value':'On'}], inputStyle={'margin-right':'10px'}))]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a country data point to hide it and rescale the color-coding and marker sizes.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_s', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_s', outline=True, color='dark'),),]),
            dcc.Download(id='data4_s'), dcc.Store(id='click4_s'),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_s'))], label='Sectors', activeTabClassName='fw-bold'),
            
    		]), title='Scatter plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**Select region or country**', style={'textAlign':'right'}), width=3),
    		        dbc.Col(dcc.Dropdown(['Worldwide (average)']+[r+' (average)' for r in regions_list]+countries_list, 'Brazil', id='unit5'), width=4),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv5', outline=True, color='dark')), dcc.Download(id='data5'),]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Area charts'),
    	
    	#dbc.AccordionItem([
    	#	dbc.Row([
    	#	dbc.Col([dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=3),
    	#		    dbc.Col(dcc.Slider(min=1995, max=2019, step=1, value=2016, marks={1995:'1995', 2019:'2019'}, tooltip={'placement':'bottom', 'always_visible':True}, id='year6'), width=7)]),
    	#	dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (country color)**', style={'textAlign':'right'}), width=3),
    	#	        dbc.Col(dmc.SegmentedControl(vulnerability_list, 'oil vulnerability', id='metric6'), width=5),]),
    	#	dbc.Row([dbc.Col(dcc.Markdown('**Select number of links\n per sector group**', style={'textAlign':'right'}), width=3),
    	#		    dbc.Col(dcc.Slider(min=10, max=100, step=10, value=40, marks={10:'10', 100:'100'}, tooltip={'placement':'bottom', 'always_visible':True}, id='nbedges6'),),
        #            dcc.Markdown('Note: All 163 sectors are aggregated into 5 sector groups. Transactions are represented by links whose colours correspond to the exporting sector group, as shown in the colour coding in the image opposite. Please click on a country (node) to display a table below detailing its major imports and exports (links), as links may overlap on the world map figure. Amounts are in millions of USD.', style={'font-size':13}), ]),]),
    	#	dbc.Col(html.Img(src=sector_group_scheme), width=4),]),
    	#	dbc.Row(dcc.Graph(figure={}, id='map6')),
        #    dbc.Row([html.Pre(id='title6_exports'),
        #            dash_table.DataTable(id='table6_exports', css=[{'selector': '.dash-spreadsheet td div', 'rule':'''max-height: 10px'''}], style_data={'whiteSpace':'normal'}, export_format='csv'),
        #            html.Br(), html.Br(),
        #            html.Pre(id='title6_imports'),
        #            dash_table.DataTable(id='table6_imports', css=[{'selector': '.dash-spreadsheet td div', 'rule':'''max-height: 10px'''}], style_data={'whiteSpace':'normal'}, export_format='csv'),]),]
    	#	, title='World map'),
			], flush=True)
			])

@callback([Output('hist1', 'figure'), Output('data1','data'), Output('csv1','n_clicks')], [Input(s, 'value') for s in ['metric1','unit1','group1_s','group1_c','year1','order1','type1','range1']], Input('changes1','active_tab'), Input('csv1', 'n_clicks'))
def update_histogram(metric1,unit1,group1_s,group1_c,year1,order1,type1,range1,changes1,csv1):
    order, color = (order1=='original')*'trace' + (order1=='descending')*'total descending' + (order1=='ascending')*'total ascending', 'sector'*(unit1 in ['country','region']) + 'country'*(unit1=='sector')
    if changes1 == 'range of years': # if looking at variations
        year0, year1 = range1
        #dataset_previous, dataset = data_y[year0][metric1].reset_index(), data_y[year1].reset_index()[units_list+[metric1]]
        dataset = data_y[1995][units_list]
        metric_past = (data_y[year0-2][metric1].reset_index(drop=True) + data_y[year0-1][metric1].reset_index(drop=True) + data_y[year0][metric1].reset_index(drop=True))/3
        metric_future = (data_y[year1-2][metric1].reset_index(drop=True) + data_y[year1-1][metric1].reset_index(drop=True) + data_y[year1][metric1].reset_index(drop=True))/3
        dataset[metric1] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = data_y[year1]
    dataset = dataset[dataset.group.isin(group1_s)&dataset.region.isin(group1_c)].reset_index(drop=True)
    xlabel = 'cumulative '+metric1+' (%)'*(metric1[-13:]=='vulnerability')
    if type1: # if decomp
        decomp = 'country'*(unit1 == 'sector') + 'sector'*(unit1 in ['country','region'])
        fig = px.bar(dataset, x=metric1, y=unit1, orientation='h', color=color, custom_data=[decomp]).update_traces(hovertemplate='<b>%{customdata[0]} (%{y}, '+str(year1)+')</b><br>'+str(metric1)+': %{x:.2f}')
    else:
        fig = px.histogram(dataset, x=metric1, y=unit1, histfunc='avg', orientation='h').update_traces(hovertemplate='<b>%{y}, '+str(year1)+'</b><br>average '+str(metric1)+': %{x:.2f}')
    fig.update_yaxes(categoryorder=order, autorange='reversed').update_layout(xaxis_title=xlabel, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector'), width=1600, plot_bgcolor='white')#, paper_bgcolor='#e7f5ff')# background color inside the plot
    y = str(year1)
    if csv1:
        if changes1 == 'range of years':
            y = str(year0)+'_'+y
        return fig, dcc.send_data_frame(dataset[units_list+[metric1]].to_csv, 'barplot_'+str(metric1)+'_'+str(unit1)+'_'+y+'.csv'), None
    else:
        return fig, None, None

@callback(Output('heatmap2_c', 'figure'), Output('data2_c','data'), Output('csv2_c', 'n_clicks'), [Input(s, 'value') for s in ['metric2_c','region2_c','order2_c','unselect2_c']], Input('csv2_c', 'n_clicks'))
def update_cheatmap(metric2_c,region2_c,order2_c,unselect2_c,csv2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    if metric2_c in country_metrics_list:
        dataset = country_network_dynamics[country_network_dynamics.region.isin(region2_c)&~data_cavg.country.isin(unselect2_c)]
    else:
        dataset = data_cavg[data_cavg.region.isin(region2_c)&~data_cavg.country.isin(unselect2_c)]
    nb_countries = len(dataset.country.unique())
    fig = px.density_heatmap(dataset, x='year', y='country', z=metric2_c, height=70*nb_countries**.63, width=450+260, labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=nb_countries, color_continuous_scale='Turbo').update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':14}, showlegend=True, coloraxis_colorbar={'title':metric2_c, 'orientation':'v', 'len':.8, 'thickness':15}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_c)+': %{z:.2f}')
    if csv2_c:
        return fig, dcc.send_data_frame(dataset.to_csv, 'heatmap_'+str(metric2_c.replace(' ','_'))+'_countries.csv'), None
    else:
        return fig, None, None

@callback(Output('region2_c','value'), Output('unselect2_c','value'), Input('reset2_c','n_clicks'))
def sync_selection_cheatmap(reset):
    if reset:
        return regions_list, []
    else:
        return no_update

@callback(Output('heatmap2_s', 'figure'), Output('data2_s','data'), Output('csv2_s', 'n_clicks'), [Input(s, 'value') for s in ['metric2_s','group2_s', 'order2_s','regions2_s','unselect2_s']], Input('csv2_s', 'n_clicks'))
def update_sheatmap(metric2_s,group2_s,order2_s,regions2_s,unselect2_s,csv2_s):
    order = (order2_s=='original')*'trace' + (order2_s=='descending')*'total descending' + (order2_s=='ascending')*'total ascending'
    if regions2_s == 'Worldwide':
        dataset = data_s1avg[data_s1avg.group.isin(group2_s)&~data_s1avg.sector.isin(unselect2_s)]
    else:
        dataset = data_s2avg[regions2_s][data_s2avg[regions2_s].group.isin(group2_s)&~data_s2avg[regions2_s].sector.isin(unselect2_s)]
    nb_sectors = len(dataset.sector.unique())
    fig = px.density_heatmap(dataset, x='year', y='sector', z=metric2_s, height=45*nb_sectors**.78, width=680+260, labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=nb_sectors, color_continuous_scale='Jet')
    fig.update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':14}, showlegend=True, coloraxis_colorbar={'title':metric2_s, 'orientation':'v', 'len':.8, 'thickness':15}, coloraxis={'colorscale':'Jet'}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_s)+': %{z:.2f}')
    if csv2_s:
        df = pd.DataFrame()
        for r in regions_list:
            df = df._append(dataset[r], ignore_index=True)
        df.insert(1, 'region', pd.Series([r for r in regions_list for idx in range(len(dataset[r]))]))
        return fig, dcc.send_data_frame(df.to_csv, 'heatmap_'+str(metric2_s.replace(' ','_'))+'_sectors_'+str(segment2_s[:-9])+'.csv'), None
    else:
        return fig, None, None

@callback([Output('group2_s','value'), Output('unselect2_s','value')], Input('reset2_s','n_clicks'))
def sync_selection_sheatmap(reset):
    if reset:
        return groups_list, []
    else:
        return no_update

@callback(Output('lines3_c', 'figure'), Output('data3_c','data'), Output('csv3_c', 'n_clicks'), [Input(s, 'value') for s in ['metric3_c','unit3_c','region3_c']], Input('csv3_c', 'n_clicks'))
def update_cline(metric3_c,unit3_c,region3_c,csv3_c):
    if metric3_c in country_metrics_list:
        dataset = country_network_dynamics[country_network_dynamics.region.isin(region3_c)]
    elif unit3_c == 'All sectors (average)':
        dataset = data_cavg[data_cavg.region.isin(region3_c)]
        unit3_c = 'all_sectors'
    else:
        dataset = H[H.region.isin(region3_c)&(H.sector==unit3_c)]
    dataset.country = deepcopy(dataset)['country'].cat.set_categories(countries, ordered=True)
    ylabel = metric3_c+' (%)'*(metric3_c[-13:]=='vulnerability')
    try:
        fig = px.line(dataset, x='year', y=metric3_c, color='country', labels={'x':'year', 'y':ylabel, 'color':'country'}, markers=True, custom_data=['country'])
    except:
        fig = px.line(dataset, x='year', y=metric3_c, color='country', labels={'x':'year', 'y':metric3_c, 'color':'country'}, markers=True, custom_data=['country'])
    fig.update_xaxes(tickvals= np.arange(1995,1994+NY,3)).update_layout(font={'size':14}, height=700, width=1300, hoverlabel={'font_size':14}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_c)+': %{y:.2f}<extra></extra>')
    if csv3_c:
        return fig, dcc.send_data_frame(dataset.to_csv, 'linechart_'+str(unit3_c)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('lines3_s', 'figure'), Output('data3_s','data'), Output('csv3_s','n_clicks'), [Input(s, 'value') for s in ['metric3_s','unit3_s','group3_s']], Input('csv3_s', 'n_clicks'))
def update_sline(metric3_s,unit3_s,group3_s,csv3_s):
    if unit3_s == 'All countries (average)':
        dataset = data_s1avg[data_s1avg.group.isin(group3_s)]
        unit3_s = 'all_countries'
    else:
        dataset = H[H.group.isin(group3_s)&(H.country==unit3_s)]
    ylabel = metric3_s+' (%)'*(metric3_s[-13:]=='vulnerability')
    try:
        fig = px.line(dataset, x='year', y=metric3_s, color='sector', labels={'x':'year', 'y':ylabel, 'color':'sector'}, markers=True, custom_data=['sector'])
    except:
        fig = px.line(dataset, x='year', y=metric3_s, color='sector', labels={'x':'year', 'y':ylabel, 'color':'sector'}, markers=True, custom_data=['sector'])
    fig.update_xaxes(tickvals=np.arange(1995,2023,3)).update_layout(font={'size':14}, height=700, width=1500, hoverlabel={'font_size':14}).update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_s)+': %{y:.2f}<extra></extra>')
    #fig.for_each_trace(lambda t: t.update(name=sector_legend_dict.get(t.name,t.name)))
    if csv3_s:
        return fig, dcc.send_data_frame(dataset.to_csv, 'linechart_'+str(unit3_s)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('scatter4_c', 'figure'), Output('data4_c','data'), Output('csv4_c','n_clicks'), [Input(s, 'value') for s in ['country4_c', 'metric4x_c', 'metric4y_c', 'metric4i_c', 'year4_c', 'range4_c', 'color4_c', 'group4_c', 'filtermetric4_c', 'filternb4_c', 'size4_c', 'symbol4_c', 'cmax4_c']], Input('changes4_c','active_tab'), Input('click4_c','data'), Input('csv4_c', 'n_clicks'))
def update_cscatter(country4_c, metric4x_c, metric4y_c, metric4i_c, year4_c, range4_c, color4_c, group4_c, filtermetric4_c, filternb4_c, size4_c, symbol4_c, cmax4_c, changes4_c, click4_c, csv4_c): # SINGLE COUNTRY select
    global clicked_sectors
    xlabel, ylabel, zlabel = metric4x_c, metric4y_c, metric4i_c
    if symbol4_c:
        symbol = symbol4_c[0]
    else:
        symbol = None
    if 'average country' in country4_c:
        df, custom1 = data_s1avg[data_s1avg.group.isin(group4_c)&(data_s1avg[filtermetric4_c]>=filternb4_c)].reset_index(drop=True), 'average country'
        df[custom1] = custom1
    else:
        df, custom1 = H[H.group.isin(group4_c)&H.country.isin(country4_c)&(H[filtermetric4_c]>=filternb4_c)].reset_index(drop=True), 'country'
    if click4_c:
        df = df[~df.sector.isin(clicked_sectors)]
    custom_data, label_var = ['sector',custom1,zlabel,xlabel,ylabel,filtermetric4_c], int(changes4_c=='range of years')*' variation'
    #dashed_line = int(changes4_c=='single year')
    if changes4_c == 'range of years':
        year0, year1 = range4_c
        dataset = df[df.year==1995]
        metric_past = (df[df.year==year0-2][xlabel].reset_index(drop=True) + df[df.year==year0-1][xlabel].reset_index(drop=True) + df[df.year==year0][xlabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][xlabel].reset_index(drop=True) + df[df.year==year1-1][xlabel].reset_index(drop=True) + df[df.year==year1][xlabel].reset_index(drop=True))/3
        dataset[xlabel] = (metric_future - metric_past).reset_index(drop=True)
        
        metric_past = (df[df.year==year0-2][ylabel].reset_index(drop=True) + df[df.year==year0-1][ylabel].reset_index(drop=True) + df[df.year==year0][ylabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][ylabel].reset_index(drop=True) + df[df.year==year1-1][ylabel].reset_index(drop=True) + df[df.year==year1][ylabel].reset_index(drop=True))/3
        dataset[ylabel] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = df[df.year==year4_c]
    color4_c = (color4_c=='vulnerability')*zlabel + (color4_c=='sector groups')*'group'
    if cmax4_c:
        cmax, cmin = dataset[zlabel].max(), dataset[zlabel].min()
    else:
        cmax, cmin = None, None
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4_c, labels={'x':xlabel, 'y':ylabel}, hover_name=custom1, opacity=.7, color_continuous_scale='Jet', custom_data=custom_data, size_max=size4_c, symbol=symbol)
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=color4_c, labels={'x':xlabel, 'y':ylabel}, hover_name=custom1, opacity=.7, color_continuous_scale='Jet', custom_data=custom_data, size_max=size4_c, symbol=symbol)
    fig.update_layout(width=900, height=650, font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, coloraxis={'cmin':cmin, 'cmax':cmax}, hoverlabel={'font_size':14}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}, clickmode='event+select').update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}', str(filtermetric4_c)+': %{customdata[5]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4)
    if csv4_c:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str(country4_c)+'_'+str(year4_c)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('group4_c','value'), Input('reset4_c','n_clicks'))
def sync_selection_cscatter(reset):
    global clicked_sectors
    if reset:
        clicked_sectors = []
        return groups_list
    else:
        return no_update

@callback(Output('click4_c','data'), Input('scatter4_c', 'clickData'))
def delete_cscatter(clickData):
    global clicked_sectors
    if clickData:
        clicked_sectors.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('scatter4_s', 'figure'), Output('data4_s','data'), Output('csv4_s','n_clicks'), [Input(s, 'value') for s in ['sector4_s', 'metric4x_s','metric4y_s', 'metric4i_s', 'year4_s', 'range4_s', 'color4_s','group4_s','filtermetric4_s', 'filternb4_s', 'size4_s', 'symbol4_s', 'cmax4_s']], Input('changes4_c','active_tab'), Input('click4_s', 'data'), Input('csv4_s', 'n_clicks'))
def update_sscatter(sector4_s, metric4x_s, metric4y_s, metric4i_s, year4_s, range4_s, color4_s, group4_s, filtermetric4_s, filternb4_s, size4_s, symbol4_s, cmax4_s, changes4_s, click4_s, csv4_s):  # SINGLE SECTOR select
    global clicked_countries
    xlabel, ylabel, zlabel = metric4x_s, metric4y_s, metric4i_s
    if symbol4_s:
        symbol = symbol4_s[0]
    else:
        symbol = None
    if 'average sector' in sector4_s:
        df, custom1 = data_cavg[data_cavg[filtermetric4_s]>filternb4_s].reset_index(drop=True), 'average sector'
        df[custom1] = custom1
    else:
        df, custom1 = H[H.region.isin(group4_s)&H.sector.isin(sector4_s)&(H[filtermetric4_s]>filternb4_s)].reset_index(drop=True), 'sector'
    if click4_s:
        df = df[~df.country.isin(clicked_countries)]
    custom_data, label_var = ['country',custom1,zlabel,xlabel,ylabel,filtermetric4_s], int(changes4_s=='range of years')*' variation'
    #dashed_line = int(changes4_s=='single year')
    if changes4_s == 'range of years':
        year0, year1 = range4_s
        dataset = df[df.year==1995]
        metric_past = (df[df.year==year0-2][xlabel].reset_index(drop=True) + df[df.year==year0-1][xlabel].reset_index(drop=True) + df[df.year==year0][xlabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][xlabel].reset_index(drop=True) + df[df.year==year1-1][xlabel].reset_index(drop=True) + df[df.year==year1][xlabel].reset_index(drop=True))/3
        dataset[xlabel] = (metric_future - metric_past).reset_index(drop=True)
        
        metric_past = (df[df.year==year0-2][ylabel].reset_index(drop=True) + df[df.year==year0-1][ylabel].reset_index(drop=True) + df[df.year==year0][ylabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][ylabel].reset_index(drop=True) + df[df.year==year1-1][ylabel].reset_index(drop=True) + df[df.year==year1][ylabel].reset_index(drop=True))/3
        dataset[ylabel] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = df[df.year==year4_s]
    color4_s = (color4_s=='vulnerability')*zlabel + (color4_s=='regions')*'region'
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    if cmax4_s:
        cmax, cmin = dataset[zlabel].max(), dataset[zlabel].min()
    else:
        cmax, cmin = None, None
    try:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_s], labels={'x':xlabel, 'y':ylabel}, hover_name=custom1, opacity=.7,color_continuous_scale='Jet', custom_data=custom_data, size_max=size4_s, symbol=symbol)
    except:
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_s], labels={'x':xlabel, 'y':ylabel}, hover_name=custom1, opacity=.7,color_continuous_scale='Jet', custom_data=custom_data, size_max=size4_s, symbol=symbol)
    fig.update_layout(width=950, height=650, font={'size':14}, hoverlabel={'font_size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}).update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}', str(filtermetric4_s)+': %{customdata[5]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4)
    if csv4_s:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str(sector4_s)+'_'+str(year4_s)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('click4_s', 'data'), Input('scatter4_s', 'clickData'))
def delete_sscatter(clickData):
    global clicked_countries
    if clickData:
        clicked_countries.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('group4_s','value'), Input('reset4_s','n_clicks'))
def sync_selection_sscatter(reset):
    global clicked_countries
    if reset:
        clicked_countries = []
        return regions_list
    else:
        return no_update

@callback(Output('waves5', 'figure'), Output('data5','data'), Output('csv5','n_clicks'), Input('unit5', 'value'), Input('csv5', 'n_clicks'))
def update_waves(unit5, csv5):
    if unit5[:-10] in regions_list+['Worldwide']:
        unit5 = unit5[:-10]
        dataset = data_rvuln[unit5]
    else:
        dataset = data_cvuln[unit5]
    try:
        fig = px.area(dataset, x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'cumulative vulnerability (%)'})
    except:
        fig = px.area(dataset, x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'cumulative vulnerability (%)'})
    fig.update_traces(hovertemplate='%{x}: %{y:.2f}%').update_layout(legend={'font_size':16})
    if csv5:
        return fig, dcc.send_data_frame(dataset.to_csv, 'areachart_'+str(unit5)+'.csv'), None
    else:
        return fig, None, None

#@callback(Output('map6', 'figure'), [Input(s, 'value') for s in ['year6', 'metric6', 'nbedges6']])
#def update_maps(year6, metric6, nbedges6):
#    dataset, colorbar_legend = worldmap_plot[worldmap_plot.year==year6].reset_index(), metric6
#    fig = go.Figure(go.Choropleth(locations=countries_code, z=data_cavg[data_cavg.year==year6][metric6], colorscale='Reds', colorbar={'title': colorbar_legend, 'len':.8, 'thickness':15}, hoverinfo='skip'))
#    for idx in range(nbedges6//10):
#        for s in range(5):
#            batch, color = deepcopy(dataset.iloc[3*(100*s+10*idx): 3*(100*s+10*(idx+1))]), sector_group_colors[s]
#            fig.add_scattergeo(lat=batch.latitude, lon=batch.longitude, mode='lines', line={'width':1.5, 'color':color}, showlegend=False, hoverinfo='skip')
#    fig.add_scattergeo(lat=worldmap_nodes.lat, lon=worldmap_nodes.lon, marker={'size':5, 'symbol':'circle-open', 'color':'black', 'opacity':0.8}, showlegend=False, customdata=worldmap_nodes.EXIOBASE_name, hovertemplate='%{customdata}<extra></extra>').update_layout(clickmode='event+select', width=1400, height=500, margin=dict(l=20, r=20, t=0, b=0))
#    _ = fig.update_geos(lataxis_range=[-55, 90], showocean=True, oceancolor='LightBlue')
#    return fig

#@callback(Output('title6_exports', 'children'), [Input('map6', 'clickData'), Input('year6', 'value')])
#def update_export_title(clickData, year6):
#    if clickData:
#        return 'Largest exports from '+str(clickData['points'][0]['customdata'])+' (source country) in '+str(year6)+'.'

#@callback(Output('table6_exports', 'data'), Input('map6', 'clickData'), [Input(s, 'value') for s in ['year6', 'nbedges6']])
#def update_export_table(clickData, year6, nbedges6):
#    if clickData:
#        dataset = worldmap_table[worldmap_table.year==year6]
#        dataset = dataset.iloc[[100*g+s for g in range(5) for s in range(nbedges6)],:-1]
#        dataset = dataset[dataset['source country']==clickData['points'][0]['customdata']]
#        return dataset.to_dict('records')

#@callback(Output('title6_imports', 'children'), [Input('map6', 'clickData'), Input('year6', 'value')])
#def update_import_title(clickData, year6):
#    if clickData:
#        return 'Largest imports to '+str(clickData['points'][0]['customdata'])+' (target country) in '+str(year6)+'.'

#@callback(Output('table6_imports', 'data'), Input('map6', 'clickData'), [Input(s, 'value') for s in ['year6', 'nbedges6']])
#def update_import_table(clickData, year6, nbedges6):
#    if clickData:
#        dataset = worldmap_table[worldmap_table.year==year6]
#        dataset = dataset.iloc[[100*g+s for g in range(5) for s in range(nbedges6)],:-1]
#        dataset = dataset[dataset['target country']==clickData['points'][0]['customdata']]
#        return dataset.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
