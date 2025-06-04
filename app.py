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

pd.options.mode.chained_assignment = None
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

H = pd.read_csv('processed_data.csv', compression='bz2')

#worldmap_nodes, worldmap_table, worldmap_plot, sector_group_scheme = pd.read_csv('worldmap_nodes.csv'), pd.read_csv('worldmap_table.csv'), pd.read_csv('worldmap_plot.csv'), Image.open('worldmap_scheme.png')
country_network_dynamics = pd.read_csv('country_network_dynamics.csv')

NS, NC, NY, NR, NG = 163, 49, 26, 5, 5
H = H[H.year<=1994+NY]
N = NS*NC
regions, countries, sectors = H.region.iloc[np.arange(0,N,NS)], H.country.unique(), H.sector.unique()
regions_list, countries_list, sectors_list = list(regions.unique()), list(countries), list(sectors)
transition_sectors = ['Mining of iron', 'Mining of copper', 'Mining of nickel', 'Mining of aluminium', 'Minium of precious metals', 'Mining of lead, zinc and tin', 'Mining of other non-ferrous metals', 'Lead, zinc and tin production', 'Quarrying of stone', 'Quarrying of sand and clay', 'Manufacture of rubber and plastic products', 'Manufacture of glass products', 'Re-processing of secondary glass', 'Manufacture of cement, lime and plaster', 'Re-processing of ash into clinker', 'Manufacture of other non-metallic mineral products', 'Manufacture of basic iron, steel and ferro-alloys', 'Re-processing of secondary steel', 'Precious metals production', 'Re-processing of secondary precious metals', 'Aluminium production', 'Re-processing of secondary aluminium', 'Re-processing of secondary lead', 'Copper production', 'Re-processing of secondary copper', 'Other non-ferrous metal production', 'Manufacture of machinery', 'Manufacture of electrical machinery', 'Manufacture of motor vehicles']+['Mining of precious metals', 'Plastics', 'Re-processing of secondary plastic', 'Chemicals', 'Re-processing of secondary other non-ferrous metals', 'Casting of metals', 'Manufacture of fabricated metal products']
circular_sectors = ["Re-processing of secondary wood material", "Re-processing of secondary paper", "Re-processing of secondary glass", "Re-processing of secondary steel", "Re-processing of secondary aluminium", "Re-processing of secondary copper", "Re-processing of secondary lead", "Re-processing of secondary precious metals", "Re-processing of secondary other non-ferrous metals", "Re-processing of secondary plastic", "Re-processing of ash into clinker", "Re-processing of secondary construction material", "Recycling of waste and scrap", "Recycling of bottles by direct reuse", "Biogasification of food waste", "Biogasification of paper", "Composting of food waste", "Composting of paper and wood"]
decarb_sectors = ["Mining of iron","Mining of copper","Mining of nickel","Mining of aluminium","Mining of precious metals","Mining of lead, zinc and tin","Mining of other non-ferrous metals","Quarrying of stone","Quarrying of sand and clay","Lead, zinc and tin production","Manufacture of cement, lime and plaster","Re-processing of ash into clinker","Manufacture of basic iron, steel and ferro-alloys","Aluminium production","Copper production","Other non-ferrous metal production","Precious metals production","Manufacture of fabricated metal products","Manufacture of glass products","Re-processing of secondary steel","Re-processing of secondary aluminium","Re-processing of secondary copper","Re-processing of secondary precious metals","Re-processing of secondary other non-ferrous metals","Re-processing of secondary plastic","Chemicals","Construction"]

sector_to_idx, country_to_idx, k1, k2 = {}, {}, 0, 0
for s in sectors:
    sector_to_idx[s], k1 = k1, k1+1, 

for c in countries:
    country_to_idx[c], k2 = k2, k2+1

units_list = ['region', 'country', 'sector', 'group']
vulnerability_list, importance_list = ['coal vulnerability', 'gas vulnerability', 'oil vulnerability', 'fossil fuel vulnerability'], ['forward linkage', 'backward linkage', 'weighted in-degree', 'weighted out-degree', 'out-degree', 'in-degree', 'betweenness', 'betweenness/in-degree', 'betweenness/out-degree']
metrics_list = vulnerability_list+importance_list
country_metrics_list = ['modularity', 'in-conductance', 'out-conductance']
ordering_list = ['original', 'descending', 'ascending']

H[metrics_list] = H[metrics_list].astype(float)
for s in ['year']+units_list:
    H[s] = H[s].astype('category').cat.set_categories(H[s].unique(), ordered=True)

for s in ['year','region','country']:
    country_network_dynamics[s] = country_network_dynamics[s].astype('category').cat.set_categories(country_network_dynamics[s].unique(), ordered=True)

data_all_cavg = H.groupby(['year','country'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','country'])
data_all_cavg.insert(1, 'region', list(regions)*NY)
data_all_cavg.insert(4, 'group', ['All sectors' for k in range(NC*NY)])

data_groups_cavg = H.groupby(['year','group','country'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','group'])
data_groups_cavg.insert(1, 'region', list(regions)*NY*NG)
data_cavg = pd.concat([data_all_cavg, data_groups_cavg], ignore_index=True)

data_world_savg = H.groupby(['year','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','sector'])
data_world_savg.insert(1, 'region', ['Worldwide' for k in range(NS*NY)])
data_world_savg.insert(4, 'group', list(H.group.iloc[:NS])*NY)
data_regions_savg = H.groupby(['year','region','sector'], observed=False)[metrics_list].mean().reset_index().sort_values(['year','region'])
data_regions_savg.insert(4, 'group', list(H.group.iloc[:NS])*NR*NY)
data_savg = pd.concat([data_world_savg, data_regions_savg], ignore_index=True)

data_cvuln, data_rvuln, year_NYlist, sector_NYlist = {}, {}, np.array([1995+y for k in range(3) for y in range(NY)]), np.array([s for s in ['oil vulnerability','gas vulnerability','coal vulnerability'] for i in range(NY)])
for c in countries:
    data_cvuln[c] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[(data_cavg.group=='All sectors')&(data_cavg.country==c)][['oil vulnerability','gas vulnerability','coal vulnerability']].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_cvuln[c]['year'], data_cvuln[c]['sector'], data_cvuln[c]['vulnerability'] = data_cvuln[c]['year'].astype(int), data_cvuln[c]['sector'].astype(str), data_cvuln[c]['vulnerability'].astype(float)

for r in regions_list:
    data_rvuln[r] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg[(data_cavg.group=='All sectors')&(data_cavg.region==r)].groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
    data_rvuln[r]['year'], data_rvuln[r]['sector'], data_rvuln[r]['vulnerability'] = data_rvuln[r]['year'].astype(int), data_rvuln[r]['sector'].astype(str), data_rvuln[r]['vulnerability'].astype(float)

data_rvuln['Worldwide'] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg.groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
data_rvuln['Worldwide'] = pd.DataFrame(np.vstack((year_NYlist, sector_NYlist, data_cavg.groupby('year', as_index=False, observed=False)[['oil vulnerability','gas vulnerability','coal vulnerability']].sum().iloc[:,1:].unstack().values)).T, columns=['year','sector','vulnerability'])
data_rvuln['Worldwide']['year'], data_rvuln['Worldwide']['sector'], data_rvuln['Worldwide']['vulnerability'] = data_rvuln['Worldwide']['year'].astype(int), data_rvuln['Worldwide']['sector'].astype(str), data_rvuln['Worldwide']['vulnerability'].astype(float)

groups_list, sector_group_colors = ['Agriculture, forestry and fishing', 'Extraction and mining', 'Manufacture and production', 'Utilities', 'Services'], ['lime', 'orange', 'red', 'cyan', 'purple']
data_y = {int(c[0]):c[1] for c in H.groupby('year', observed=False)[units_list+metrics_list]}
clicked_countries1, clicked_countries2, clicked_sectors1, clicked_sectors2 = [], [], [], []

app.layout = html.Div(children=[
    dcc.Markdown('''# Global Economic Vulnerability''', style={'textAlign':'center'}),
    html.Hr(),
    dbc.Accordion([
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric1'), width=3),
                    dbc.Col(dcc.Markdown('**Decompose metric**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dbc.Checklist(id='type1', switch=True, value=['On'], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select unit**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(units_list[:-1], 'country', id='unit1'), width=3),
                    dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order1'), width=5),]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select type of data**', style={'textAlign':'right'}), width=2),
    		        dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2020, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year1'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                            dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range1'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes1', active_tab='single year'), width=5)]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(regions_list, regions_list, id='group1_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv1', n_clicks=0, outline=True, color='dark'), width=3), dcc.Download(id='data1')]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select sector group**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, id='group1_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), )]),
       		dcc.Graph(figure={}, id='hist1')], title='Bar plots'),
    	
    	dbc.AccordionItem(
    		dbc.Tabs([dbc.Tab([
    				dbc.Row([
    					dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list+country_metrics_list, 'coal vulnerability', id='metric2_c'), width=3),
    					dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_c',), width=4)]),
                    dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                        dbc.Col([dcc.Input(id='colorlow2_c', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh2_c', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2)]),
    				dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
    				    dbc.Col(dcc.Checklist(regions_list, regions_list, id='region2_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6), ]),                    
                    dbc.Row([dbc.Col(dcc.Markdown('**Hide countries**', style={'textAlign':'right'}), width=2),
                        dbc.Col(dcc.Dropdown(countries, [], id='unselect2_c', multi=True), width=3),
                        dbc.Col(dbc.Button('Reset selection', id='reset2_c', outline=True, color='dark'), width=2),
                        dbc.Col(dbc.Button('Download spreadsheet', id='csv2_c', outline=True, color='dark')), dcc.Download(id='data2_c'),]),
                    dbc.Row(dcc.Graph(figure={}, id='heatmap2_c')),
    				], label='Countries', activeTabClassName='fw-bold'),
    			
    			dbc.Tab([
                dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dcc.Dropdown(metrics_list, 'coal vulnerability', id='metric2_s'), width=3),
                        dbc.Col(dcc.Markdown('**Select ordering**', style={'textAlign':'right'}), width=2),
    					dbc.Col(dmc.SegmentedControl(ordering_list, 'original', id='order2_s'), width=4)]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                        dbc.Col([dcc.Input(id='colorlow2_s', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh2_s', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2)]),
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
    				dbc.Col(dcc.Dropdown(['All sectors']+list(sectors), 'All sectors (average)', id='unit3_c'), width=4),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select metric**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(metrics_list+country_metrics_list, 'coal vulnerability', id='metric3_c'), width=3),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv3_c', outline=True, color='dark')), dcc.Download(id='data3_c')]),
    			dbc.Row([dbc.Col(dcc.Markdown('**Select region**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Checklist(regions_list, regions_list, id='region3_c', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=6)]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_c')),],
	    		label='Countries', activeTabClassName='fw-bold'),
    			dbc.Tab([
    			dbc.Row([dbc.Col(dcc.Markdown('**Select countries**', style={'textAlign':'right'}), width=2),
    				dbc.Col(dcc.Dropdown(['Worldwide']+regions_list+list(countries), 'All countries (average)', id='unit3_s'), width=3),]),
	    		dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
                        dbc.Col(dcc.Dropdown(metrics_list, 'fossil fuel vulnerability', id='metric3_s'), width=3),
                        dbc.Col(dbc.Button('Download spreadsheet', id='csv3_s', outline=True, color='dark')), dcc.Download(id='data3_s')]),
                dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, id='group3_s', inline=True, inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), ) ]),
    			dbc.Row(dcc.Graph(figure={}, id='lines3_s')),],
    			label='Sectors', activeTabClassName='fw-bold')])
    		
    		, title='Line chart'),
    	
    	dbc.AccordionItem(
    	    
    	    dbc.Tabs([
    	    
            dbc.Tab([
			dbc.Row([dbc.Col(dcc.Markdown('**Select region**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(['Worldwide']+regions_list, ['Worldwide'], id='region4_r', multi=True), width=2)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_r'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4x_r', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='xlow4_r', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='xhigh4_r', type='number', debounce=True, placeholder='max')], width=3),
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_r'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4y_r', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='ylow4_r', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='yhigh4_r', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			
            dbc.Row([dbc.Col(dcc.Markdown('**Select marker color**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(['vulnerability', 'regions'], 'vulnerability', id='color4_r'), width=4)]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (size)**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dmc.SegmentedControl(vulnerability_list, 'fossil fuel vulnerability', id='metric4i_r'), width=4)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                    dbc.Col([dcc.Input(id='colorlow4_r', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh4_r', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select sector group**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, inline=True, id='group4_r', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), )]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Restrict sectors to**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(['energy transition', 'decarbonization', 'circular economy', 'no restriction'], 'no restriction', id='restrict4_r'))]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2020, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_r'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                    dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_r'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_r', active_tab='single year'), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a sector data point to hide it.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_r', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_r', outline=True, color='dark')),
                            dcc.Store(id='click4_r'), dcc.Download(id='data4_r')]),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_r')),], label='Regions', activeTabClassName='fw-bold'),
            
    	    dbc.Tab([
			dbc.Row([dbc.Col(dcc.Markdown('**Select country**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(countries_list, ['United States'], id='country4_c', multi=True), width=2)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_c'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4x_c', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='xlow4_c', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='xhigh4_c', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_c'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4y_c', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='ylow4_c', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='yhigh4_c', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			
            dbc.Row([dbc.Col(dcc.Markdown('**Select marker color**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(['vulnerability', 'regions'], 'vulnerability', id='color4_c'), width=4)]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (size)**', style={'textAlign':'right'}), width=2),
			        dbc.Col(dmc.SegmentedControl(vulnerability_list, 'fossil fuel vulnerability', id='metric4i_c'), width=4)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                    dbc.Col([dcc.Input(id='colorlow4_c', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh4_c', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select sector group**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Checklist(groups_list, groups_list, inline=True, id='group4_c', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), )]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Restrict sectors to**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(['energy transition', 'decarbonization', 'circular economy', 'no restriction'], 'no restriction', id='restrict4_c'))]),
            
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2020, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_c'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                    dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_c'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_c', active_tab='single year'), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a sector data point to hide it.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_c', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_c', outline=True, color='dark')),
                            dcc.Store(id='click4_c'), dcc.Download(id='data4_c')]),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_c')),], label='Countries', activeTabClassName='fw-bold'),
            
            dbc.Tab([
    	    dbc.Row([dbc.Col(dcc.Markdown('**Select sector groups**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(['All sectors']+groups_list, ['All sectors'], id='group4_g', multi=True), width=4),
                    ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_g'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4x_g', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='xlow4_g', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='xhigh4_g', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_g'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4y_g', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='ylow4_g', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='yhigh4_g', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (size)**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(vulnerability_list, 'fossil fuel vulnerability', id='metric4i_g'), width=4)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                    dbc.Col([dcc.Input(id='colorlow4_g', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh4_g', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2),]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Checklist(regions_list, regions_list, inline=True, id='region4_g', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2020, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_g'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                    dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_g'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_g', active_tab='single year'), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a country data point to hide it.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_g', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_g', outline=True, color='dark')),]),
            dcc.Download(id='data4_g'), dcc.Store(id='click4_g'),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_g'))], label='Sector groups', activeTabClassName='fw-bold'),
            
            dbc.Tab([
    	    dbc.Row([dbc.Col(dcc.Markdown('**Select sector**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(sectors_list, ['Mining of iron'], id='sector4_s', multi=True), width=4),
                    ]),
    		dbc.Row([dbc.Col(dcc.Markdown('**Select x-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'forward linkage', id='metric4x_s'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4x_s', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='xlow4_s', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='xhigh4_s', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
			dbc.Row([dbc.Col(dcc.Markdown('**Select y-axis**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dcc.Dropdown(importance_list, 'backward linkage', id='metric4y_s'), width=2),
                    dbc.Col(dcc.Markdown('**log-scaling**', style={'textAlign':'right'}), width=1),
                    dbc.Col(dbc.Checklist(id='scale4y_s', switch=True, value=[], options=[{'label':'', 'value':'On'}],
                                                                                inputStyle={'margin-right':'10px'}), width=1),
                    dbc.Col(dcc.Markdown('**boundaries**', style={'textAlign':'right'}), width=1),
                    dbc.Col([dcc.Input(id='ylow4_s', type='number', debounce=True, placeholder='min'),
                            dcc.Input(id='yhigh4_s', type='number', debounce=True, placeholder='max')], width=3)
                    ]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select vulnerability (size)**', style={'textAlign':'right'}), width=2),
                    dbc.Col(dmc.SegmentedControl(vulnerability_list, 'fossil fuel vulnerability', id='metric4i_s'), width=4)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select color boundaries**', style={'textAlign':'right'}), width=2),
                    dbc.Col([dcc.Input(id='colorlow4_s', type='number', min=0, max=100, step=0.5, debounce=True, placeholder='min'),
                            dcc.Input(id='colorhigh4_s', type='number', min=1, max=100, step=0.5, debounce=True, placeholder='max')], width=2),]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select regions**', style={'textAlign':'right'}), width=2),
                dbc.Col(dcc.Checklist(regions_list, regions_list, inline=True, id='group4_s', inputStyle={'margin-top':'10px', 'margin-right':'5px', 'margin-left':'30px'}), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('**Select year**', style={'textAlign':'right'}), width=2),
                dbc.Col(dbc.Tabs([dbc.Tab(dcc.Slider(min=1995, max=1994+NY, step=1, value=2020, marks={1995:'1995', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='year4_s'), label='single year', tab_id='single year', activeTabClassName='fw-bold'),
                    dbc.Tab(dcc.RangeSlider(min=1997, max=1994+NY, step=1, value=[2000,1994+NY], marks={1997:'1997', 1994+NY:str(1994+NY)}, tooltip={'placement':'bottom', 'always_visible':True}, id='range4_s'), label='range of years', tab_id='range of years', activeTabClassName='fw-bold'),], id='changes4_s', active_tab='single year'), width=5)]),
            dbc.Row([dbc.Col(dcc.Markdown('Note: Please click on a country data point to hide it.'), width=7),
                    dbc.Col(dbc.Button('Reset selection', id='reset4_s', outline=True, color='dark'), width=2),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv4_s', outline=True, color='dark')),]),
            dcc.Download(id='data4_s'), dcc.Store(id='click4_s'),
            dbc.Row(dcc.Graph(figure={}, id='scatter4_s'))], label='Sectors', activeTabClassName='fw-bold'),
            
    		]), title='Scatter plots'),
    	
    	dbc.AccordionItem([
    		dbc.Row([dbc.Col(dcc.Markdown('**Select region or country**', style={'textAlign':'right'}), width=3),
    		        dbc.Col(dcc.Dropdown(['Worldwide']+regions_list+countries_list, 'Brazil', id='unit5'), width=4),
                    dbc.Col(dbc.Button('Download spreadsheet', id='csv5', outline=True, color='dark')), dcc.Download(id='data5'),]),
    		dbc.Row(dcc.Graph(figure={}, id='waves5'))]
    		, title='Area charts'),
			], flush=True)
			])

@callback([Output('hist1', 'figure'), Output('data1','data'), Output('csv1','n_clicks')], [Input(s, 'value') for s in ['metric1','unit1','group1_s','group1_c','year1','order1','type1','range1']], Input('changes1','active_tab'), Input('csv1', 'n_clicks'))
def update_histogram(metric1,unit1,group1_s,group1_c,year1,order1,type1,range1,changes1,csv1):
    order, color = (order1=='original')*'trace' + (order1=='descending')*'total descending' + (order1=='ascending')*'total ascending', 'sector'*(unit1 in ['country','region']) + 'country'*(unit1=='sector')
    if changes1 == 'range of years': # if looking at variations
        year0, year1 = range1
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
        fig = px.histogram(dataset, x=metric1, y=unit1, histfunc='avg', orientation='h').update_traces(hovertemplate='<b>%{y}, '+str(year1)+'</b><br>average '+str(metric1)+': %{x:.2f}<extra></extra>')
    fig.update_yaxes(categoryorder=order, autorange='reversed').update_xaxes(showgrid=True, gridcolor='#dbe9f2').update_layout(xaxis_title=xlabel, yaxis_title=unit1, font={'size':11}, height=820+1450*(unit1=='sector'), width=1600, plot_bgcolor='white')
    y = str(year1)
    if csv1:
        if changes1 == 'range of years':
            y = str(year0)+'_'+y
        return fig, dcc.send_data_frame(dataset[units_list+[metric1]].to_csv, 'barplot_'+str(metric1)+'_'+str(unit1)+'_'+y+'.csv'), None
    else:
        return fig, None, None

@callback(Output('heatmap2_c', 'figure'), Output('data2_c','data'), Output('csv2_c', 'n_clicks'), [Input(s, 'value') for s in ['metric2_c', 'colorlow2_c', 'colorhigh2_c', 'region2_c','order2_c','unselect2_c']], Input('csv2_c', 'n_clicks'))
def update_cheatmap(metric2_c,colorlow2_c, colorhigh2_c, region2_c,order2_c,unselect2_c,csv2_c):
    order = (order2_c=='original')*'trace' + (order2_c=='descending')*'total descending' + (order2_c=='ascending')*'total ascending'
    if metric2_c in country_metrics_list:
        dataset = country_network_dynamics[country_network_dynamics.region.isin(region2_c)&~data_cavg.country.isin(unselect2_c)]
    else:
        dataset = data_cavg[(data_cavg.group=='All sectors')&data_cavg.region.isin(region2_c)&~data_cavg.country.isin(unselect2_c)]
    if colorlow2_c == None:
        colorlow2_c = 0
    if colorhigh2_c == None:
        colorhigh2_c = dataset[metric2_c].max()
    nb_countries = len(dataset.country.unique())
    fig = px.density_heatmap(dataset, x='year', y='country', z=metric2_c, height=70*nb_countries**.63, width=450+260, labels={'x':'year','y':'country'}, nbinsx=NY, nbinsy=nb_countries, color_continuous_scale='Turbo', range_color=[colorlow2_c, colorhigh2_c]).update_xaxes(dtick=3, ticklen=10, tickwidth=3, ticks='outside').update_yaxes(tickmode='linear', ticklen=7, tickwidth=2, ticks='outside', autorange='reversed', categoryorder=order).update_layout(font={'size':14}, showlegend=True, coloraxis_colorbar={'title':metric2_c, 'orientation':'v', 'len':.8, 'thickness':15}).update_traces(hovertemplate='<b>%{y}</b><br>year: %{x}<br>'+str(metric2_c)+': %{z:.2f}')
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

@callback(Output('heatmap2_s', 'figure'), Output('data2_s','data'), Output('csv2_s', 'n_clicks'), [Input(s, 'value') for s in ['metric2_s', 'colorlow2_s', 'colorhigh2_s', 'group2_s', 'order2_s','regions2_s','unselect2_s']], Input('csv2_s', 'n_clicks'))
def update_sheatmap(metric2_s, colorlow2_s, colorhigh2_s, group2_s,order2_s,regions2_s,unselect2_s,csv2_s):
    order = (order2_s=='original')*'trace' + (order2_s=='descending')*'total descending' + (order2_s=='ascending')*'total ascending'
    #if regions2_s == 'Worldwide':
    #    dataset = data_s1avg[data_s1avg.group.isin(group2_s)&~data_savg.sector.isin(unselect2_s)&data_savg.region==]
    #else:
    #    dataset = data_s2avg[regions2_s][data_s2avg[regions2_s].group.isin(group2_s)&~data_s2avg[regions2_s].sector.isin(unselect2_s)]
    dataset = data_savg[(data_savg.region==regions2_s)&data_savg.group.isin(group2_s)&~data_savg.sector.isin(unselect2_s)]
    nb_sectors = len(dataset.sector.unique())
    if colorlow2_s == None:
        colorlow2_s = 0
    if colorhigh2_s == None:
        colorhigh2_s = dataset[metric2_s].max()
    fig = px.density_heatmap(dataset, x='year', y='sector', z=metric2_s, height=45*nb_sectors**.78, width=680+260, labels={'x':'year','y':'sector'}, nbinsx=NY, nbinsy=nb_sectors, color_continuous_scale='Jet', range_color=[colorlow2_s, colorhigh2_s])
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
    elif unit3_c == 'All sectors':
        dataset = data_cavg[(data_cavg.group=='All sectors')&data_cavg.region.isin(region3_c)]
    elif unit3_c in groups_list:
        dataset = data_cavg[(data_cavg.group==unit3_c)&data_cavg.region.isin(region3_c)]
    else:
        dataset = H[H.region.isin(region3_c)&(H.sector==unit3_c)]
    dataset.country = deepcopy(dataset)['country'].cat.set_categories(countries, ordered=True)
    ylabel = metric3_c+' (%)'*(metric3_c[-13:]=='vulnerability')
    fig = px.line(dataset, x='year', y=metric3_c, color='country', labels={'x':'year', 'y':metric3_c, 'color':'country'}, markers=True, custom_data=['country'])
    fig.update_xaxes(tickvals= np.arange(1995,1994+NY,3), showgrid=True, gridcolor='#dbe9f2').update_yaxes(showgrid=True, gridcolor='#dbe9f2').update_layout(font={'size':14}, height=700, width=1300, hoverlabel={'font_size':14}, plot_bgcolor='white').update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_c)+': %{y:.2f}<extra></extra>')
    if csv3_c:
        return fig, dcc.send_data_frame(dataset.to_csv, 'linechart_'+str(unit3_c)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('lines3_s', 'figure'), Output('data3_s','data'), Output('csv3_s','n_clicks'), [Input(s, 'value') for s in ['metric3_s','unit3_s','group3_s']], Input('csv3_s', 'n_clicks'))
def update_sline(metric3_s,unit3_s,group3_s,csv3_s):
    if unit3_s not in countries:
        dataset = data_savg[data_savg.group.isin(group3_s) & data_savg.region==unit3_s]
    else:
        dataset = H[H.group.isin(group3_s)&(H.country==unit3_s)]
    
    ylabel = metric3_s+' (%)'*(metric3_s[-13:]=='vulnerability')
    fig = px.line(dataset, x='year', y=metric3_s, color='sector', labels={'x':'year', 'y':ylabel, 'color':'sector'}, markers=True, custom_data=['sector'])
    fig.update_xaxes(tickvals=np.arange(1995,2023,3), showgrid=True, gridcolor='#dbe9f2').update_yaxes(showgrid=True, gridcolor='#dbe9f2').update_layout(font={'size':14}, height=700, width=1500, hoverlabel={'font_size':14}, plot_bgcolor='white').update_traces(line={'width':4}, marker={'size':10}, hovertemplate='<b>%{customdata[0]}</b><br>year: %{x}<br>'+str(metric3_s)+': %{y:.2f}<extra></extra>')
    #fig.for_each_trace(lambda t: t.update(name=sector_legend_dict.get(t.name,t.name)))
    if csv3_s:
        return fig, dcc.send_data_frame(dataset.to_csv, 'linechart_'+str(unit3_s)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('scatter4_r', 'figure'), Output('data4_r','data'), Output('csv4_r','n_clicks'), [Input(s, 'value') for s in ['region4_r', 'metric4x_r', 'scale4x_r', 'xlow4_r', 'xhigh4_r', 'metric4y_r', 'ylow4_r', 'yhigh4_r', 'scale4y_r', 'color4_r', 'metric4i_r', 'restrict4_r', 'year4_r', 'range4_r', 'group4_r', 'colorlow4_r', 'colorhigh4_r']], Input('changes4_r','active_tab'), Input('click4_r','data'), Input('csv4_r', 'n_clicks'))
def update_rscatter(region4_r, metric4x_r, scale4x_r, xlow4_r, xhigh4_r, metric4y_r, scale4y_r, ylow4_r, yhigh4_r, color4_r, metric4i_r, restrict4_r, year4_r, range4_r, group4_r, colorlow4_r, colorhigh4_r, changes4_r, click4_r, csv4_r): # SINGLE REGION select
    global clicked_sectors2
    xlabel, ylabel, zlabel = metric4x_r, metric4y_r, metric4i_r
    xscale, yscale = 'log' if scale4x_r else 'linear', 'log' if scale4y_r else 'linear'
    df = data_savg[data_savg.group.isin(group4_r) & data_savg.region.isin(region4_r)].reset_index(drop=True)    
    if restrict4_r == 'energy transition':
        df = df[df.sector.isin(transition_sectors)]
    elif restrict4_r == 'decarbonization':
        df = df[df.sector.isin(decarb_sectors)]
    elif restrict4_r == 'circular economy':
        df = df[df.sector.isin(circular_sectors)]
    if click4_r:
        df = df[~df.sector.isin(clicked_sectors2)]
    custom_data = ['sector','region',zlabel,xlabel,ylabel]
    if changes4_r == 'range of years':
        year0, year1 = range4_r
        dataset = df[df.year==year0]
        metric_past = (df[df.year==year0-2][xlabel].reset_index(drop=True) + df[df.year==year0-1][xlabel].reset_index(drop=True) + df[df.year==year0][xlabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][xlabel].reset_index(drop=True) + df[df.year==year1-1][xlabel].reset_index(drop=True) + df[df.year==year1][xlabel].reset_index(drop=True))/3
        dataset[xlabel] = (metric_future - metric_past).reset_index(drop=True)
        metric_past = (df[df.year==year0-2][ylabel].reset_index(drop=True) + df[df.year==year0-1][ylabel].reset_index(drop=True) + df[df.year==year0][ylabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][ylabel].reset_index(drop=True) + df[df.year==year1-1][ylabel].reset_index(drop=True) + df[df.year==year1][ylabel].reset_index(drop=True))/3
        dataset[ylabel] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = df[df.year==year4_r]
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    if colorlow4_r == None:
        colorlow4_r = 0
    if colorhigh4_r == None:
        colorhigh4_r = dataset[zlabel].max()
    color4_r = (color4_r=='vulnerability')*zlabel + (color4_r=='regions')*'region'
    #if color4_r == 'vulnerability':
    #    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data)
    #elif color4_r == 'regions':
    #    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color='region', labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_discrete_sequence=px.colors.qualitative.Bold, custom_data=custom_data)
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_r], labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_r, colorhigh4_r])
    #color4_r = (color4_r=='vulnerability')*zlabel + (color4_r=='regions')*'region'
    if color4_r == 'vulnerability':
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_r, colorhigh4_r])
    elif color4_r == 'regions':
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color='region', labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_discrete_sequence=px.colors.qualitative.Bold, custom_data=custom_data)
    #fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_r], labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_r, colorhigh4_r])
    fig.update_layout(width=900, height=650, font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, hoverlabel={'font_size':14}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}, clickmode='event+select', plot_bgcolor='white').update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4).update_xaxes(showgrid=True, gridcolor='#dbe9f2', type=xscale).update_yaxes(showgrid=True, gridcolor='#dbe9f2', type=yscale)
    if csv4_r:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str('_'.join(c.replace(' ','') for r in region4_r)+str(year4_r)+'.csv')), None
    else:
        return fig, None, None

@callback(Output('group4_r','value'), Input('reset4_r','n_clicks'))
def sync_selection_rscatter(reset):
    global clicked_sectors2
    if reset:
        clicked_sectors2 = []
        return groups_list
    else:
        return no_update

@callback(Output('click4_r','data'), Input('scatter4_r', 'clickData'))
def delete_rscatter(clickData):
    global clicked_sectors2
    if clickData:
        clicked_sectors2.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('scatter4_c', 'figure'), Output('data4_c','data'), Output('csv4_c','n_clicks'), [Input(s, 'value') for s in ['country4_c', 'metric4x_c', 'scale4x_c', 'xlow4_c', 'xhigh4_c', 'metric4y_c', 'scale4y_c', 'ylow4_c', 'yhigh4_c', 'color4_c', 'metric4i_c', 'restrict4_c', 'year4_c', 'range4_c', 'group4_c', 'colorlow4_c', 'colorhigh4_c']], Input('changes4_c','active_tab'), Input('click4_c','data'), Input('csv4_c', 'n_clicks'))
def update_cscatter(country4_c, metric4x_c, scale4x_c, xlow4_c, xhigh4_c, metric4y_c, scale4y_c, ylow4_c, yhigh4_c, color4_c, metric4i_c, restrict4_c, year4_c, range4_c, group4_c, colorlow4_c, colorhigh4_c, changes4_c, click4_c, csv4_c): # SINGLE COUNTRY select
    global clicked_sectors1
    xlabel, ylabel, zlabel = metric4x_c, metric4y_c, metric4i_c
    xscale, yscale = 'log' if scale4x_c else 'linear', 'log' if scale4y_c else 'linear'
    df = H[H.group.isin(group4_c) & H.country.isin(country4_c)].reset_index(drop=True)
    if restrict4_c == 'energy transition':
        df = df[df.sector.isin(transition_sectors)]
    elif restrict4_c == 'decarbonization':
        df = df[df.sector.isin(decarb_sectors)]
    elif restrict4_c == 'circular economy':
        df = df[df.sector.isin(circular_sectors)]
    if click4_c:
        df = df[~df.sector.isin(clicked_sectors1)]
    custom_data = ['sector','country',zlabel,xlabel,ylabel]
    if changes4_c == 'range of years':
        year0, year1 = range4_c
        dataset = df[df.year==year0]
        metric_past = (df[df.year==year0-2][xlabel].reset_index(drop=True) + df[df.year==year0-1][xlabel].reset_index(drop=True) + df[df.year==year0][xlabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][xlabel].reset_index(drop=True) + df[df.year==year1-1][xlabel].reset_index(drop=True) + df[df.year==year1][xlabel].reset_index(drop=True))/3
        dataset[xlabel] = (metric_future - metric_past).reset_index(drop=True)
        metric_past = (df[df.year==year0-2][ylabel].reset_index(drop=True) + df[df.year==year0-1][ylabel].reset_index(drop=True) + df[df.year==year0][ylabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][ylabel].reset_index(drop=True) + df[df.year==year1-1][ylabel].reset_index(drop=True) + df[df.year==year1][ylabel].reset_index(drop=True))/3
        dataset[ylabel] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = df[df.year==year4_c]
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    if colorlow4_c == None:
        colorlow4_c = 0
    if colorhigh4_c == None:
        colorhigh4_c = dataset[zlabel].max()
    color4_c = (color4_c=='vulnerability')*zlabel + (color4_c=='regions')*'region'
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_c], labels={'x':xlabel, 'y':ylabel}, hover_name='country', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_c, colorhigh4_c])
    #color4_c = (color4_c=='vulnerability')*zlabel + (color4_c=='regions')*'region'
    if color4_c == 'vulnerability':
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_r, colorhigh4_r])
    elif color4_c == 'regions':
        fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color='region', labels={'x':xlabel, 'y':ylabel}, hover_name='region', opacity=.6, color_discrete_sequence=px.colors.qualitative.Bold, custom_data=custom_data)
    #fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=dataset[color4_c], labels={'x':xlabel, 'y':ylabel}, hover_name='country', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_c, colorhigh4_c])
    fig.update_layout(width=900, height=650, font={'size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, hoverlabel={'font_size':14}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}, clickmode='event+select', plot_bgcolor='white').update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4).update_xaxes(showgrid=True, gridcolor='#dbe9f2', type=xscale).update_yaxes(showgrid=True, gridcolor='#dbe9f2', type=yscale)
    if csv4_c:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str('_'.join(c.replace(' ','') for c in country4_c)+'_')+str(year4_c)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('group4_c','value'), Input('reset4_c','n_clicks'))
def sync_selection_cscatter(reset):
    global clicked_sectors1
    if reset:
        clicked_sectors1 = []
        return groups_list
    else:
        return no_update

@callback(Output('click4_c','data'), Input('scatter4_c', 'clickData'))
def delete_cscatter(clickData):
    global clicked_sectors1
    if clickData:
        clicked_sectors1.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('scatter4_g', 'figure'), Output('data4_g','data'), Output('csv4_g','n_clicks'), [Input(s, 'value') for s in ['group4_g', 'metric4x_g', 'scale4x_g', 'xlow4_g', 'xhigh4_g', 'metric4y_g', 'scale4y_g', 'ylow4_g', 'yhigh4_g', 'metric4i_g', 'year4_g', 'range4_g','region4_g', 'colorlow4_g', 'colorhigh4_g']], Input('changes4_c','active_tab'), Input('click4_g', 'data'), Input('csv4_g', 'n_clicks'))
def update_gscatter(group4_g, metric4x_g, scale4x_g, xlow4_g, xhigh4_g, metric4y_g, scale4y_g, ylow4_g, yhigh4_g, metric4i_g, year4_g, range4_g, region4_g, colorlow4_g, colorhigh4_g, changes4_g, click4_g, csv4_g):  # SINGLE GROUP select
    global clicked_countries2
    xlabel, ylabel, zlabel = metric4x_g, metric4y_g, metric4i_g
    xscale, yscale = 'log' if scale4x_g else 'linear', 'log' if scale4y_g else 'linear'
    df = data_cavg[data_cavg.region.isin(region4_g)&data_cavg.group.isin(group4_g)].reset_index(drop=True)
    if click4_g:
        df = df[~df.country.isin(clicked_countries2)]
    custom_data = ['country','group', zlabel,xlabel,ylabel]
    if changes4_g == 'range of years':
        year0, year1 = range4_g
        dataset = df[df.year==1995]
        metric_past = (df[df.year==year0-2][xlabel].reset_index(drop=True) + df[df.year==year0-1][xlabel].reset_index(drop=True) + df[df.year==year0][xlabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][xlabel].reset_index(drop=True) + df[df.year==year1-1][xlabel].reset_index(drop=True) + df[df.year==year1][xlabel].reset_index(drop=True))/3
        dataset[xlabel] = (metric_future - metric_past).reset_index(drop=True)
        metric_past = (df[df.year==year0-2][ylabel].reset_index(drop=True) + df[df.year==year0-1][ylabel].reset_index(drop=True) + df[df.year==year0][ylabel].reset_index(drop=True))/3
        metric_future = (df[df.year==year1-2][ylabel].reset_index(drop=True) + df[df.year==year1-1][ylabel].reset_index(drop=True) + df[df.year==year1][ylabel].reset_index(drop=True))/3
        dataset[ylabel] = (metric_future - metric_past).reset_index(drop=True)
    else:
        dataset = df[df.year==year4_g]
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    if colorlow4_g == None:
        colorlow4_g = 0
    if colorlow4_g == None:
        colorlow4_g = dataset[zlabel].max()
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, hover_name='group', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_g, colorhigh4_g])
    fig.update_layout(width=950, height=650, font={'size':14}, hoverlabel={'font_size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}, plot_bgcolor='white').update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4).update_xaxes(showgrid=True, gridcolor='#dbe9f2', type=xscale).update_yaxes(showgrid=True, gridcolor='#dbe9f2', type=yscale)
    if csv4_g:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str('_'.join(c.replace(' ','') for s in group4_g)+'_')+'_'+str(year4_g)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('click4_g', 'data'), Input('scatter4_g', 'clickData'))
def delete_gscatter(clickData):
    global clicked_countries2
    if clickData:
        clicked_countries2.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('group4_g','value'), Input('reset4_g','n_clicks'))
def sync_selection_gscatter(reset):
    global clicked_countries2
    if reset:
        clicked_countries2 = []
        return regions_list
    else:
        return no_update

@callback(Output('scatter4_s', 'figure'), Output('data4_s','data'), Output('csv4_s','n_clicks'), [Input(s, 'value') for s in ['sector4_s', 'metric4x_s','scale4x_s', 'xlow4_s', 'xhigh4_s', 'metric4y_s','scale4y_s', 'ylow4_s', 'yhigh4_s', 'metric4i_s', 'year4_s', 'range4_s','group4_s', 'colorlow4_s', 'colorhigh4_s']], Input('changes4_c','active_tab'), Input('click4_s', 'data'), Input('csv4_s', 'n_clicks'))
def update_sscatter(sector4_s, metric4x_s, scale4x_s, xlow4_s, xhigh4_s, metric4y_s, scale4y_s, ylow4_s, yhigh4_s, metric4i_s, year4_s, range4_s, group4_s, colorlow4_s, colorhigh4_s, changes4_s, click4_s, csv4_s):  # SINGLE SECTOR select
    global clicked_countries1
    xlabel, ylabel, zlabel = metric4x_s, metric4y_s, metric4i_s
    xscale, yscale = 'log' if scale4x_s else 'linear', 'log' if scale4y_s else 'linear'
    df = H[H.region.isin(group4_s)&H.sector.isin(sector4_s)].reset_index(drop=True)
    if click4_s:
        df = df[~df.country.isin(clicked_countries1)]
    custom_data = ['country','sector',zlabel,xlabel,ylabel]
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
    dashed_xline, dashed_yline = dataset[xlabel].median(), dataset[ylabel].median()
    if colorlow4_s == None:
        colorlow4_s = 0
    if colorlow4_s == None:
        colorlow4_s = dataset[zlabel].max()
    fig = px.scatter(dataset, x=xlabel, y=ylabel, size=zlabel, color=zlabel, labels={'x':xlabel, 'y':ylabel}, hover_name='sector', opacity=.6, color_continuous_scale='Jet', custom_data=custom_data, range_color=[colorlow4_s, colorhigh4_s])
    fig.update_layout(width=950, height=650, font={'size':14}, hoverlabel={'font_size':14}, coloraxis_colorbar={'title':'vulnerability (%)', 'orientation':'v', 'len':.8, 'thickness':15}, legend={'orientation':'h', 'yanchor':'bottom', 'y':1.02, 'entrywidth':200, 'title':None}, plot_bgcolor='white').update_traces(hovertemplate='<br>'.join(['<b>%{customdata[0]} (%{customdata[1]})</b>', str(zlabel)+': %{customdata[2]:.2f}%', str(xlabel)+': %{customdata[3]:.2f}', str(ylabel)+': %{customdata[4]:.2f}<extra></extra>'])).add_hline(y=dashed_yline, line_width=3, line_dash='dash', line_color='red', opacity=.4).add_vline(x=dashed_xline, line_width=3, line_dash='dash', line_color='red', opacity=.4).update_xaxes(showgrid=True, gridcolor='#dbe9f2', type=xscale).update_yaxes(showgrid=True, gridcolor='#dbe9f2', type=yscale)
    if csv4_s:
        return fig, dcc.send_data_frame(dataset.to_csv, 'scatterplot_'+str('_'.join(c.replace(' ','') for s in sector4_s)+'_')+'_'+str(year4_s)+'.csv'), None
    else:
        return fig, None, None

@callback(Output('click4_s', 'data'), Input('scatter4_s', 'clickData'))
def delete_sscatter(clickData):
    global clicked_countries1
    if clickData:
        clicked_countries1.append(clickData['points'][0]['customdata'][0])
        return clickData
    else:
        return no_update

@callback(Output('group4_s','value'), Input('reset4_s','n_clicks'))
def sync_selection_sscatter(reset):
    global clicked_countries1
    if reset:
        clicked_countries1 = []
        return regions_list
    else:
        return no_update

@callback(Output('waves5', 'figure'), Output('data5','data'), Output('csv5','n_clicks'), Input('unit5', 'value'), Input('csv5', 'n_clicks'))
def update_waves(unit5, csv5):
    if unit5 not in countries_list:
        dataset = data_rvuln[unit5]
    else:
        dataset = data_cvuln[unit5]
    fig = px.area(dataset, x='year', y='vulnerability', color='sector', width=950, height=450, labels={'y':'cumulative vulnerability (%)'})
    fig.update_traces(hovertemplate='%{x}: %{y:.2f}%').update_layout(legend={'font_size':16}, plot_bgcolor='white').update_xaxes(showgrid=True, gridcolor='#dbe9f2').update_yaxes(showgrid=True, gridcolor='#dbe9f2')
    if csv5:
        return fig, dcc.send_data_frame(dataset.to_csv, 'areachart_'+str(unit5)+'.csv'), None
    else:
        return fig, None, None

if __name__ == '__main__':
    app.run(debug=True, port=8051)
