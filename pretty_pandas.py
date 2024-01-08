from collections import defaultdict
import dataframe_image
import seaborn as sns
import pandas as pd
import numpy as np
import json

from scipy import stats
from colors import colors_rgb, colors_hex, palettes_rgb, palettes_hex


class PrettyPandas():
    
    def __init__(self) -> None:
        self.COLORS_RGB = colors_rgb
        self.COLORS_HEX = colors_hex
        # use our helpers to populate hex code dict (we will want to speak generally in hex for this work)
        for k in self.COLORS_RGB.keys():
            if k not in self.COLORS_HEX.keys():
                self.COLORS_HEX[k] = self.make_hex_color(k)
        for k in self.COLORS_HEX.keys():
            if k not in self.COLORS_RGB.keys():
                self.COLORS_RGB[k] = self.make_rgb_color(k)


    def make_hex_color(self, c, round='nearest'):
        if c is None or str(c)[0] == '#':
            color = c # do nothing - it's already hex
        elif type(c) == list:
            color = self.rgb_to_hex(c)
        elif c in self.COLORS_HEX.keys():
            color = self.COLORS_HEX[c]
        elif c in self.COLORS_RGB.keys():
            color = self.rgb_to_hex(self.COLORS_RGB[c])
        return color


    def make_rgb_color(self, c):
        if c is None or type(c) == list:
            color = c # do nothing - it's already RGB
        elif c in self.COLORS_RGB.keys():
            color = self.COLORS_RGB[c]
        elif c in self.COLORS_HEX.keys():
            color = self.hex_to_rgb(self.COLORS_HEX[c])
        elif str(c)[0] == '#':
            color = self.hex_to_rgb(c)
        return color

    
    def hex_to_rgb(self, c):
        c = c.replace('#','')
        ch = 4 if len(c) in [4,8] else 3
        n = [c[i*int(len(c)/ch):(i+1)*int(len(c)/ch)] for i in range(ch)]
        if len(c) in [3,4]: n = [s+s for s in n]
        rgb = [int(n,16) for n in n]
        # handle alpha component
        if len(rgb) == 4 and rgb[-1] > 1: rgb[-1] = rgb[-1]/255
        return rgb


    def rgb_to_hex(self, channels, round='nearest'):
        # make a full hex color from 3 or 4 RGB channels
        hex_color = '#'
        # handle alpha component
        if len(channels) == 4 and channels[-1] <= 1.0: channels[-1] = channels[-1]*255
        for c in channels:
            c = max(0, min(255, c))
            if round=='up': c = np.ceil(c,0)
            elif round=='down': c = np.floor(c,0)
            else: c = np.round(c,0)
            hex_color += ('0'+hex(int(c))[2:].upper())[-2:]
        return hex_color


    def divide_range(self, mymin, mymax, size):
        # make a range of evenly spaced floats of a given min, max and length
        return [mymin+(k*(mymax-mymin)/(size-1)) for k in range(size)]
        # np.arange(mymin,mymax+(1/(size-1)),(1/(size-1))) # alternative way
    

    def make_quantiles(self, values, n, mn, mx, spacing='relative'):
        if type(n)==list: n=len(n)
        if spacing == 'even': # evenly distribute the color palette ignoring the magnitude of the values
            return [np.floor((n-1)*((values<=v).mean()+(values<v).mean())/2) for v in values]
        elif spacing == 'relative': # factor in the megnitude of the values when making the visuals (default)
            return [np.maximum(0,np.minimum(int((n-1)*(v-mn)/(mx-mn)),n-2)) for v in values] # prevent negative values
        

    def generate_color(self, value, thresholds, colors):
        # generates an RGB color from a given float, based on distance from defined min/max values and associated rgb colors
        (min,max) = thresholds
        if len(colors) == 1: colors = colors + colors
        (min_color,max_color) = colors
        diff = [min_color[i]-max_color[i] for i in range(len(max_color))]
        return [min_color[j]-(diff[j]*(value-min)/(max-min)) for j in range(len(max_color))]
    
    
    def luminosity_handler(self, text, fill, threshold=100):
        if type(fill) != list:
            fill = self.make_rgb_color(fill)
        if type(text) != list:
            text = [text]
        luminosity = (0.2126*fill[0]+0.7152*fill[1]+0.0722*fill[2])
        return text[0] if luminosity > threshold else text[-1]
    

    def type_format(self, data, val, default):
        if val is None:
            if default == 'min':
                return np.min(data)
            elif default == 'max':
                return np.max(data)
        else:
            return np.quantile(data,val/100)
        

    def ensure_list(self, var):
        if type(var) != list:
            var = [var]
        return var


    def ensure_scalar(self, var):
        if type(var) == list:
            var = var[0]
        return var
    
    
    def run_transform(self, data, funcs=None):
        if funcs is not None:
            if type(funcs) != list:
                funcs = [funcs]
            for f in funcs:
                if f=='log':
                    f = np.log
                    data = np.abs(data)
                elif f=='exp':
                    f = np.exp
                try:
                    data = f(data)
                except:
                    data = f(np.abs(data))
                data = data.replace(np.nan, 0)
        return data


    def make_palette(
        self,
        intervals=[0, 100],
        range_overflow=False,
        fill_palette=None,
        text_palette=None,
        border_palette=None,
        border_top_palette=None,
        border_bottom_palette=None,
        border_right_palette=None,
        border_left_palette=None,
        border_style=None,
        border_top_style=None,
        border_right_style=None,
        border_bottom_style=None,
        border_left_style=None,
        scope='table',
        number='pct',
        edge='fill',
        **kwargs
    ):
        configs = []
        data = np.array(kwargs['data']).flatten()
        
        # handle absolute boundaries
        if number != 'pct':
            if min(intervals) > max(data) or max(intervals) < min(data):
                intervals = []
            else:
                # TO DO: ban duplicates with error handling
                intervals = list(np.unique([stats.percentileofscore(data, v) for i,v in enumerate(intervals)]))

        # decouple the color boundaries from the overall fill boundaries if we want the range_overflow to manifest
        if range_overflow is True:
            max_intervals = max([i for i in intervals])
            min_intervals = min([i for i in intervals])
            fill_intervals = [100*(i-min_intervals)/(max_intervals-min_intervals) for i in intervals]
        else:
            fill_intervals = intervals

        def manage_default_borders(palette, style):
            if palette is None and style is not None:
                palette = self.ensure_list(self.make_hex_color(kwargs['default_border_color']))
            if style is None and palette is not None:
                style = kwargs['default_border_style']
            return palette, style
        
        border_top_palette, border_top_style = manage_default_borders(border_top_palette, border_top_style)
        border_right_palette, border_right_style = manage_default_borders(border_right_palette, border_right_style)
        border_bottom_palette, border_bottom_style = manage_default_borders(border_bottom_palette, border_bottom_style)
        border_left_palette, border_left_style = manage_default_borders(border_left_palette, border_left_style)
        border_palette, border_style = manage_default_borders(border_palette, border_style)

        palette_standardize = lambda p: None if p is None else [p[0] for i in intervals] if len(p) == 1 else p[:len(intervals)]

        for i in range(max(0,len(intervals)-1)):
            configs.append({
                'fill_palette': fill_palette if len(intervals)==2 else None if fill_palette is None else palette_standardize(fill_palette)[i:i+2],
                'text_palette': text_palette if len(intervals)==2 else None if text_palette is None else palette_standardize(text_palette)[i:i+2],
                'border_palette': None if border_palette is None else palette_standardize(border_palette)[i:i+2],
                'border_top_palette': None if border_top_palette is None else palette_standardize(border_top_palette)[i:i+2],
                'border_right_palette': None if border_right_palette is None else palette_standardize(border_right_palette)[i:i+2],
                'border_bottom_palette': None if border_bottom_palette is None else palette_standardize(border_bottom_palette)[i:i+2],
                'border_left_palette': None if border_left_palette is None else palette_standardize(border_left_palette)[i:i+2],
                'border_style': border_style,
                'border_top_style': border_top_style,
                'border_right_style': border_right_style,
                'border_bottom_style': border_bottom_style,
                'border_left_style': border_left_style,
                'intervals': intervals[i:i+2],
                'mymin': intervals[i],
                'mymax': intervals[i+1],
                'fill_intervals': fill_intervals[i:i+2],
                'mymin_fill': fill_intervals[i],
                'mymax_fill': fill_intervals[i+1],
                'range_overflow': range_overflow,
                'edge': edge,
                'number': number,
                'scope': scope,
            })
        for c in configs:
            for k in kwargs.keys(): c[k]=kwargs[k]
        return configs
    

    def apply_colors(
        self,
        col,
        scope='table',
        edge='fill',
        value_transform=None,
        default_fill_color='#FFF',
        default_text_color=['#222','#FFF'],
        default_border_color=None,
        default_border_style=None,
        fill_palette=None,
        text_palette=None,
        text_weight=None,
        font=None,
        border_palette=None,
        border_top_palette=None,
        border_bottom_palette=None,
        border_right_palette=None,
        border_left_palette=None,
        border_style=None,
        border_top_style=None,
        border_bottom_style=None,
        border_right_style=None,
        border_left_style=None,
        data=None,
        rows=None,
        columns=None,
        rows_all=None,
        columns_all=None,
        intervals=[0, 100],
        fill_intervals=[0, 100],
        mymin=None,
        mymax=None,
        mymin_fill=None,
        mymax_fill=None,
        range_overflow=None
    ):
        if value_transform is not None:
            col = self.run_transform(col, value_transform)

        data = np.array(data).flatten()
        global_border_default = ' '.join([default_border_style, self.make_hex_color(default_border_color)])
        scope = ['table' if t is None else t for t in scope]
        range_overflow = [False if t is None else t for t in range_overflow]

        # ------------------------ styles ------------------------- #
        css_palettes = {
            'color': text_palette,
            'background-color': fill_palette,
            'border-top': [border_top_palette[p] or border_palette[p] for p in range(len(border_palette))],
            'border-right': [border_right_palette[p] or border_palette[p] for p in range(len(border_palette))],
            'border-bottom': [border_bottom_palette[p] or border_palette[p] for p in range(len(border_palette))],
            'border-left': [border_left_palette[p] or border_palette[p] for p in range(len(border_palette))],
        }

        css_border_styles = {
            'border-top': [border_top_style[s] or border_style[s] for s in range(len(border_style))],
            'border-right': [border_right_style[s] or border_style[s] for s in range(len(border_style))],
            'border-bottom': [border_bottom_style[s] or border_style[s] for s in range(len(border_style))],
            'border-left': [border_left_style[s] or border_style[s] for s in range(len(border_style))]
        }

        default_styles = {}
        for k in css_palettes.keys(): default_styles[k] = ''
        styles = [default_styles.copy() for j in range(len(col.values))]
        
        for z,(css_style,palette) in enumerate([(p,css_palettes[p]) for p in css_palettes.keys()]):
            rgb_fill_vals = []
            palette = [None if p is None else list(self.make_rgb_color(c) for c in p) for p in palette]
            for i in range(len(palette)):
                quantiles = [q/100. for q in intervals[i]]
                fill_quantiles = [q/100. for q in fill_intervals[i]]
                if palette[i] is not None:
                    if len(palette[i]) == 1:
                        rgb_fill_val = self.make_rgb_color(palette[i][0])
                        rgb_fill_vals += [[rgb_fill_val  for c in col.values]]
                        min_val, max_val, min_val_fill, max_val_fill = [mymin[i], mymax[i], mymin_fill[i], mymax_fill[i]]
                        fill_thresholds = [min_val_fill, max_val_fill]
                    else:
                        if scope[i] == 'table':
                            min_val, max_val, min_val_fill, max_val_fill = [mymin[i], mymax[i], mymin_fill[i], mymax_fill[i]]
                        elif scope[i] == 'column':
                            colvals = [c for s,c in enumerate(col.values) if s in rows[i]]
                            min_val, max_val, min_val_fill, max_val_fill = [np.quantile(colvals, q) for q in quantiles+fill_quantiles]
                        elif scope[i] == 'row':
                            colvals = [c for s,c in enumerate(col.values) if s in [columns_all.index(a) for a in columns[i]]]
                            min_val, max_val, min_val_fill, max_val_fill = [np.quantile(colvals, q) for q in quantiles+fill_quantiles]

                        # palette_orig = np.unique([i for p in fill_intervals for i in p])
                        # palette_colors = palette[0] + [p[1] for p in palette[1:]]
                        # palette_scaled = [min(data) + (p*(max(data)-min(data))/100) for p in palette_orig]
                        # fill_thresholds_global = self.divide_range(min(data), max(data), len(palette_colors))

                        fill_thresholds = self.divide_range(min_val_fill, max_val_fill, len(palette[i]))
                        fill_quantiles = self.make_quantiles(col.values, palette[i], min_val_fill, max_val_fill)
                        rgb_fill_vals += [[self.generate_color(c, fill_thresholds[q:q+2], palette[i][q:q+2]) for c,q in zip(col.values, fill_quantiles)]]
                else:
                    min_val, max_val, min_val_fill, max_val_fill = [mymin[i], mymax[i], mymin_fill[i], mymax_fill[i]]
                    rgb_fill_vals += [[None for c in col.values]]

                for j,v in enumerate(rgb_fill_vals[-1]):

                    try: # elif scope[i] in ['row']:
                        row_selector = rows[i]
                        row_val = rows_all.index(col.name)
                        column_selector = [columns_all.index(c) for c in columns[i]]
                        column_val = j
                    except: # if scope[i] in ['table', 'column']:
                        row_selector = rows[i]
                        row_val = j
                        column_selector = columns[i]
                        column_val = col.name

                    # styles that don't have a palette or gradation, such as font-weight and font-family
                    # these can just run on the first loop and piggyback off our data layer 
                    if z==0 and (column_val in column_selector) and (row_val in row_selector) and (min_val <= col.values[j] <= max_val):
                        if text_weight[i] is not None:
                            styles[j]['font-weight'] = str(text_weight[i])
                        if font[i] is not None:
                            styles[j]['font-family'] = str(font[i])

                    # here we apply actual colors
                    # palette_orig = np.unique([i for p in intervals for i in p])
                    # palette_spread = np.array([0, 33.33, 66.66, 100])
                    if (palette[i] is not None) and (column_val in column_selector) and (row_val in row_selector):
                        if (min_val <= col.values[j] <= max_val):
                            styles[j][css_style] = self.make_hex_color(v)

                            # only add a border style in the relevant ranges and if we're dealing with a border color palette
                            # the last config set covering each cell gets priority
                            if css_style in css_border_styles:
                                styles[j][css_style] = css_border_styles[css_style][i]+' #'+styles[j][css_style].split('#')[-1]
                                
                        elif col.values[j] < min_val and edge[i]=='fill' and styles[j][css_style]=='':
                            proxy_val = min(max_val,max(min_val,col.values[j]))
                            proxy_color = self.generate_color(proxy_val, fill_thresholds[:2], palette[i][:2])
                            if css_style[:6] != 'border': styles[j][css_style] = self.make_hex_color(proxy_color)
                        elif col.values[j] > max_val and edge[i]=='fill':
                            proxy_val = min(max_val,max(min_val,col.values[j]))
                            proxy_color = self.generate_color(proxy_val, fill_thresholds[-2:], palette[i][-2:])
                            if css_style[:6] != 'border': styles[j][css_style] = self.make_hex_color(proxy_color)

                        # elif col.values[j] < min_val and edge[i]=='fill' and styles[j][css_style]=='':
                        #     proxy_val = min(max_val,max(min_val,col.values[j]))
                        #     proxy_color = self.generate_color(proxy_val, fill_thresholds_global[:sum(fill_thresholds_global<proxy_val)+1][-2:], palette_colors[:sum(fill_thresholds_global<proxy_val)+1][-2:])
                        #     print(fill_thresholds_global[:sum(fill_thresholds_global<proxy_val)+1][-2:], palette_colors[:sum(fill_thresholds_global<proxy_val)+1][-2:])
                        #     if css_style[:6] != 'border': styles[j][css_style] = self.make_hex_color(proxy_color)
                        # elif col.values[j] > max_val and edge[i]=='fill':
                        #     proxy_val = min(max_val,max(min_val,col.values[j]))
                        #     proxy_color = self.generate_color(proxy_val, fill_thresholds_global[-(sum(fill_thresholds_global>proxy_val)+1):][:2], palette_colors[-(sum(fill_thresholds_global>proxy_val)+1):][:2]) # BUG still to fix (for general case)
                        #     if css_style[:6] != 'border': styles[j][css_style] = self.make_hex_color(proxy_color)

        for x,s in enumerate(styles):
            if styles[x]['background-color'] in ['', None]: styles[x]['background-color'] = default_fill_color
            if styles[x]['color'] in ['', None]: styles[x]['color'] = self.luminosity_handler(default_text_color, styles[x]['background-color'])
            styles[x]['border-top'] = styles[x]['border-top'] or global_border_default
            styles[x]['border-right'] = styles[x]['border-right'] or global_border_default
            styles[x]['border-bottom'] = styles[x]['border-bottom'] or global_border_default
            styles[x]['border-left'] = styles[x]['border-left'] or global_border_default
            styles[x] = ' '.join([k+': '+s[k]+' !important;' for k in s.keys()])
        return styles
    
    
    def pretty_pandas(
        self,
        df,
        scope='table',
        rows=None, 
        columns=None,
        index='show',
        font_size=None,
        precision=3,
        header_size=None,
        fill_palette=None,
        text_palette=None,
        default_fill_color='#FFF',
        default_text_color=['#222','#FFF'],
        default_border_color=[49, 51, 63, 0.2],
        default_border_style='1px solid',
        range_overflow=False,
        bg='white',
        mymin=None,
        mymax=None,
        value_transform=None,
        configs=None,
        **kwargs
    ):
        sdf = df.style
        rows_all, columns_all = list(df.index), list(df.columns)

        # if we have configs, parse them appropriately
        if configs is not None:

            generated_configs = []
            extra_params = ['fill_palette', 'default_border_color', 'default_border_style']

            for c in configs:
                if 'rows' not in c: c['rows'] = rows_all
                if 'columns' not in c: c['columns'] = columns_all
                c['data'] = df.loc[c['rows'],c['columns']]
                for ep in extra_params:
                    if ep not in c.keys() and locals()[ep] is not None:
                        c[ep] = locals()[ep]
                generated_configs.extend([d for d in self.make_palette(**c)])
            
            configs_default = [defaultdict(lambda: None) for c in generated_configs]
            for c in range(len(generated_configs)):
                for i in generated_configs[c].keys():
                    configs_default[c][i] = generated_configs[c][i]
            fill_palette = [c['fill_palette'] for c in configs_default]
            text_palette = [c['text_palette'] for c in configs_default]
            text_weight = [c['text_weight'] for c in configs_default]
            font = [c['font'] for c in configs_default]
            border_palette=[c['border_palette'] for c in configs_default]
            border_top_palette=[c['border_top_palette'] for c in configs_default]
            border_right_palette=[c['border_right_palette'] for c in configs_default]
            border_bottom_palette=[c['border_bottom_palette'] for c in configs_default]
            border_left_palette=[c['border_left_palette'] for c in configs_default]
            border_style = [c['border_style'] for c in configs_default]
            border_top_style = [c['border_top_style'] for c in configs_default]
            border_right_style = [c['border_right_style'] for c in configs_default]
            border_bottom_style = [c['border_bottom_style'] for c in configs_default]
            border_left_style = [c['border_left_style'] for c in configs_default]
            rows = [c['rows'] if c['rows'] is not None else rows_all for c in configs_default]
            row_indices = [list(rows_all.index(i) for i in r) for r in rows]
            columns = [c['columns'] if c['columns'] is not None else columns_all for c in configs_default]
            scope = [c['scope'] if c['scope'] is not None else scope for c in configs_default]
            range_overflow = [c['range_overflow'] if c['range_overflow'] is not None else range_overflow for c in configs_default]
            intervals = [c['intervals'] for c in configs_default]
            fill_intervals = [c['fill_intervals'] for c in configs_default]
            edge = [c['edge'] for c in configs_default]
            data = [c['data'] for c in configs_default]

            df_subset = [self.run_transform(df if c['range_overflow'] is True else df.loc[rows[i],columns[i]],value_transform) for i,c in enumerate(configs_default)]
            mymin = [self.type_format(df_subset[i].values,c['mymin'],'min') for i,c in enumerate(configs_default)]
            mymax = [self.type_format(df_subset[i].values,c['mymax'],'max') for i,c in enumerate(configs_default)]
            mymin_fill = [self.type_format(df_subset[i].values,c['mymin_fill'],'min') for i,c in enumerate(configs_default)]
            mymax_fill = [self.type_format(df_subset[i].values,c['mymax_fill'],'max') for i,c in enumerate(configs_default)]

        # if, instead, we just have simple parameters, parse these
        else:
            if mymin is None: mymin=np.min(df.values)
            if mymax is None: mymax=np.max(df.values)
            if rows is None: rows = rows_all
            row_indices = [rows.index(r) for r in rows]
            if columns is None: columns = columns_all
            mymin, mymax, rows, columns = [mymin], [mymax], [rows], [columns]
    
        if index=='hide': sdf.hide_index()
        if header_size is None: header_size=font_size
        
        if len(configs_default) > 0:
            axis = 1 if scope[0] == 'row' else 0
            sdf.apply(self.apply_colors, scope=scope, value_transform=value_transform, default_fill_color=default_fill_color, default_text_color=default_text_color,
                    default_border_color=default_border_color, edge=edge, default_border_style=default_border_style, fill_palette=fill_palette,
                    text_palette=text_palette, text_weight=text_weight, font=font, intervals=intervals, fill_intervals=fill_intervals, border_palette=border_palette,
                    border_top_palette=border_top_palette, border_right_palette=border_right_palette, border_bottom_palette=border_bottom_palette,
                    border_left_palette=border_left_palette, border_style=border_style, border_top_style=border_top_style, border_right_style=border_right_style,
                    border_bottom_style=border_bottom_style, border_left_style=border_left_style, rows=row_indices, columns=columns, rows_all=rows_all,
                    columns_all=columns_all, data=data, mymin=mymin, mymax=mymax, mymin_fill=mymin_fill, mymax_fill=mymax_fill, range_overflow=range_overflow, axis=axis)
        return sdf.format('{:.'+str(precision)+'f}') #.set_table_styles([{'selector':'tr','props':[('background-color',bg+' !important')]}])