import pandas as pd
import numpy as np
import seaborn as sns

# set of predefined RGB colors
PALETTES_RGB = {
    'yellow':[252,239,166],
    'green':[122,188,129],
    'red':[231,114,112],
    'white':[255,255,255],
    'blue':[101,147,194],
    'grey':[144,144,148],
    'sns_blue':[13,29,85],
    'sns_yellow':[255,255,221],
    'sns_green':[103,182,193]
}
PALETTES_HEX = {}

# extract the hex value from a given color and round accordingly, ensuring length==2
def make_hex_color(s, round='nearest'):
    if round=='up':
        s_round = np.ceil(s,0)
    elif round=='down':
        s_round = np.floor(s,0)
    else:
        s_round = np.round(s,0)
    return ('0'+hex(int(s_round))[2:].upper())[-2:]

# make a full hex color from 3 RGB channels
def rgb_to_hex(channels, round='nearest'):
    return '#'+(''.join([make_hex_color(c, round) for c in channels]))

# use our helpers to populate hex code dict (we will want to speak generally in hex for this work)
for i in PALETTES_RGB.keys():
    PALETTES_HEX[i] = '#'+(''.join([make_hex_color(color) for color in PALETTES_RGB[i]]))

# make a range of evenly spaced floats of a given min, max and length
def divide_range(mymin, mymax, size):
    return [mymin+(k*(mymax-mymin)/(size-1)) for k in range(size)]
    # np.arange(mymin,mymax+(1/(size-1)),(1/(size-1))) # alternative way

def make_quantiles(values, n, mn, mx, spacing='relative'):
    if type(n)==list: n=len(n)
    if spacing == 'even':  # evenly distribute the color palette ignoring the magnitude of the values
        return [np.floor((n-1)*((values<=v).mean()+(values<v).mean())/2) for v in values]
    elif spacing == 'relative':  # factor in the megnitude of the values when making the visuals (default)
        return [np.maximum(0,np.minimum(int((n-1)*(v-mn)/(mx-mn)),n-2)) for v in values] # prevent negative values

# get RGB colors from hex if we want to go the other way
def get_rgb_colors(c):
    if c in PALETTES_RGB:
        return PALETTES_RGB[c]
    else:
        c = c.replace('#','')
        n = [c[i*int(len(c)/3):(i+1)*int(len(c)/3)] for i in range(3)]
        if len(c)==3: n = [s+s for s in n]
        return [int(n,16) for n in n]

# generates an RGB color value from a given float, based on its distance from defined min/max values and their associated rgb colors
def generate_color(value, thresholds, colors):
    (min,max) = thresholds
    (min_color,max_color) = colors
    diff = [min_color[i]-max_color[i] for i in range(3)]
    return [min_color[j]-(diff[j]*(value-min)/(max-min)) for j in range(3)]

def luminosity(v):
    return (0.2126*v[0]+0.7152*v[1]+0.0722*v[2])


def apply_colors(col, palette=['yellow', 'green'], default_fill_color='#FFF', default_text_color='#000', type='shade', rows=None, columns=None, mymin=None, mymax=None):
    # by default, use column-wise min and max if nothing is provided
    if mymax is None: mymin, mymax = min(col.values), max(col.values)
    
    # to prevent a divide by zero later on - max must always be greater than min
    if mymax==mymin: mymax=mymin+1
    palette = [get_rgb_colors(p) for p in palette]
    
    if len(palette) == 1:
        # if the palette length is just 1 we just apply it globally - the trivial case
        rgb_vals = [palette[0] for c in col.values]
    else:
        # if the palette length is greater than 1, we assign each value a bespoke color based on its position in the full range
        thresholds = divide_range(mymin, mymax, len(palette))
        quantiles = make_quantiles(col.values, palette, mymin, mymax)
        rgb_vals = [generate_color(c, thresholds[q:q+2], palette[q:q+2]) for c,q in zip(col.values, quantiles)]

    def filter_cells(inputs, default=''):  
        if (columns is not None):
            inputs = [inputs[j] if (mymin <= col.values[j] <= mymax) and (col.name in columns) else default for j in range(len(col.values))]
        if (rows is not None):
            inputs = [inputs[j] if (mymin <= col.values[j] <= mymax) and (j in rows) else default for j in range(len(col.values))]
        return inputs
    
    if type == 'shade':
        res = ['background-color: #'+(''.join([make_hex_color(c) for c in v])) for v in rgb_vals]
        default = 'background-color: '+default_fill_color
        return filter_cells(res, default)
    elif type == 'text_shade':
        tx = ['color: '+('#000' if luminosity(v)>=100 else '#FFF') for v in rgb_vals]
        default = 'color: '+default_text_color
        return filter_cells(tx, default)
    else:
        return ['' for c in col.values]

def pretty_pandas(df, fill_palette=['yellow','green'], rows=None, columns=None, index='show', group=None, font_size=None, header_size=None,
                  default_fill_color = '#FFF', default_text_color = '#000', bg='white', mymin=None, mymax=None):
    """Generate efficient dataframe styling with fully customizable inputs.

    Keyword arguments:
    todo
    """
    sdf = df.style
    rows_all,columns_all = list(df.index),list(df.columns)
    if mymin is None: mymin=np.min(df.values)
    if mymax is None: mymax=np.max(df.values)

    if index=='hide': sdf.hide_index()
    if header_size is None: header_size=font_size
    if type(fill_palette[0]) != list: fill_palette=[fill_palette] 
    
    for palette in fill_palette:
        row_subset = rows_all if rows is None else [r for r in rows if r in rows_all]
        row_index_subset = [rows_all.index(r) for r in row_subset]
        col_subset = columns_all if columns is None else [c for c in columns if c in columns_all]
        d = df.loc[row_subset,col_subset]
        mymin = max(mymin, np.min(d.values)) if group is None else None
        mymax = min(mymax, np.max(d.values)) if group is None else None
        sdf.apply(apply_colors, palette=palette, default_fill_color=default_fill_color, default_text_color=default_text_color,
                  type='shade', rows=row_index_subset, columns=col_subset, mymin=mymin, mymax=mymax, axis=0)
        sdf.apply(apply_colors, palette=palette, default_fill_color=default_fill_color, default_text_color=default_text_color,
                  type='text_shade', rows=row_index_subset, columns=col_subset, mymin=mymin, mymax=mymax, axis=0)

    return sdf.format('{:.3f}').set_table_styles([{'selector':'tr','props':[('background-color',bg+' !important')]}])

    # sdf.set_properties(**{'font-size': str(font_size)+'pt'})
    # .set_table_styles([{'selector': 'th', 'props': [('font-size', str(22)+'pt !important')]}])


# EXAMPLES


alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
test_df = pd.DataFrame([np.arange(26)+(2*np.random.random(26)) for i in range(26)],columns=[a for a in alpha][:26])
# test_df = np.round(test_df,2)

fruits = ['Apple','Watermelon','Orange','Pear','Cherry','Strawberry','Nectarine','Grape','Mango','Blueberry','Pomegranate','Starfruit','Plum','Banana',
          'Raspberry','Mandarin','Jackfruit','Papaya','Kiwi','Pineapple','Lime','Lemon','Apricot','Grapefruit','Melon','Coconut','Avocado','Peach']

test_df.index = fruits[:26]
np.random.shuffle(fruits)

# Example 1
pretty_pandas(test_df)

# Example 2
pretty_pandas(
    test_df, index='show', font_size=11, header_size=12,
    # fill_palette=['#FFFFDD','#DAECB8','#87C6BD','#4B96BE','#2E4C9B','#0D1D55'],
    fill_palette=['#e8f6b1', '#b2e1b6', '#65c3bf', '#2ca1c2', '#216daf', '#253997'],
    rows = list(test_df.index)[8:18], #['Starfruit','Plum','Banana','Raspberry'],
    columns = ['B','C','D','E','F','G','H','I'],
    default_fill_color = '#F9F9F9',
    default_text_color = '#DDDDE4',
)

# Example 3
pretty_pandas(
    test_df, index='show', font_size=11, header_size=12, mymin=4, mymax=18,
    fill_palette=list(sns.color_palette('YlOrRd').as_hex()),
    default_fill_color = '#F9F9F9',
    default_text_color = '#555',
)