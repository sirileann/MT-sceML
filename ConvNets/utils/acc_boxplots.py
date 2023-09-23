import glob
import plotly.io as pio
import re
import json
import plotly.graph_objects as go
from natsort import natsorted
import random

model_version = "v1"
plotly_title = "Normal Dataset"


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2 ** 16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}
    return pio.from_json(json.dumps(plotly_json))


# Step 1: Get a list of file paths matching the pattern
file_paths = natsorted(glob.glob(f'../logs/model_{model_version}/version_*/plotlys/acc_*.html'))

# Step 2: Initialize lists for data and layout
combined_data = []

# Step 3: Initialize combined figure and layout
combined_fig = go.Figure()

# Define the number of colors you want
num_colors = len(file_paths)

"""
# Generate the color palette randomly
color_palette = []
random.seed(666)
for i in range(num_colors):
    # Generating a random number in between 0 and 2^24
    hex_color = 0
    while len(str(hex_color)) != 8:
        color = random.randrange(0, 2**24)

        # Converting that number from base-10 (decimal) to base-16 (hexadecimal)
        hex_color = hex(color)

    std_color = "#" + hex_color[2:]
    color_palette.append(std_color)
"""

# 15 pre-defined colors
color_palette = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#800000',
                 '#008000', '#000080', '#808000', '#008080', '#800080', '#FFA500', '#A52A2A',
                 '#FFFFF0']

# Step 4: Loop through each file and merge the data
for file_index, file_path in enumerate(file_paths):
    fig = read_from_html(file_path)

    # Extract data information
    data = fig.data

    # Assign color to boxplots from the same acc_* file
    color_index = file_index % len(color_palette)
    for trace in data:
        trace.marker.color = color_palette[color_index]

        # Set legend group to the file name
        trace.legendgroup = file_path

    # Merge data
    combined_data += data

    # Merge traces
    for trace in data:
        combined_fig.add_trace(trace)

# Step 5: Update layout
combined_fig.update_layout(
    yaxis_title='Error [mm]',
    boxmode='group',  # group together boxes of the different traces
    showlegend=True,  # enable legend
    title=plotly_title
)

# Step 6: Create a legend trace for each file
legend_data = []
for file_index, file_path in enumerate(file_paths):
    # Create an invisible trace for each file to represent it in the legend
    legend_trace = go.Scatter(
        x=[None],  # no data
        y=[None],  # no data
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)'),  # invisible marker
        legendgroup=file_path,
        name=file_path,
        showlegend=True,
    )
    legend_data.append(legend_trace)

# Add the legend traces to the combined figure
for legend_trace in legend_data:
    combined_fig.add_trace(legend_trace)

# Step 7: Show and save the combined plot
# combined_fig.show()
combined_fig.write_html(f"../logs/model_{model_version}/accuracies_model_{model_version}_overview.html")
