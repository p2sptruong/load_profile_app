####################################################################################################################
#
# eatlib - Energy Automation Team (EAT) Library
#
# this is a library of functions and classes to be used in EAT scripts



####################################################################################################################
# IMPORTS

import os
import random
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import openpyxl
# import warnings
# import time
# import tkinter as tk
# from tkinter import filedialog
# from tkinter import messagebox
# from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



####################################################################################################################
# FUNCTIONS:

###########################################################################
# e.g. DATA CLEANING FUNCTIONS:

###########################################################################
# e.g. ENERGY FUNCTIONS:

###########################################################################
# PLOTTING FUNCTIONS:

# testing

#####################################################
# plot_time(load_df) - variable vs. time plotting function
#
#   Imports:
#
#   import pandas as pd
#   import datetime
#   import plotly.graph_objects as go (IF USING PLOTLY)
#   from matplotlib import pyplot as plt (IF USING MATPLOTLIB)
#
#
#   Inputs:
#
#   load_df - a pandas DataFrame object with timestamps in the first column
#
#
#   Outputs:
#
#   No outputs. This function just draws a plot. We could change it so it returns a matplotlib object (figure or axes) and then use that to plot later.
#
#
#   TODO:
#
#   -Make more flexible/robust
#   -Show interpolated data in a different color
#
# PLOTLY.GRAPH_OBJECTS VERSION - STABLE
def plot_time(df):
    # convert timestamps to datetime64 objects
    timestamps = pd.to_datetime(df.iloc[:, 0])
    df.iloc[:, 0] = timestamps

    # use graph_objects to create the figure
    fig = go.Figure()

    # create traces for each column vs. time
    for i in range(df.shape[1] - 1):
        fig.add_trace(go.Scatter(x=timestamps, y=df.iloc[:, i + 1],
                                 visible='legendonly',
                                 mode='lines',
                                 name=df.columns[i + 1]))
    # add title and configure layout
    fig.update_layout(
        title=dict(
            text='Simple Trend Data Visualization',
            xanchor='left',
            yanchor='top',
            y=0.9,
            font=dict(color='black')
        ),
        showlegend=True,
        legend_title_text='Points:',
        autosize=True,
        hovermode='x'
    )

    # add annotations
    fig.add_annotation(
        text= 'Click on points<br>in the legend<br>to toggle visibility',
        align='left',
        showarrow=False,
        bordercolor='black',
        borderwidth= 1,
        xref='paper',
        yref='paper',
        xanchor = 'left',
        yanchor = 'bottom',
        x=1.02,
        y=1.05,
        bgcolor="white",
        borderpad = 10,
        font = dict(size = 14, color='black')
    )

    # Print or show the figure, suppress lines as required
    st.plotly_chart(fig, use_container_width=True) # print to streamlit
    # fig.show()    # changed this to return a fig instead of plotting. can change back if we want.

    return fig

#####################################################
# plot_x(load_df) - variable vs. variable plotting function
#
#   Imports:
#
#   import pandas as pd
#   import plotly.express as px (IF USING PLOTLY)
#   from matplotlib import pyplot as plt (IF USING MATPLOTLIB)
#
#
#   Inputs:
#
#   load_df - 'nx2' pandas DataFrame object with x-values in the first column & y-values in the second column.
#
#
#   Outputs:
#
#   No outputs. This function just draws a plot. We could change it so it returns a matplotlib object (figure or axes) and then use that to plot later.
#
# PLOTLY VERSION - STABLE
def plot_x(df):
    # import pandas as pd
    # import plotly.express as px
    print("\n HERE'S A PREVIEW OF THE DATA YOU'RE PLOTTING:")  # show a preview of the data passed to the function
    print(df)   # print some information about the data being plotted
    print()
    print(df.dtypes)

    x_values = df.iloc[:, 0]  # Values in the 1st column will be plotted on the x-axis
    y_values = df.iloc[:, 1]  # Values in the 2nd column will be plotted on the y-axis

    x_label = df.columns[0]  # Name of 1st column is x-axis label
    y_label = df.columns[1]  # Name of 2nd column is y-axis label
    xy_labels = {'x': x_label, 'y': y_label}    # create a dictionary of the labels to pass to px.line

    if (type(x_values[1]) == str) or (type(y_values[1]) == str):    # do a quick check that the data is plottable (i.e. not a string - this could be more robust)
        print("\nERROR: Please make sure you are plotting numerical data.\n")
        return

    fig = px.scatter(x=x_values, y=y_values, labels=xy_labels, title=y_label + ' vs. ' + x_label, trendline="ols")  # plot using plotly
    fig.sow()
    return


####################################################################################################################
# CLASSES:

# class