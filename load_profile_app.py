####################################################################################################################
# load_profile_app.py
#
# testing plotting load profile stuff

####################################################################################################################
# IMPORTS
import numpy as np
import pandas as pd
from eatlib import *  # import eatlib - the only library you'll ever need
import plotly.express as px
from plotly.subplots import make_subplots
import openpyxl

####################################################################################################################
# VARIABLES

# do some housekeeping and create some variables
example_data_path = './Input Load Profiles/'
logo = './images/P2S LOGO_COLOR.png'
sheet_name = 'Data'
data_range = 'A:I'
meta_data_range = 'K:L'
meta_data_count = 8
y_axis_options = ['Operating Hours', 'Instantaneous Output']
x_axis_options = ['MBH', 'Btu/sf']
mbh_flag = False


####################################################################################################################
# FUNCTIONS

# helper function for displaying text input cleanly
def text_field(label, columns=None, **input_params):
    c1, c2 = st.beta_columns(columns or [1, 4])

    # Display field name with some alignment
    c1.markdown("##")
    c1.markdown(label)

    # Sets a default key parameter to avoid duplicate key errors
    input_params.setdefault("key", label)

    # Forward text input parameters
    return c2.text_input("", **input_params)  # Notice that you can forward text_input parameters naturally

# open the file and get the sheet names
def process_upload(uploaded_file, data_range=data_range, meta_data_range=meta_data_range):
    excel_file = pd.ExcelFile(uploaded_file, engine='openpyxl')
    sheet_names = excel_file.sheet_names

    # let the user select a sheet
    if len(sheet_names) > 1:
        sheet_name = st.selectbox('Choose a sheet:', sheet_names)
    else:
        sheet_name = 0

    # read load data into dataframe and calculate the total load, drop empty rows
    load_df = excel_file.parse(
        sheet_name=sheet_name,
        usecols=data_range,
        engine='openpyxl'
    )
    load_df.dropna(axis='index', how='all', inplace=True)

    # read static inputs & metadata into dataframe - hard coded
    meta_df = excel_file.parse(
        sheet_name=sheet_name,
        index_col=0,
        header=0,
        nrows=meta_data_count,
        usecols=meta_data_range,
        engine='openpyxl'
    )
    meta_df.drop(index='P2S Project No')

    return load_df, meta_df, excel_file, sheet_names

def data_inspector(load_df):
    # <editor-fold desc="Process timestamps, characterize time series, calculate some variables of interest">
    # process timestamps, characterize time series, calculate some variables of interest
    data_count = load_df.shape[0]
    start = pd.to_datetime(load_df['Timestamp'].iloc[0])
    start_plus_one = pd.to_datetime(load_df['Timestamp'].iloc[1])
    end = pd.to_datetime(load_df['Timestamp'].iloc[-1])
    td_in_hrs = round((start_plus_one - start).seconds / 3600, 2)
    total_hrs = len(load_df) * td_in_hrs
    total_op_hrs = len(load_df['Heating Load (MBH)'][load_df['Heating Load (MBH)'] > 0]) * td_in_hrs
    total_load = load_df['Heating Load (MBH)'].sum()
    max_load = round(load_df['Heating Load (MBH)'].max(), 2)
    neg_loads = load_df['Heating Load (MBH)'][load_df['Heating Load (MBH)'] < 0]
    check_data = pd.DataFrame.from_dict(
        {'Data count (# of rows)': str(data_count),
         'Trend start': str(start),
         'Trend stop': str(end),
         'Timestamp interval (hrs)': str(td_in_hrs),
         'Total hours': str(total_hrs),
         'Total operating hours': str(total_op_hrs)
         },
        orient='index',
        columns=['Trend Data Quality Checks']
    )
    # </editor-fold>

    # <editor-fold desc="Configure data inspector">
    # let the user look at the data in table & graph format
    with st.expander('Click to inspect data'):
        st.table(check_data)
        st.write(load_df)
        plot_time(load_df)
    # </editor-fold>

# the main event
def plot_load_profile(load_df, meta_df):
    # <editor-fold desc="Process timestamps, characterize time series, calculate some variables of interest">
    # process timestamps, characterize time series, calculate some variables of interest
    # TODO: get rid of redundancy b/w this and data_inspector()
    data_count = load_df.shape[0]
    start = pd.to_datetime(load_df['Timestamp'].iloc[0])
    start_plus_one = pd.to_datetime(load_df['Timestamp'].iloc[1])
    end = pd.to_datetime(load_df['Timestamp'].iloc[-1])
    td_in_hrs = round((start_plus_one - start).seconds / 3600, 2)
    total_hrs = len(load_df) * td_in_hrs
    total_op_hrs = len(load_df['Heating Load (MBH)'][load_df['Heating Load (MBH)'] > 0]) * td_in_hrs
    total_load = load_df['Heating Load (MBH)'].sum()
    max_load = round(load_df['Heating Load (MBH)'].max(), 2)
    neg_loads = load_df['Heating Load (MBH)'][load_df['Heating Load (MBH)'] < 0]
    check_data = pd.DataFrame.from_dict(
        {'Data count (# of rows)': str(data_count),
         'Trend start': str(start),
         'Trend stop': str(end),
         'Timestamp interval (hrs)': str(td_in_hrs),
         'Total hours': str(total_hrs),
         'Total operating hours': str(total_op_hrs)
         },
        orient='index',
        columns=['Trend Data Quality Checks']
    )
    #</editor-fold>

    # # <editor-fold desc="Configure data inspector">
    # # let the user look at the data in table & graph format
    # with st.expander('Click to inspect data'):
    #     st.table(check_data)
    #     st.write(load_df)
    #     plot_time(load_df)
    # # </editor-fold>

    # <editor-fold desc="Handle missing GSF/MBH inputs">
    # read GSF & design MBH from spreadsheet and do some stuff if it's not a real input
    gsf = meta_df.iloc[1, 0]
    mbh_design = round(meta_df.iloc[0, 0], 2)
    if pd.isna(mbh_design):
        slider_label = 'Missing installed capacity data. Set BTU/sf to adjust limits of graph. Default is 30 BTU/sf'
        slider_default = 30.00
        mbh_flag = True
    else:
        slider_label = 'Override design BTU/sf value below. Calculated value from uploaded file is {:,} BTU/sf'.format(
            round(1000 * mbh_design / gsf, 2))
        slider_default = mbh_design * 1000 / gsf
        mbh_flag = False
    # </editor-fold>

    # <editor-fold desc="Let user change graph settings">
    # Let user change graph settings
    with st.expander('Click to change graph settings'):
        # let user choose units for x- & y- axes
        col1, col2 = st.columns(2)
        y_axis_units = col1.radio('Choose units for the y-axis:', y_axis_options)
        # x_axis_units = col2.radio('Choose units for the x-axis:', x_axis_options)

        # display a slider for user to adjust x-axis limit
        btu_sf_override = st.slider(
            label=slider_label,
            min_value=0.0,
            max_value=100.0,
            value=slider_default,
            step=0.01
        )
        mbh_design = round(gsf * btu_sf_override / 1000, 2)
    # </editor-fold>

    # <editor-fold desc="Calculate 5% load increment & Btu/sf for design & actual">
    # calculate 5% load increment & Btu/sf for design & actual
    mbh_increment = mbh_design / 20
    btu_sf_design = round(1000 * mbh_design / gsf, 2)
    btu_sf_actual = round(1000 * max_load / gsf, 2)
    # </editor-fold>

    # <editor-fold desc="Create bins and axes for plotting">
    # create bins and axes for plotting
    decimal_labels = list(np.zeros(20))
    increment_labels = list(np.zeros(20))
    labels = list(np.zeros(20))
    for i in range(20):
        decimal_labels[i] = str(5 * (i + 1) / 100) + 'x'
        increment_labels[i] = '(' + str(round(mbh_increment * i)) + '-' + str(round(mbh_increment * (i + 1))) + ' MBH)'
        labels[i] = '<b>' + decimal_labels[i] + '</b><br>' + increment_labels[i]
    # </editor-fold>

    # <editor-fold desc="Create some variables">
    # create some variables
    binned_loads = [0] * 20
    cumulative_loads = [0] * 20
    cumulative_percent = [0] * 20
    c_load = 0
    cumulative_hours = [0] * 20
    c_hours = 0
    # bins and variables for hours histogram
    # counts, bin_edges = np.histogram(load_df['Heating Load (MBH)'][load_df['Heating Load (MBH)'] > 0], bins=20, range=(0,mbh_design))
    counts = [0] * 20
    # </editor-fold>

    # <editor-fold desc="Populate variables">
    # populate variables
    for i in range(20):
        binned_loads[i] = load_df['Heating Load (MBH)'][(load_df['Heating Load (MBH)'] > (i * mbh_increment)) & (
                    load_df['Heating Load (MBH)'] <= ((i + 1) * mbh_increment))].sum()
        counts[i] = load_df['Heating Load (MBH)'][(load_df['Heating Load (MBH)'] > (i * mbh_increment)) & (
                    load_df['Heating Load (MBH)'] <= ((i + 1) * mbh_increment))].count()

        c_load += binned_loads[i]
        c_hours += counts[i] * td_in_hrs

        cumulative_loads[i] = c_load
        cumulative_percent[i] = 100 * cumulative_loads[i] / total_load

        cumulative_hours[i] = c_hours
    # </editor-fold>

    # <editor-fold desc="Set y-axis variables based on y-axis units">
    # Set y-axis variables based on y-axis units
    if y_axis_units == 'Operating Hours':
        y1 = [x / sum(counts) * 100 for x in counts]
        y2 = [x / total_op_hrs * 100 for x in cumulative_hours]
        customdata = np.stack([decimal_labels, increment_labels]).transpose()
        y1_title_text = "<b>Operating hours</b>"
        y2_title_text = "<b>Cumulative</b><br>(100% = {:,}".format(int(sum(counts) * td_in_hrs)) + " hours)"
        hovertemplate1 = '<b>%{y:.2f}% of total operating hours</b> <extra>@ %{customdata[0]} design capacity</extra>'
        hovertemplate2 = '<b>%{y:.2f}% of total operating hours</b> <extra>@ ≤%{customdata[0]} design capacity</extra>'
        color = '#3B6D89'
    elif y_axis_units == 'Instantaneous Output':
        y1 = [round(x / total_load * 100, 2) for x in binned_loads]
        y2 = cumulative_percent
        customdata = np.stack([decimal_labels, increment_labels]).transpose()
        y1_title_text = "<b>Instantaneous Heating Output</b>"
        y2_title_text = "<b>Cumulative</b><br>(100% = {:,}".format(round(total_load)) + " kBtus)"
        hovertemplate1 = '<b>%{y:,}% of total heating output</b> <extra>@ %{customdata[0]} design capacity</extra>'
        hovertemplate2 = '<b>%{y:.2f}% of total heating output</b> <extra>@ ≤%{customdata[0]} design capacity</extra>'
        color = '#00C496'
    # </editor-fold>

    # <editor-fold desc="Set x-axis variables based on x-axis units">
    # # Set x-axis variables based on x-axis units
    # #TODO: add x-axis option for Btu/sf
    # if x_axis_units == 'MBH':
    #     customdata = np.stack([decimal_labels, increment_labels]).transpose()
    #     y1_title_text = "<b>Operating hours</b>"
    #     y2_title_text = "<b>Cumulative</b><br>(100% = {:,}".format(int(sum(counts) * td_in_hrs)) + " hours)"
    #     hovertemplate1 = '<b>%{y:.2f}% of total operating hours</b> <extra>@ %{customdata[0]} design capacity</extra>'
    #     hovertemplate2 = '<b>%{y:.2f}% of total operating hours</b> <extra>@ ≤%{customdata[0]} design capacity</extra>'
    #     color = '#3B6D89'
    # </editor-fold>

    # <editor-fold desc="Create the figure">
    # Create a figure with a secondary y1-axis
    fig = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}]]
    )
    # </editor-fold>

    # <editor-fold desc="Set figure title">
    title = '<b>Heating Load Distribution</b> from {} to {}'.format(start.strftime('%B %-d, %Y'),
                                                                    end.strftime('%B %-d, %Y'))
    for i in range(len(meta_df) - 2):
        title += '<br>' + meta_df.index[i + 2] + ': ' + str(meta_df.iloc[i + 2, 0])
    fig.update_layout(
        autosize=True,
        title=dict(
            text=title,
            xanchor='left',
            yanchor='top',
            y=0.95,
            font=dict(color='black')
        )
    )
    # </editor-fold>

    # <editor-fold desc="Add the bar chart on the primary axis">
    fig.add_trace(
        go.Bar(
            x=labels,
            y=y1,
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=hovertemplate1
        ),
        secondary_y=False,
        row=1,
        col=1
    )
    # </editor-fold>

    # <editor-fold desc="Add the cumulative percent line on the secondary axis">
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=y2,
            mode='lines+markers',
            marker=dict(
                color="#FB9A2D",
            ),
            customdata=customdata,
            hovertemplate=hovertemplate2
        ),
        secondary_y=True,
        row=1,
        col=1
    )
    # </editor-fold>

    # <editor-fold desc="Add asterisks on 'Design MBH' & 'Design Btu/sf' to indicate that these are assumptions">
    # TODO: Clean this up
    # Add asterisks on "Design MBH" & "Design Btu/sf" to indicate that these are assumptions
    if mbh_flag:
        annotation_text = "<b>*Design MBH</b>: {:,}<br><b>*Design Btu/sf</b>: {:,}<br><br><b>Max. actual MBH</b>: {:,} \
                          <br><b>Max. actual *Btu/sf</b>: {:,}<br>".format(mbh_design, btu_sf_design, max_load,
                                                                           btu_sf_actual)
    else:
        annotation_text = "<b>Design MBH</b>: {:,}<br><b>Design Btu/sf</b>: {:,}<br><br><b>Max. actual MBH</b>: {:,} \
                          <br><b>Max. actual Btu/sf</b>: {:,}<br>".format(mbh_design, btu_sf_design, max_load,
                                                                          btu_sf_actual)
    # </editor-fold>

    # <editor-fold desc="Add annotations">
    # add annotations
    fig.add_annotation(
        text=annotation_text,
        align='left',
        showarrow=False,
        bordercolor='black',
        borderwidth=2,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='bottom',
        x=0.94,
        y=1.05,
        bgcolor="white",
        borderpad=10,
        font=dict(size=14, color='black')
    )
    # </editor-fold>

    # <editor-fold desc="Throw a warning if input data contains negative loads">
    # throw a warning if input data contains negative loads
    if len(neg_loads) > 0:
        fig.add_annotation(
            text="<b>WARNING</b>:<br>Input file contains negative load data-<br> # of negative data points: {} of {}<br> \
    total negative kBtus: {}".format(len(neg_loads), len(load_df), round(sum(neg_loads) * td_in_hrs, 2)),
            align='left',
            showarrow=False,
            bordercolor='red',
            borderwidth=2,
            xref='paper',
            yref='paper',
            xanchor='right',
            yanchor='bottom',
            x=0.6,
            y=1.05,
            bgcolor="white",
            borderpad=10,
            font=dict(size=14, color='red')
        )
    # </editor-fold>

    # <editor-fold desc="Configure y1-axis">
    # set y1-axis title
    fig.update_yaxes(
        title_text=y1_title_text,
        color=color,
        showline=True,
        linewidth=2,
        linecolor='black',
        secondary_y=False,
        nticks=4,
        ticksuffix='%',
        row=1
    )
    # </editor-fold>

    # <editor-fold desc="Configure y2-axis">
    # set y2-axis title
    fig.update_yaxes(
        title_text=y2_title_text,
        color="#FB9A2D",
        showline=True,
        linewidth=2,
        linecolor='black',
        secondary_y=True,
        showgrid=False,
        range=[0, 105],
        nticks=2,
        ticksuffix='%',
        row=1
    )
    # </editor-fold>

    # <editor-fold desc="Configure x-axis">
    # Set x-axis title
    fig.update_xaxes(
        title_text='<b>Part load operating point</b><br>(1.0x = design capacity, ' + str(mbh_design) + ' MBH)',
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        row=1,
        type='category'
    )
    # </editor-fold>

    # <editor-fold desc="Customize hover labels">
    # customize hover labels
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        )
    )
    # </editor-fold>

    # <editor-fold desc="Set margins">
    # set margins
    fig.update_layout(
        margin=dict(
            l=50,
            r=50,
            t=175,
            b=50
        )
    )
    # </editor-fold>

    # <editor-fold desc="Set legend">
    # Set legend
    fig.update_layout(
        legend=dict(
            xanchor='right',
            yanchor='top',
            x=0.9,
            y=0.35
        ),
        showlegend=False
    )
    # </editor-fold>]

    # <editor-fold desc="Set figure height">
    # Set figure height
    fig.update_layout(
        autosize=False,
        height=750
    )
    # </editor-fold>

    # print plot to streamlit app
    st.plotly_chart(fig, use_container_width=True)

    return fig


####################################################################################################################

# BEGINNING OF STREAMLIT APP
st.set_page_config(layout="wide")

# display a banner logo
st.image(logo)

# display an expanding 'About' section
with st.expander('About'):
    """Welcome to the party! This is an experimental app being developed by the P2S Energy Automation 
    Team/Decarbonization Task Group. If you run into any bugs/errors or have suggestions for additional 
    features/functionality, please use the "Report a bug with this app" tool in the drop down menu in the top right 
    corner of this page. Thanks for playing! """

"""
# Heating Load Profile Analyzer App

This app visualizes heating load profile data. 

Upload some data to get started. Make a copy of the "Load Profile 
Template.xlsx" file at this location and populate it with data: 

*L:\Office Folder\Decarbonization Task Force\Modeling, Sizing, and Trend Data Working Group*
"""

# display a select box for uploading a file vs. seeing an example
app_mode = st.selectbox('Would you like to see an example, or upload a file?', ['Upload a file', 'See example'])

# APP MODE: EXAMPLE
if app_mode == 'See example':
    # set example file here, or choose a random example file from the folder
    # example_file = example_data_path + random.choice(os.listdir(example_data_path))
    example_file = './Input Load Profiles/SDSU Heat Load Analysis Multiple Bldgs.xlsx'
    # example_file = '/Users/maxsun/PycharmProjects/load_profile_app/Input Load Profiles/test.xlsx'
    # example_file = '/Users/maxsun/PycharmProjects/load_profile_app/Input Load Profiles/Dining HHW Only_TEST.xlsx'

    # process the uploaded file and turn it into a dataframe
    load_df, meta_df, excel_file, sheet_names = process_upload(example_file)

    # show the data inspector
    data_inspector(load_df)

    # run plotting function - includes interactive user input
    run_plotter = st.checkbox('Run Analysis',value=False)
    if run_plotter:
        fig = plot_load_profile(load_df, meta_df)

# APP MODE: UPLOAD A FILE
elif app_mode == 'Upload a file':
    # display a file_uploader widget
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # process the uploaded file and turn it into a dataframe
        load_df, meta_df, excel_file, sheet_names = process_upload(uploaded_file)

        # show the data inspector
        data_inspector(load_df)

        # run plotting function - includes interactive user input
        run_plotter = st.checkbox('Run Analysis', value=False)
        if run_plotter:
            fig = plot_load_profile(load_df, meta_df)

# # Example of giving user the option of overwriting metadata in the app TODO: let them save it back to the excel file, fix issue with str v. float
# with st.expander('Click to overwrite static inputs/metadata (this will NOT change the uploaded file)'):
#     for i in range(len(meta_df)):
#         # text_field() is a streamlit specific helper function for generating dialog boxes
#         meta_df.iloc[i, 0] = text_field(meta_df.index[i], value = meta_df.iloc[i, 0])

# display a radio buton for selecting y1-axis units
# y_axis_units = st.radio('Choose units for the y-axis:', y_axis_options)
