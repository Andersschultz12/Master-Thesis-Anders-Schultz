import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import json
import math

@st.cache_data
def load_data(subjects):
    aggregated_df = pd.DataFrame()
    frequency_df = pd.DataFrame()

    for subject in subjects:
        aggregated_path = f'Data/parquet_aggregated_data_{subject}'
        a_df = pd.read_parquet(aggregated_path)
        a_df['subject'] = f'Subject {subject}'

        frequency_path = f'Data/parquet_partitioned_spectrogram_{subject}'
        f_df = pd.read_parquet(frequency_path)
        f_df['subject'] = f'Subject {subject}'

        aggregated_df = pd.concat([aggregated_df, a_df], ignore_index=True)
        frequency_df = pd.concat([frequency_df, f_df], ignore_index=True)

    return aggregated_df, frequency_df

def create_custom_domain_dictionary(keys, data):

    final_dict = {}
    quantile_values = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

    for key in keys:
        custom_domain = [data[key].min()] + data[key].quantile(quantile_values).tolist() + [data[key].max()]
        final_dict[key] = custom_domain

    return final_dict

def generate_colorbar_chart(selected_statistic, aggregated_df, domain_labels, brush_y1, brush_y2, outliers_df_upper, outliers_df_lower, colormap, color_spread):
    def make_chart(df, color_value, height):
        return alt.Chart(df).mark_rect(opacity=1).encode(
            y=alt.Y(f'{selected_statistic}:O', axis=alt.Axis(labelOverlap='parity', labelFlush=True, labelBound=True, ticks=False, title=None, format='.1e'), sort='descending'),
            color=color_value,
            opacity=alt.condition(brush_y1 & brush_y2, alt.value(1), alt.value(0.1)),
            tooltip=[alt.Tooltip(f'{selected_statistic}:Q', title='Value', format='.1e')]
        ).properties(width=25, height=height).add_params(brush_y1)

    domain_positions = domain_labels.get(selected_statistic, []) if color_spread == 'Box-Whisker' else [aggregated_df[selected_statistic].min(), aggregated_df[selected_statistic].max()]
    base_color = alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap))
    colorbar = make_chart(aggregated_df, base_color, 120)

    if len(outliers_df_upper) > 0:
        outliers_chart_upper = make_chart(outliers_df_upper, alt.value('red'), 20)
        colorbar = outliers_chart_upper & colorbar

    if len(outliers_df_lower) > 0:
        outliers_chart_lower = make_chart(outliers_df_lower, alt.value('purple'), 20)
        colorbar = colorbar & outliers_chart_lower

    return colorbar

def generate_tick_chart(selected_statistic, aggregated_df, brush_y1, brush_x1, brush_y2, outlier_df_upper, outlier_df_lower, colormap, color_spread, domain_labels, point_selection):
    label_format = '.1e'
    num_electrodes = len(aggregated_df['Electrode'].unique())
    domain_positions = domain_labels.get(selected_statistic, []) if color_spread == 'Box-Whisker' else [aggregated_df[selected_statistic].min(), aggregated_df[selected_statistic].max()]

    def make_chart(df, color_value, height):
        return alt.Chart(df).mark_tick().encode(
            alt.X('time_hms:T', title='Time (H:M:S)', axis=alt.Axis(format='%H:%M:%S', labelAngle=25, labelOverlap=False, labelPadding=5, titlePadding=5)),
            alt.Y('Electrode:N', title=None),
            color=color_value,
            opacity=alt.condition(point_selection & brush_y1 & brush_y2 & brush_x1, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('time_hms:T', title='Timestamp (H:M:S)', format='%H:%M:%S'),
                     alt.Tooltip(f'{selected_statistic}:Q', title='Value', format=label_format)]
        ).properties(width=575, height=height)

    #base_color = alt.condition(brush_y1 & brush_y2 & brush_x1, alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap), legend=None), alt.value('grey'))
    base_color = alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap))
    # Create the base condition
    #base_color = alt.condition(
    #    brush_y1 & brush_y2 & brush_x1,
    #    alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap)),
    #    alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme='plasma'))
    #)

    tick_chart = make_chart(aggregated_df, base_color, 20 * num_electrodes)

    if len(outlier_df_upper) > 0:
        color_upper = alt.value('red')
        #color_upper = alt.condition(brush_y1 & brush_y2 & brush_x1, alt.value('red'), alt.value('grey'))
        outliers_chart_upper = make_chart(outlier_df_upper, color_upper, 20 * num_electrodes)
        tick_chart = tick_chart + outliers_chart_upper

    if len(outlier_df_lower) > 0:
        color_lower = alt.value('DarkMagenta')
        color_lower = alt.condition(brush_y1 & brush_y2 & brush_x1, alt.value('DarkMagenta'), alt.value('grey'))
        outliers_chart_lower = make_chart(outlier_df_lower, color_lower, 20 * num_electrodes)
        tick_chart = tick_chart + outliers_chart_lower

    tick_chart = tick_chart.add_params(brush_x1)
    item = "Spectrogram result" if selected_statistic == 'value' else selected_statistic
    tick_chart = tick_chart.properties(title=f"{item} over time for each electrode")
    
    return tick_chart

def generate_histogram_chart(selected_statistic, aggregated_df, domain_labels, brush_y1, brush_y2, outlier_df_upper, outlier_df_lower, colormap, color_spread):
    def create_histogram(df):
        domain_positions = domain_labels.get(selected_statistic, []) if color_spread == 'Box-Whisker' else [df[selected_statistic].min(), df[selected_statistic].max()]
        return alt.Chart(df).mark_bar(opacity=1).encode(
            y=alt.Y(f'{selected_statistic}:Q', axis=alt.Axis(title=None, ticks=False, labels=False), scale=alt.Scale(domain=[domain_positions[0], domain_positions[-1]]), bin=alt.Bin(maxbins=20)),
            x=alt.X('count()', axis=alt.Axis(title=None, ticks=False, labels=False)),
            #color=alt.condition(brush_y1 & brush_y2,alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap), legend=None), alt.value('grey')),
            color = alt.Color(f'{selected_statistic}:Q', scale=alt.Scale(domain=domain_positions, scheme=colormap), legend=None),
            opacity=alt.condition(brush_y1 & brush_y2, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('count()', title='Count'), alt.Tooltip(f'{selected_statistic}:Q', title='Value', format='.1e')]
        ).properties(width=50, height=120)

    histogram = create_histogram(aggregated_df)

    def add_text_annotation(df, side):
        data_size = len(df)
        text = f'{data_size} Outliers {side}'
        return alt.Chart(pd.DataFrame({'text': [text]})).mark_text(
            align='left', baseline='top', dx=0, dy=4, fontSize=10
        ).encode(x=alt.value(0), y=alt.value(0), text='text:N')

    if len(outlier_df_upper) > 0:
        text_annotation_upper = add_text_annotation(outlier_df_upper, "RHS")
        histogram = text_annotation_upper & histogram

    if len(outlier_df_lower) > 0:
        text_annotation_lower = add_text_annotation(outlier_df_lower, "LHS")
        histogram = histogram & text_annotation_lower

    return histogram

def generate_line_chart(data, x_brush, electrode, title_suffix, brush_interaction):
    label_format = '.1e'
    y_min, y_max = data['y'].min(), data['y'].max()
    line_chart = alt.Chart(data).mark_line(strokeWidth=1).encode(
        alt.X('time_hms:T', title='Timestamp (H:M:S)', axis=alt.Axis(format='%H:%M:%S')),
        alt.Y('y:Q', axis=alt.Axis(title='\u03bcV')).scale(domain=(y_min, y_max)),
        tooltip=[alt.Tooltip('time_hms:T', title='Timestamp (H:M:S)', format='%H:%M:%S'), 
                 alt.Tooltip('y:Q', title='Measure \u03bcV', format=label_format)]
    ).properties(width=350, height=100, title=f'{title_suffix}')

    if brush_interaction == 'filter':
        line_chart = line_chart.transform_filter(x_brush)
    elif brush_interaction == 'add_params':
        line_chart = line_chart.add_params(x_brush)

    return line_chart

def convert_observation_list_to_hm_format(observations_list, obs_rate=250):

    time_strings = []
    for observations in observations_list:
        seconds = observations / obs_rate
        hours = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        seconds = int(seconds) % 60
        time_string = f"{hours:02d}:{minutes:02d}"
        time_strings.append(time_string)
        
    return time_strings

def convert_to_observation_number(time_str, obs_rate=250):
    hours, minutes = map(int, time_str.split(':'))
    total_observations = (hours * 3600 + minutes * 60) * obs_rate
    return total_observations

def create_outlier_IQR(df, statistic):

    Q1, Q3 = df[statistic].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    mask = df[statistic].between(lower, upper)
    outliers_upper = df[~mask & (df[statistic] > upper)].reset_index(drop=True)
    outliers_lower = df[~mask & (df[statistic] < lower)].reset_index(drop=True)
    filtered_df = df[mask].reset_index(drop=True)
    
    return filtered_df, outliers_upper, outliers_lower

def create_outlier_modified_z_score(df, statistic, threshold=3.5):  #(Modified Z-Score)

    median = df[statistic].median()
    abs_deviation = np.abs(df[statistic] - median)
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * abs_deviation / mad
    
    outliers = modified_z_score > threshold
    outlier_direction = ["Right-tailed" if x > median else "Left-tailed" for x in df[statistic]]
    df['Outlier Direction'] = [direction if outlier else "Not an outlier" for outlier, direction in zip(outliers, outlier_direction)]

    filtered_df = df.loc[df['Outlier Direction'] == 'Not an outlier']
    outliers_upper = df.loc[df['Outlier Direction'] == 'Right-tailed']
    outliers_lower = df.loc[df['Outlier Direction'] == 'Left-tailed']
    
    return filtered_df, outliers_upper, outliers_lower

def create_outlier_mad(df, statistic, threshold=2.5):

    median = df[statistic].median()
    abs_deviation = np.abs(df[statistic] - median)
    mad = abs_deviation.median() * 1.4826  
    modified_z_score = 0.6745 * abs_deviation / mad

    df['Outlier'] = modified_z_score > threshold
    df['Outlier Direction'] = ["Not an outlier" if not outlier else
                               ("Right-tailed" if x > median else "Left-tailed")
                               for x, outlier in zip(df[statistic], df['Outlier'])]

    filtered_df = df.loc[df['Outlier Direction'] == 'Not an outlier']
    outliers_upper = df.loc[df['Outlier Direction'] == 'Right-tailed']
    outliers_lower = df.loc[df['Outlier Direction'] == 'Left-tailed']
    
    return filtered_df, outliers_upper, outliers_lower


def generate_overview_frequency_line_chart_seperated(data, freq_brush, point_selection, num_electrodes):
    
    overview = alt.Chart(data).mark_line().encode(
        alt.X('frequency:Q', axis = alt.Axis(title = 'Frequency (hz)', labelPadding=5  , titlePadding=5)),
        alt.Y('value:Q', axis=alt.Axis(title='sum results'), scale=alt.Scale(type='log')),
        alt.Color('Electrode:N', legend=None),
        opacity=alt.condition(point_selection, alt.value(1), alt.value(0.2))
    ).properties(width = 300, height = 280, title = 'Frequency prevalence (Spectrogram)'
    ).add_params(freq_brush)

    legend = alt.Chart(data).mark_point().encode(
        alt.Y('Electrode:N', axis = alt.Axis(labelAngle=0, )).title(None),
        color=alt.condition(point_selection,alt.Color('Electrode:N', legend=None), alt.value('lightgray')) ,
    ).properties(height = 20*num_electrodes, title='Channels').add_params(point_selection)

    combined = legend | (overview)

    return combined

def compute_correlation_matrix_dataframe(aggregated_df, selected_statistic, num_electrodes, electrodes):
    temp_df = aggregated_df[['x_start', 'Electrode', selected_statistic]]
    temp_df = temp_df.loc[(~temp_df['Electrode'].isin(electrodes)) & (temp_df['Electrode'].isin(np.sort(temp_df['Electrode'].unique())[0:num_electrodes]))]
    pivot_df = temp_df.pivot(index='x_start', columns='Electrode', values=selected_statistic)
    corr_matrix = pivot_df.corr()
    corr_matrix = np.round(corr_matrix, 2)
    corr_matrix_reset = corr_matrix.reset_index()
    corr_df = corr_matrix_reset.melt(id_vars='Electrode', var_name='Comparison Electrode', value_name='Correlation')
    return corr_df

def generate_correlation_matrix_chart(corr_df, selected_statistic, num_electrodes, point_selection):
    
    corr_chart = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('Comparison Electrode:N', axis=alt.Axis(title=None)),
        y=alt.Y('Electrode:N', axis=alt.Axis(title=None)),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme = "redblue", domain=[-1, 0, 1]),  legend=alt.Legend(title=None)),
        opacity=alt.condition(point_selection, alt.value(1), alt.value(0.2)),
        tooltip=['Correlation']
    ).properties(width = 17*num_electrodes, height = 17*num_electrodes, title= f'Correlation Matrix {selected_statistic}')

    return corr_chart

def update_data_raw(selected_electrode_one, selected_electrode_two, selected_time_start, selected_interval_size, detail_df_one, detail_df_two,brush_2, line_brush, time_int_1, time_int_2):

    time_int_1 = convert_to_observation_number(time_int_1, obs_rate=250)
    time_int_2 = convert_to_observation_number(time_int_2, obs_rate=250)
    interval_start = convert_to_observation_number(selected_time_start, obs_rate=250)
    
    new_df_2 = detail_df_one.loc[(detail_df_one['x'] >= time_int_1) & (detail_df_one['x'] <= time_int_2)]
    new_df_2 = new_df_2.iloc[::500, :]
    new_df_3 = detail_df_two.loc[(detail_df_two['x'] >= time_int_1) & (detail_df_two['x'] <= time_int_2)]
    new_df_3 = new_df_3.iloc[::500, :]
    detail_df_one = detail_df_one.loc[(detail_df_one['x'] >= interval_start) & (detail_df_one['x'] <= interval_start+250*selected_interval_size)]
    detail_df_one_v2 = detail_df_one.iloc[::20, :]

    line_chart_detail1 = generate_line_chart(new_df_2[['y', 'time_hms']], brush_2, selected_electrode_one, f'Sampled view of {selected_electrode_one}', 'filter') #"add_params"
    line_chart_detail2 = generate_line_chart(detail_df_one_v2, line_brush, selected_electrode_one, f'Sampled view of {selected_electrode_one} Interval', "add_params")
    line_chart_detail3 = generate_line_chart(detail_df_one, line_brush, selected_electrode_one, f'Raw data of {selected_electrode_one} Interval', 'filter')
    full_line_chart_1 = line_chart_detail1 | line_chart_detail2 | line_chart_detail3
    
    detail_df_two = detail_df_two.loc[(detail_df_two['x'] >= interval_start) & (detail_df_two['x'] <= interval_start+250*selected_interval_size)]
    detail_df_two_v2 = detail_df_two.iloc[::20, :]
    
    line_chart_detail1 = generate_line_chart(new_df_3[['y', 'time_hms']], brush_2, selected_electrode_two, f'Sampled view of {selected_electrode_two}', 'filter') #"add_params"
    line_chart_detail2 = generate_line_chart(detail_df_two_v2, line_brush, selected_electrode_two, f'Sampled view of {selected_electrode_two} Interval', "add_params")
    line_chart_detail3 = generate_line_chart(detail_df_two, line_brush, selected_electrode_two, f'Raw data of {selected_electrode_two} Interval', 'filter')
    full_line_chart_2 = line_chart_detail1 | line_chart_detail2 | line_chart_detail3
    
    return (full_line_chart_1 & full_line_chart_2)


def extract_relevant_data(data1, data2, subject_number, electrodes, time_int_1, time_int_2, num_electrodes, lower_freq=0, upper_freq=0):

    manip_df1 = data1.loc[(data1['subject'].isin([subject_number]))]
    data2.drop(['time_hms'], axis=1, inplace=True)
    manip_df2 = data2.query('(@lower_freq <= frequency <= @upper_freq)  and (@subject_number == subject)')
    manip_df1 = manip_df1.loc[(~manip_df1['Electrode'].isin(electrodes)) & (manip_df1['Electrode'].isin(np.sort(manip_df1['Electrode'].unique())[0:num_electrodes]))]
    manip_df2 = manip_df2.loc[(~manip_df2['Electrode'].isin(electrodes)) & (manip_df2['Electrode'].isin(np.sort(manip_df2['Electrode'].unique())[0:num_electrodes]))] 
     
    manip_df2 = manip_df2.groupby(['Electrode', 'time']).sum().reset_index()
    manip_df2 = manip_df2.loc[(manip_df2['Electrode'].isin(np.sort(manip_df2['Electrode'].unique())[0:num_electrodes]))] 
    #st.write(manip_df2)
    #st.write(manip_df1)#
	
    manip_df1['value'] = manip_df2['value'].to_numpy()

    time_int_1 = convert_to_observation_number(time_int_1, obs_rate=250)
    time_int_2 = convert_to_observation_number(time_int_2, obs_rate=250)

    manip_df_full = manip_df1.loc[(manip_df1['x_start'] >= time_int_1) & (manip_df1['x_start'] <= time_int_2)]

    return manip_df_full

def update_freq_overview(subject_number, frequency_df, electrodes, num_electrodes, point_selection):
    temp_df = frequency_df.loc[frequency_df['subject'] == subject_number][['time','frequency','value','Electrode']]
    temp_df = temp_df.loc[(~temp_df['Electrode'].isin(electrodes)) & (temp_df['Electrode'].isin(np.sort(temp_df['Electrode'].unique())[0:num_electrodes]))]

    agg_freq_df_1 = temp_df.groupby(['Electrode', 'frequency']).sum().reset_index()
    agg_freq_df_1 = agg_freq_df_1.loc[(agg_freq_df_1['Electrode'].isin(np.sort(agg_freq_df_1['Electrode'].unique())[0:num_electrodes]))]

    #agg_freq_df_2 = temp_df[['time', 'frequency', 'value']]
    #agg_freq_df_2 = agg_freq_df_2.groupby(['frequency']).sum().reset_index()

    freq_brush = alt.selection_interval(encodings=['x'])
    overview = generate_overview_frequency_line_chart_seperated(agg_freq_df_1, freq_brush, point_selection, num_electrodes)

    return overview

def update_stat_tick_mark(aggregated_df, frequency_df, subject_number, electrodes, time_int_1, time_int_2, num_electrodes, \
                          selected_statistic, brush_2, brush_freq, brush_stat, colormap, color_spread, point_selection, lower_freq, upper_freq, outlier_method):
           
    manipulated_df = extract_relevant_data(aggregated_df, frequency_df, subject_number, electrodes, time_int_1, time_int_2, num_electrodes, lower_freq, upper_freq)

    if outlier_method == 'MAD':
        manip_df_stat, outliers_df_upper, outliers_df_lower = create_outlier_mad(manipulated_df, selected_statistic)
        manip_df_freq, outliers_freq_upper, outliers_freq_lower = create_outlier_mad(manipulated_df, 'value')
    elif outlier_method == 'Mod Z Score':
        manip_df_stat, outliers_df_upper, outliers_df_lower = create_outlier_modified_z_score(manipulated_df, selected_statistic)
        manip_df_freq, outliers_freq_upper, outliers_freq_lower = create_outlier_modified_z_score(manipulated_df, 'value')
    elif outlier_method == 'IQR * 1.5':
        manip_df_stat, outliers_df_upper, outliers_df_lower = create_outlier_IQR(manipulated_df, selected_statistic)
        manip_df_freq, outliers_freq_upper, outliers_freq_lower = create_outlier_IQR(manipulated_df, 'value')

    domain_labels1 = create_custom_domain_dictionary([f'{selected_statistic}'], manip_df_stat)
    domain_labels2 = create_custom_domain_dictionary(['value'], manip_df_freq)

    tick = generate_tick_chart(selected_statistic, manip_df_stat, brush_stat, brush_2, brush_freq, outliers_df_upper, outliers_df_lower, colormap, color_spread, domain_labels1, point_selection)
    colorbar = generate_colorbar_chart(selected_statistic, manip_df_stat, domain_labels1, brush_stat, brush_freq, outliers_df_upper, outliers_df_lower, colormap, color_spread)
    histogram = generate_histogram_chart(selected_statistic, manip_df_stat, domain_labels1, brush_stat, brush_freq, outliers_df_upper, outliers_df_lower, colormap, color_spread)
    full_combined_chart_1 = tick | colorbar | histogram

    tick = generate_tick_chart('value', manip_df_freq, brush_freq, brush_2, brush_stat, outliers_freq_upper, outliers_freq_lower, colormap, color_spread,  domain_labels2, point_selection)
    colorbar = generate_colorbar_chart('value', manip_df_freq, domain_labels2, brush_freq, brush_stat, outliers_freq_upper, outliers_freq_lower, colormap, color_spread)
    histogram = generate_histogram_chart('value', manip_df_freq, domain_labels2, brush_freq, brush_stat, outliers_freq_upper, outliers_freq_lower, colormap, color_spread)
    full_combined_chart_2 = tick | colorbar | histogram

    full_combined_chart = (full_combined_chart_2 & full_combined_chart_1).resolve_scale(color ='independent')

    return full_combined_chart

def update_overview_data_section(aggregated_df, frequency_df, selected_statistic, lower_freq, upper_freq, time_int_1, \
                time_int_2, color_spread, colormap, electrodes,subject_number, num_electrodes, brush_2, brush_freq, brush_stat, point_selection, outlier_method): 

    overview = update_freq_overview(subject_number, frequency_df, electrodes, num_electrodes, point_selection)
    corr_df = compute_correlation_matrix_dataframe(aggregated_df, selected_statistic, num_electrodes, electrodes)
    corr_chart = generate_correlation_matrix_chart(corr_df, selected_statistic, num_electrodes, point_selection)
    
    overview_left = overview & corr_chart
    overview_right = update_stat_tick_mark(aggregated_df, frequency_df, subject_number, electrodes, time_int_1, time_int_2, \
                                           num_electrodes, selected_statistic, brush_2, brush_freq, brush_stat, colormap, color_spread, point_selection, lower_freq, upper_freq, outlier_method)
    
    resolved_chart = (overview_left | overview_right).resolve_scale(color = 'independent')
    return resolved_chart


def save_combined_chart(resolved_chart, line_charts_combined):
    full_chart = alt.vconcat(resolved_chart, line_charts_combined, spacing=100)
    #full_chart = resolved_chart & line_charts_combined
    full_chart.save('full_chart_test.html')

def open_HTML_chart_specification(file_path, num_electrodes): 
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        st.components.v1.html(html_content, width=1500, height=40*num_electrodes + 150 + 600) # Use components.html to render HTML
    except FileNotFoundError:
        st.error("The file was not found at the specified path.")

# Load saved configurations
def load_configurations():
    configs = {}
    for file in os.listdir('configs'):
        if file.endswith('.json'):
            with open(f'configs/{file}', 'r') as f:
                configs[file[:-5]] = json.load(f)
    return configs

# Save current configuration
def save_configuration(config, name):
    with open(f'configs/{name}.json', 'w') as f:
        json.dump(config, f)

def create_br_space(num_needed):
    for _ in range(num_needed):
        st.markdown('<br>', unsafe_allow_html=True)

def tab_tick_mark_overview():
    if not os.path.exists('configs'):
        os.makedirs('configs')

    alt.data_transformers.disable_max_rows()
    alt.renderers.enable('default', embed_options={'renderer': 'canvas'})

    aggregated_df, frequency_df = load_data(['003']) #'001'
    aggregated_df.columns = [col.replace(':', '').replace(',', '').replace(' ', '_') for col in aggregated_df.columns]
    column_names = list(aggregated_df.columns)
    
    columns_to_remove = ['index', 'Electrode', 'x_start', 'time_hms', 'subject']
    column_keys = [column for column in column_names if column not in columns_to_remove]
    domain_dict = create_custom_domain_dictionary(column_keys, aggregated_df)
    metrics = list(domain_dict.keys())
    col0, spacer0, col1, spacer1, col2 = st.columns([0.5, 0.02, 3, 0.02, 0.4])
    
    brush_stat = alt.selection_interval(encodings=['y'])
    brush_freq = alt.selection_interval(encodings=['y'])
    brush_2 = alt.selection_interval(encodings=['x'])
    line_brush = alt.selection_interval(encodings=['x'])
    point_selection = alt.selection_point(fields=['Electrode'])
    
    minutes_max = math.ceil((int(aggregated_df['x_start'].max())-int(aggregated_df['x_start'].min())) / 250 / 60)
    options = [i for i in range(int(aggregated_df['x_start'].min()), int(aggregated_df['x_start'].max()), int((aggregated_df['x_start'].max()-aggregated_df['x_start'].min()) / minutes_max))][1:]
    options_time = convert_observation_list_to_hm_format(options, 250)

    with col0:
        configurations = load_configurations()
        st.subheader('Overview')
        selected_config = st.selectbox('Load Configuration', options=[''] + list(configurations.keys()))
        config_name = st.text_input('Enter configuration name', value='Save State 1')
        button_clicked = st.button('Save Configuration')

        if selected_config:
            config = configurations[selected_config]
            default_subject_number = config.get('subject_number', 'Subject 003')
            default_num_electrodes = config.get('num_electrodes', 5)
            default_electrodes = config.get('electrodes', [])
        else:
            config = None
            default_subject_number = 'Subject 003'
            default_num_electrodes = 5
            default_electrodes = []

        subject_number = st.selectbox('Patient number',  options=['Subject 003'], index=['Subject 003'].index(default_subject_number))
        elec_options_ = list(aggregated_df.loc[aggregated_df['subject'] == subject_number]['Electrode'].unique())
        num_electrodes = st.select_slider('Number of Electrodes:', options=[i for i in range(1, len(elec_options_)+1)], value=default_num_electrodes)
        elec_options = elec_options_[:num_electrodes]
        electrodes = st.multiselect("Remove Electrodes", elec_options, default=default_electrodes)
        buttom_pressed = st.button(' Apply Changes ', key='button1')

    with col2:
        color_scales = ['viridis', 'cividis', 'inferno', 'magma', 'plasma']
        color_mappings = ['Box-Whisker', 'Standard']
        outlier_options = ['MAD', 'Mod Z Score', 'IQR * 1.5']
        options_freq_ = [f for f in range(0, 101, 1)]

        # Set default values based on whether a configuration is loading
        if selected_config:
            config = configurations[selected_config]
            default_time_interval = (config.get('time_int_1', options_time[0]), config.get('time_int_2', options_time[25]))
            default_freq_interval = (config.get('lower_freq', options_freq_[0]), config.get('upper_freq', options_freq_[10]))
            default_colormap = config.get('colormap', 'viridis')
            default_color_spread = config.get('color_spread', 'Box-Whisker')
            default_outlier = config.get('outlier_method', 'MAD')
            default_statistic = config.get('selected_statistic', 'envelope')
        else:
            config = None
            default_time_interval = (options_time[0], options_time[25])
            default_freq_interval = (options_freq_[0], options_freq_[10])
            default_colormap = 'viridis'
            default_color_spread = 'Box-Whisker'
            default_outlier = 'MAD'
            default_statistic = metrics[3]  # Assuming the default index is 3

        # Widget definitions
        time_int_1, time_int_2 = st.select_slider('Time interval',options=options_time,value=default_time_interval)
        lower_freq, upper_freq = st.select_slider('Frequency interval (hz)',options=options_freq_,value=default_freq_interval)
        selected_statistic = st.selectbox('Statistic:',options=metrics, index=metrics.index(default_statistic))
        colormap = st.selectbox('Colorscale:',options=color_scales, index=color_scales.index(default_colormap))
        color_spread = st.selectbox('Colormap:',options=color_mappings, index=color_mappings.index(default_color_spread))
        outlier_method = st.selectbox('Outlier Detection:', options=outlier_options, index=outlier_options.index(default_outlier))

    with col0:
        excess_electrodes = num_electrodes - len(electrodes)
        if excess_electrodes > 7:
            create_br_space(excess_electrodes - 6)

        st.subheader('Detail View')
        left, right = st.columns([0.5, 0.5])

        # Define electrode options excluding the already selected electrodes
        detail_elec_options_1 = [item for item in elec_options if item not in electrodes]

        # Determine the default selections based on config or set defaults for new configurations
        if selected_config:
            config = configurations[selected_config]
            default_selected_electrode_one = config.get('selected_electrode_one', detail_elec_options_1[0])
            default_selected_electrode_two = config.get('selected_electrode_two', None) 
            default_interval_size = config.get('selected_interval_size', 60)
            default_time_start = config.get('selected_time_start', options_time[options_time.index(time_int_1)])
        else:
            default_selected_electrode_one = detail_elec_options_1[0]
            default_selected_electrode_two = None
            default_interval_size = 60
            default_time_start = options_time[options_time.index(time_int_1)]

        with left:
            selected_electrode_one = st.selectbox('Electrode #1',options=detail_elec_options_1, index=detail_elec_options_1.index(default_selected_electrode_one), key='elec1detail')
        detail_elec_options_2 = [item for item in detail_elec_options_1 if item != selected_electrode_one]
        default_selected_electrode_two = default_selected_electrode_two if default_selected_electrode_two in detail_elec_options_2 else detail_elec_options_2[0]

        with right:
            selected_electrode_two = st.selectbox('Electrode #2', options=detail_elec_options_2, index=detail_elec_options_2.index(default_selected_electrode_two), key='elec2detail')
        
        detail_df_one = pd.read_parquet(f"Data/parquet_partitioned_{subject_number.split(' ')[-1]}/Electrode={selected_electrode_one}")
        detail_df_two = pd.read_parquet(f"Data/parquet_partitioned_{subject_number.split(' ')[-1]}/Electrode={selected_electrode_two}")

        selected_interval_size = st.select_slider("Interval Size", options=[i for i in range(5, 121, 5)], value=default_interval_size)
        selected_time_start = st.select_slider("Interval start", options=options_time[options_time.index(time_int_1):options_time.index(time_int_2)+1], value=default_time_start)

    with col2:
        create_br_space(3)
        #config_name = st.text_input('Enter configuration name', value='Save State 1')
        if button_clicked:  # Button to save the current configuration
            if config_name:
                save_configuration({
                    'subject_number': subject_number,
                    'num_electrodes': num_electrodes,
                    'electrodes': electrodes,
                    'time_int_1': time_int_1,
                    'time_int_2': time_int_2,
                    'lower_freq': lower_freq,
                    'upper_freq': upper_freq,
                    'colormap': colormap,
                    'color_spread': color_spread,
                    'selected_statistic': selected_statistic,
                    'outlier_method': outlier_method,
                    'selected_electrode_one': selected_electrode_one,
                    'selected_electrode_two': selected_electrode_two,
                    'selected_interval_size': selected_interval_size,
                    'selected_time_start': selected_time_start,
                }, config_name)
                st.success(f'Configuration {config_name} saved!')

    if 'initialized' not in st.session_state:
        file_path = 'full_chart_test.html'
        st.session_state['initialized'] = True
        resolved_chart = update_overview_data_section(aggregated_df, frequency_df, selected_statistic, lower_freq, upper_freq, time_int_1, \
                     time_int_2, color_spread, colormap, electrodes,subject_number, num_electrodes, brush_2, brush_freq, brush_stat, point_selection, outlier_method)
        line_charts_combined = update_data_raw(selected_electrode_one, selected_electrode_two, selected_time_start, \
                                               selected_interval_size, detail_df_one, detail_df_two, brush_2, line_brush, time_int_1, time_int_2)
        save_combined_chart(resolved_chart, line_charts_combined)
        with col1:
            open_HTML_chart_specification(file_path, num_electrodes)
    else:
        if buttom_pressed or selected_config:
            file_path = 'full_chart_test.html'
            resolved_chart = update_overview_data_section(aggregated_df, frequency_df, selected_statistic, lower_freq, upper_freq, time_int_1, \
                        time_int_2, color_spread, colormap, electrodes,subject_number, num_electrodes, brush_2, brush_freq, brush_stat, point_selection, outlier_method)
            line_charts_combined = update_data_raw(selected_electrode_one, selected_electrode_two, selected_time_start, \
                                                selected_interval_size, detail_df_one, detail_df_two, brush_2, line_brush, time_int_1, time_int_2)
            save_combined_chart(resolved_chart, line_charts_combined)
            with col1:
                open_HTML_chart_specification(file_path, num_electrodes)
        else:
            with col1:
                file_path = 'full_chart_test.html'
                with open(file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()

                st.components.v1.html(html_content, width=1500, height=40*24 + 150 + 400) # Use components.html to render HTML

    
    a0, a1, a2, a3 = st.columns([0.5, 1, 0.5, 2.5])

    if 'editor_content' not in st.session_state:
        st.session_state.editor_content = ""

    html_string1 = """ <div style='font-size: 20px; font-weight: bold;'> Electrode Artifact </div> """
    html_string2 = """ <div style='font-size: 20px; font-weight: bold;'> Region Artifact </div> """
    html_string3 = """ <div style='font-size: 20px; font-weight: bold;'> Specific Artifact </div> """

    with a0:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(html_string1, unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(html_string2, unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(html_string3, unsafe_allow_html=True)

    with a1:
        electrodes_ = st.multiselect("Select Faulty Electrode Channels:", elec_options, key="electrodes_2")
        timestamp1, timestamp2 = st.select_slider( 'Select Faulty Time regions (All electrodes)',options=options_time,value=(options_time[0], options_time[25]), key="region1")

        b1, b2 = st.columns([1,1])
        with b1:
            electrodes_2 = st.multiselect("Electrode:", elec_options, key="electrodes_3")
        with b2:
            timestamp3, timestamp4 = st.select_slider('Time region',options=options_time,value=(options_time[0], options_time[25]), key="region2")

    with a3:
        editor_area = st.text_area("Text Editor", value=st.session_state.editor_content, height=300, key="editor_area")

    with a2:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Add to list', key="1"):
            # Update the session state variable for editor content
            st.session_state.editor_content += f"Defect electrodes {electrodes_}\n\n"
            # Rerun the app to reflect the updated content in the text area
            st.experimental_rerun()
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Add to list', key="2"):
            # Update the session state variable for editor content
            st.session_state.editor_content += f"Defect time regions {[timestamp1, timestamp2]}\n\n"
            # Rerun the app to reflect the updated content in the text area
            st.experimental_rerun()
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button('Add to list', key="3"):
            st.session_state.editor_content += f"Defect electrodes {electrodes_2} at time regions {[timestamp3, timestamp4]}\n\n"
            # Rerun the app to reflect the updated content in the text area
            st.experimental_rerun()
    with a3:
        c1, c2 = st.columns([1, 0.2])
        with c2:
            if st.button('Save content'):
                # Specify the directory and file name where you want to save the content
                save_path = 'saved_content.txt'
                
                # Use Python's with statement to open a file and write the content to it
                with open(save_path, 'w') as file:
                    file.write(st.session_state.editor_content)
                
                # Provide feedback to the user that the content has been saved
                st.success(f"Content saved to {save_path}!")
                
                # Optional: Provide a link to download the file directly from the app
                # This requires the file to be located in a directory accessible to the app
                with open(save_path, "rb") as fp:
                    btn = st.download_button(
                        label="Download text file",
                        data=fp,
                        file_name="saved_content.txt",
                        mime="text/plain"
                    )

def preprocessing_tab():#
    st.title('[CONCEPT:] Upload and Process .set File for EEG Data')
    st.header('Future add on can incude the ability to preprocess new subject directly from the application')
    # File uploader
    uploaded_file = st.file_uploader("Choose a .set file", type='set')
    if uploaded_file is not None:
        # To save file to disk
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())

        # User inputs for data processing
        obs_rate = st.number_input('Observation Rate (Hz)', min_value=1, value=250)  # Default to 250 Hz
        low_freq = st.number_input('Low Frequency Filter (Hz)', min_value=0.1, value=0.1, format="%.1f")
        high_freq = st.number_input('High Frequency Filter (Hz)', min_value=1.0, value=100.0, format="%.1f")

        if st.button('Process Data'):
            st.success('File uploaded and processed successfully!')

def display_tabs():
    tabs = {
    "Spectrogram View": tab_tick_mark_overview,
    "Preprocessing of new data": preprocessing_tab
    }

    st.sidebar.header("EGG Data visualization tool")
    password = st.sidebar.text_input("Enter Password", type="password")
    correct_password = "123321"
    
    if password == correct_password:
        selected_tab = st.sidebar.radio("Select a tab:", list(tabs.keys()))
        tabs[selected_tab]()

st.set_page_config(layout="wide",)
display_tabs()