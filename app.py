import pandas as pd
import plotly.express as px
import streamlit as st
import json
from streamlit_plotly_events import plotly_events
st.set_page_config(page_title="Airbnb Data Viz", page_icon='dashboard/  airbnb_l.svg', layout='wide', initial_sidebar_state='auto')

# st.set_page_config(layout="wide")

states = ['Massachusetts', 'New York','Illinois','Texas']

cities = {
    'Massachusetts': ['Boston', 'Cambridge'],
    'New York': ['New York City', 'Rochester'],
    'Illinois': ['Chicago'],
    'Texas': ['Austin', 'Dallas']
}
city_centers = {
    'Boston': {'lat': 42.3601, 'lon': -71.0589},
    'Cambridge': {'lat': 42.3736, 'lon': -71.1097},
    'New York City': {'lat': 40.7128, 'lon': -74.0060},
    'Rochester': {'lat': 43.161030, 'lon': -77.610924},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298},
    'Austin': {'lat': 30.2672, 'lon': -97.7431},
    'Dallas': {'lat': 32.7767, 'lon': -96.7970},
}

# Function to load data based on selected state and city
def load_data(state, city):
    filename = f'data/{state.lower()}/{city.lower()}/listings.csv'
    cal_filename = f'data/{state.lower()}/{city.lower()}/calendar.csv'
    try:
        df = pd.read_csv(filename)
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
        cal_df = pd.read_csv(cal_filename)
        return df,cal_df
    except FileNotFoundError:
        st.error(f"Data file not found: {filename},{cal_filename}")
        return pd.DataFrame()

# Function to load GeoJSON based on selected state and city
def load_geojson(state, city):
    geojson_path = f'data/{state.lower()}/{city.lower()}/neighbourhoods.geojson'
    try:
        with open(geojson_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON file not found: {geojson_path}")
        return None

# Sidebar for state and city selection
selected_state = st.sidebar.selectbox('Select a state', states, key='state')
selected_city = st.sidebar.selectbox('Select a city', cities[selected_state], key='city')

# Load data and geojson based on the selections
df,cal_df = load_data(selected_state, selected_city)
neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

if not df.empty and neighborhoods_geojson:
    # Initialize session state
    if 'selected_neighborhood' not in st.session_state:
        st.session_state['selected_neighborhood'] = 'All'

    # Sidebar for neighborhood selection
    neighborhoods = ['All'] + sorted(df['neighbourhood_cleansed'].unique())
    selected_neighborhood = st.sidebar.selectbox('Select a neighborhood', neighborhoods, key='neighborhood_dropdown')

    # Reset button
    # if st.button('Reset Selection'):
    #     st.session_state['selected_neighborhood'] = 'All'
    import overpy

    def fetch_museum_artwork_places(city, city_centers):
        api = overpy.Overpass()

        # Use city_centers to get the latitude and longitude
        center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})
        lat, lon = center['lat'], center['lon']

        # Fetch museum and artwork places
        query = f"""
        [out:json];
        (
        node["tourism"="museum"](around:20000,{lat},{lon});
        node["leisure"="park"](around:20000,{lat},{lon});
        node["historic"="monument"](around:20000,{lat},{lon});
        node["historic"="castle"](around:20000,{lat},{lon});
                );
        out center;
        """

        result = api.query(query)

        # Collecting museum and artwork places
        places = []
        for node in result.nodes:
            places.append({
                'name': node.tags.get('name', 'Unnamed'),
                'lat': node.lat,
                'lon': node.lon,
                # 'type': node.tags.get('tourism')
                'type': node.tags.get('tourism', node.tags.get('historic', node.tags.get('leisure', node.tags.get('religion', 'Unknown'))))

            })

        return places
    
    import plotly.graph_objs as go

    def create_choropleth_map(data, city):#, city_centers, neighborhoods_geojson):
        avg_scores = data.groupby('neighbourhood_cleansed')['review_scores_location'].mean().reset_index()
        min_val = avg_scores['review_scores_location'].min() - 0.1
        
        for feature in neighborhoods_geojson['features']:
            neighborhood = feature['properties']['neighbourhood']
            score = avg_scores[avg_scores['neighbourhood_cleansed'] == neighborhood]['review_scores_location']
            feature['properties']['avg_review_score_location'] = score.values[0] if not score.empty else None

        center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})

        fig = px.choropleth_mapbox(
            avg_scores,
            geojson=neighborhoods_geojson,
            locations='neighbourhood_cleansed',
            featureidkey="properties.neighbourhood",
            color='review_scores_location',
            color_continuous_scale="bugn",
            range_color=[min_val, 5],
            mapbox_style="carto-positron",
            # mapbox_style="basic",
            zoom=10,
            center=center,
            opacity=0.5,
            labels={'review_scores_location': 'Location Ratings'}
        )

        color_map = {
            'museum': 'orangered',
            'monument': 'gold',
            'theatre': 'purple',
            'park': 'darkgreen',  # Changed from darkgreen to brown
            'gallery': 'orange',
            # 'religion': 'gold',
            'Unknown': 'gray',  # For any unclassified or missing types
            'artwork': 'magenta',  # Changed from green to magenta

        }

        # # Fetch museum and artwork places
        tourism_places = fetch_museum_artwork_places(city, city_centers)

        # Organize data by type for legend management
        for place_type, color in color_map.items():
            filtered_places = [p for p in tourism_places if p['type'] == place_type]
            if filtered_places:  # Only add traces if there are places of this type
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[p['lat'] for p in filtered_places],
                        lon=[p['lon'] for p in filtered_places],
                        mode='markers+text',
                        marker=go.scattermapbox.Marker(
                            size=9,
                            color=color
                        ),
                        text=[p['name'] for p in filtered_places],
                        textposition='bottom right',
                        name=place_type.capitalize(),  # Use the type as the name for the legend
                        showlegend=True  # Enable legend for this trace
                    )
                )

        fig.update_layout(
            margin={"r":0, "t":0, "l":0, "b":0},
            legend=dict(
                title='Place Types',
                orientation='v',
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=0.01  
            )
        )

        fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        return fig

    st.title(f"Airbnb : {selected_city}, {selected_state}")
    choropleth_map_fig = create_choropleth_map(df,selected_city)
    selected_points = plotly_events(choropleth_map_fig, click_event=True, select_event=False, override_height=600, key="neighborhood")
    print('selected_points',selected_points)
    if selected_points:
        if selected_points[0]['pointIndex']<=len(neighborhoods):
            print(selected_points[0]['pointIndex'])
            st.session_state['selected_neighborhood'] = neighborhoods[selected_points[0]['pointIndex'] + 1]

        else:
            st.session_state['selected_neighborhood'] = neighborhoods[0]

            print("ERROR")

    # Filter data for selected neighborhood
    if st.session_state['selected_neighborhood'] != 'All':
        df_filtered = df[df['neighbourhood_cleansed'] == st.session_state['selected_neighborhood']]
        display_charts = True
    else:
        df_filtered = df
        display_charts = False

    def create_availability_plot(dataframe):
        availability_metrics = dataframe.groupby('neighbourhood_cleansed').agg({
            'availability_30': lambda x: x.mean() / 30 * 100,
            'availability_60': lambda x: x.mean() / 60 * 100,
            'availability_90': lambda x: x.mean() / 90 * 100,
            'availability_365': lambda x: x.mean() / 365 * 100
        }).reset_index()

        availability_long = availability_metrics.melt(id_vars=['neighbourhood_cleansed'], var_name='Availability Period', value_name='Average Availability Percentage')

        fig = px.bar(
            availability_long,
            x='neighbourhood_cleansed',
            y='Average Availability Percentage',
            color='Availability Period',
            title='Average Availability Percentage by Neighbourhood',
            labels={'neighbourhood_cleansed': 'Neighbourhood'},
            barmode='group'
        )

        return fig    
    
    def create_room_type_bar_plot(data):
        
        room_types = ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room']


        room_type_counts = data['room_type'].value_counts().reindex(room_types, fill_value=0).reset_index()
        room_type_counts.columns = ['room_type', 'count']

        # Define a representative Airbnb color
        airbnb_color = '#FF5A5F'

        fig = px.bar(room_type_counts, x='room_type', y='count', 
                    title='Room Type Distribution',
                    labels={'count': 'Count', 'room_type': 'Room Type'},
                    color_discrete_sequence=[airbnb_color])

        fig.update_layout(showlegend=False)
        return fig

    def create_histogram(dataframe):
        # Define the number of bins, or alternatively, set the range and size of each bin
        nbins = 20  # For example, 20 bins
        range_x = [0, 5]  # Assuming the review scores range from 0 to 10
        bin_size = 0.5  # Each bin will have a size of 0.5

        fig = px.histogram(dataframe, x='review_scores_rating', nbins=nbins,
                        title='Distribution of Review Scores',
                        labels={'review_scores_rating': 'Review Score Rating'},
                        color_discrete_sequence=["#FF5A5F"],
                        range_x=range_x,
                        histnorm='percent')  # Optional: normalize to show percentages

        return fig    

    # Define function to create plots for the selected neighborhood
    def create_plots(dataframe):
        fig2 = px.box(dataframe, x='neighbourhood_cleansed', y='price', title='Price Distribution', color_discrete_sequence=["#FF5A5F"])
        fig3 = create_histogram(dataframe)        
        fig4 = create_availability_plot(df_filtered)
        fig5 = create_room_type_bar_plot(df_filtered)
        return fig2, fig3, fig4, fig5
    
    def render_table(df_filtered):
        columns = ['listing_url', 'name',
                'picture_url','host_name',
                'host_picture_url', 'host_is_superhost', 'neighbourhood_cleansed', 
                'property_type', 'room_type', 'accommodates', 
                'bath', 'beds','price']

        def extract_first_word(text):
            try:
                first_word = text.split()[0]
                return pd.to_numeric(first_word)
            except:
                return float('nan')

        # Create new column 'bath'
        df_filtered['bath'] = df_filtered['bathrooms_text'].apply(extract_first_word)


        df_display = df_filtered[columns]
        columns_to_drop_na = ['room_type', 'price', 'host_is_superhost', 'beds', 'accommodates','bath']
        df_display.dropna(subset=columns_to_drop_na, inplace=True)

        return df_display
    
    def create_spider_chart(df, df_filtered):
        categories = ['review_scores_accuracy', 'review_scores_cleanliness', 
                            'review_scores_checkin', 'review_scores_communication', 
                            'review_scores_location', 'review_scores_value']    
        display_categories = [category.replace('review_scores_', '').replace('_', ' ').title() for category in categories]

        # Calculate the mean scores for each review category for the entire dataset
        mean_scores = df[categories].mean().reset_index(name='Value')
        print(mean_scores)
        mean_scores['Type'] = selected_city #'Overall'
        mean_scores['Variable'] = display_categories


        # Calculate the mean scores for each review category for the filtered dataset (specific neighborhood)
        mean_scores_filtered = df_filtered[categories].mean().reset_index(name='Value')
        mean_scores_filtered['Type'] = st.session_state['selected_neighborhood']#'Neighborhood'
        mean_scores_filtered['Variable'] = display_categories

        # Combine the two dataframes
        # print(radar_data.columns)
        radar_data = pd.concat([mean_scores, mean_scores_filtered])
        min_val,max_val = min(radar_data['Value'])-0.5, 5

        print(radar_data['Value'],min(radar_data['Value']))
        radar_data.columns = ['Index', 'Value', 'Type','Variable']
        print(mean_scores_filtered,mean_scores)
        # Create the radar chart
        fig = px.line_polar(radar_data, r='Value', theta='Variable', color='Type', line_close=True,
                            color_discrete_sequence=px.colors.qualitative.D3, # Using a predefined color sequence
                            template="plotly_white", # Using a light theme that fits well with most Streamlit themes
                            range_r=[min_val,max_val],title='Radar plot for Review scores')

        # Update the layout to make it cleaner
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_val,max_val]
                )),
            showlegend=True
        )
        # Add fill with translucency
        fig.update_traces(fill='toself', fillcolor='rgba(0,100,200,0.2)') # Adjust RGBA values as needed

        # Show the plot in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

    from pandas.api.types import (
        is_categorical_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_object_dtype,
    )
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        df[['accommodates','price']] = df[['accommodates','price']].astype({'accommodates':int,'price':float}) 
        with modification_container:
            # to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            to_filter_columns = ['Room Type','Price','Host is superhost?','Beds','Accommodates','Bath']#,
        
        df.rename(columns={'room_type':'Room Type','host_is_superhost':'Host is superhost?','accommodates':'Accommodates','bath':'Bath','beds':'Beds','price':'Price'},inplace=True)

        for i in range(0, len(to_filter_columns), 2):
            cols = st.columns(2)  # Create two columns
            for j, column in enumerate(to_filter_columns[i:i+2]):
                with cols[j]:
                    # st.write("↳")
                    # print(df[column].unique())
                    # Treat columns with < 10 unique values as categorical
                    if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                        user_cat_input = st.multiselect(
                            # f"Values for {column}",
                            f"{column}:",

                            list(sorted(df[column].unique())),
                            default=sorted(list(df[column].unique())),
                        )
                        df = df[df[column].isin(user_cat_input)]
                    elif is_numeric_dtype(df[column]):
                        _min = float(df[column].min())
                        _max = float(df[column].max())
                        step = (_max - _min) / 100
                        user_num_input = st.slider(
                            # f"Values for {column}",
                            f"{column}",
                            _min,
                            _max,
                            (_min, _max),
                            step=step,
                        )
                        df = df[df[column].between(*user_num_input)]
                    elif is_datetime64_any_dtype(df[column]):
                        user_date_input = st.date_input(
                            f"Values for {column}",
                            value=(
                                df[column].min(),
                                df[column].max(),
                            ),
                        )
                        if len(user_date_input) == 2:
                            user_date_input = tuple(map(pd.to_datetime, user_date_input))
                            start_date, end_date = user_date_input
                            df = df.loc[df[column].between(start_date, end_date)]
                    else:
                        user_text_input = st.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            df = df[df[column].str.contains(user_text_input)]

        return df

    def create_parallel_coordinates_plot(data):
        # Define the columns to be included in the plot
        # Ensure that 'price' is at the end to serve as the last axis
        columns = [
            'host_is_superhost',
            'review_scores_rating','accommodates', 'price'
        ]#,'number_of_reviews'

        # Filter the dataframe to include only the relevant columns
        filtered_data = data[columns].dropna()
        filtered_data['host_is_superhost'] = filtered_data['host_is_superhost'].apply(lambda x: 1 if x=='t' else 0)

        # Define the ranges for each dimension, specifically setting price's minimum to 0
        dimensions = [
            dict(range=[filtered_data['host_is_superhost'].min(), filtered_data['host_is_superhost'].max()], label='Is Superhost', values=filtered_data['host_is_superhost']),
            dict(range=[filtered_data['review_scores_rating'].min(), filtered_data['review_scores_rating'].max()], label='Review Score', values=filtered_data['review_scores_rating']),
            dict(range=[0, filtered_data['price'].max()], label='Price', values=filtered_data['price'])
        ]
        filtered_data= pd.concat([filtered_data,pd.DataFrame([{'host_is_superhost':0,'review_scores_rating':0,'price':0}])],ignore_index=True)
        
        # Create the parallel coordinates plot
        fig = px.parallel_coordinates(filtered_data, color="price",
                                    labels={
                                        "host_is_superhost":'Superhost',
                                        "review_scores_rating": "Review Score",
                                        "price": "Price"
                                    },
                                    title="Parallel Plots")
        fig.update_layout(
                margin=dict(l=50)  # Adjust left margin to prevent cutoff , t=50, b=50
            )
        fig.update_traces(line=dict(color='white', width=0.5), selector=dict(mode='lines'))

        
        return fig


    import calplot
    import matplotlib.pyplot as plt
    import io
    def create_neighborhood_calendar_heatmap(df_display_in, cal_df_in, st,neigborhood):

        if type(neigborhood)==str:
            formatted_neighborhood = neigborhood.lower().replace(' ', '_').replace('/', '_')
        else:
            formatted_neighborhood = str(neigborhood).lower().replace(' ', '_').replace('/', '_')


        neighborhood_dir = f'data/{selected_state.lower()}/{selected_city.lower()}/calendar/'

        neighborhood_cal_df = pd.read_csv(f'{neighborhood_dir}{formatted_neighborhood}.csv')
        neighborhood_cal_df['date'] = pd.to_datetime(neighborhood_cal_df['date'])
        neighborhood_cal_df = neighborhood_cal_df[neighborhood_cal_df['date'] > pd.to_datetime('today')]
        data_for_plot = neighborhood_cal_df.set_index('date')['available']

        fig, ax = calplot.calplot(data_for_plot, cmap='OrRd', suptitle=f'{neigborhood} Availability Calendar Heatmap', figsize=(15, 3))
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        st.image(buf, use_column_width=True)

        buf.close()
        plt.close(fig)
    
    from wordcloud import WordCloud
    import re
    def generate_wordcloud(data):
        # Join all the entries in the neighborhood overview column into a single text
        text = ' '.join(description for description in data if description and not pd.isnull(description))
        text = re.sub(r'<.*?>', '', text)
        words_to_remove = ['neighborhood', 'neighbourhood',f'{st.session_state["selected_neighborhood"]}','street','one','city']  # ,selected_city,selected_state Add words to remove
        
        for word in words_to_remove:
            text = text.replace(word, '')        
        wordcloud = WordCloud(width = 800, height = 300, background_color ='white').generate(text)
        
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  
        plt.show()
        
        return fig

    if display_charts:
        st.subheader(f'Neighborhood: {st.session_state["selected_neighborhood"]}')

        word_cloud = generate_wordcloud(df_filtered['neighborhood_overview'])
        st.pyplot(word_cloud)
        fig2, fig3, fig4,fig_bar_room_type = create_plots(df_filtered,)
        col1, col2 = st.columns(2)

        with col1:
            create_spider_chart(df, df_filtered)
            fig5 = create_parallel_coordinates_plot(df_filtered)
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(fig_bar_room_type, use_container_width=True)

        df_display = render_table(df_filtered)

        create_neighborhood_calendar_heatmap(df_filtered, cal_df, st,st.session_state["selected_neighborhood"])

        def highlight_superhost(row):
            print(['color: red' if _ == 't' else 'color: blue' for _ in row])
            # if row=='t':
            # return ['color: red' if _=='t' for _ in row]
            return ['color: red' if _ == 't' else 'color: blue' for _ in row]
        
        def return_index_superhost(df_temp):
            indices = []
            for index, row in df_temp.iterrows():
                if row['Host is superhost?'] == 't':
                    indices.append(index)
            return indices
                    

        df_display = filter_dataframe(df_display)
        super_host_index = return_index_superhost(df_display)
        print('accommodates',df_display['Accommodates'].unique())
        print('bath',df_display['Bath'].unique())
        df_display['neighbourhood_cleansed'] = df_display['neighbourhood_cleansed'].astype(str)

        df_display = (df_display.style.map(lambda _: "background-color: #FF5A5F;",subset=(super_host_index, slice(None))))

        df_display =  st.data_editor(
                        df_display,
                        column_config={
                            "host_picture_url": st.column_config.ImageColumn(
                                "Host Image", help="Host Preview"
                            ),
                            "picture_url": st.column_config.ImageColumn(
                                "Listing Image", help="Listings Preview",width="medium"
                            ),
                            "listing_url": st.column_config.LinkColumn(
                                "Listing URL",help="Listings URL"
                            ),
                            "host_name": st.column_config.TextColumn(
                                "Host",help="Host name"
                            ),                            
                            "name": st.column_config.TextColumn(
                                "Listing Topic",help="Listing Topic"
                            ),                            
                            "neighbourhood_cleansed": st.column_config.TextColumn(
                                "Neighborhood",help="Neighborhood"
                            ),                            
                            "property_type": st.column_config.TextColumn(
                                "Property Type",help="Property Type"
                            ),                            
                            "neighbourhood_cleansed": st.column_config.TextColumn(
                                "Neighborhood",help="Neighborhood"
                            ),                            
                            "room_type": st.column_config.TextColumn(
                                "Room Type",help="Neighborhood"
                            ),                            
                            "accommodates": st.column_config.NumberColumn(
                                "Accommodates",help="Accommodates how many?"
                            ),                            
                            "bath": st.column_config.NumberColumn(
                                "Bath",help="how many bathroom?"
                            ),                            
                            "beds": st.column_config.NumberColumn(
                                "Beds",help="how many beds?"
                            ),                            
                            "price": st.column_config.NumberColumn(
                                "Price (in USD)",help="how many beds?",format="$%d",
                            ),                            
                                                        
                        },
                        hide_index=True,
                        use_container_width=True
                    )
