# Airbnb Data Visualization App

## Overview
This project is a data visualization application built with Streamlit, designed to provide insightful analytics for Airbnb listings across several cities. It aims to assist travelers, hosts, and market analysts by offering a detailed look at the accommodation landscape through various interactive visualizations.

## Features
- **Dynamic Selection**: Users can select states, cities, and specific neighborhoods to view data.
- **Interactive Maps**: Choropleth maps display neighborhoods based on ratings such as location, cleanliness, and overall satisfaction.
- **Visualization Suite**:
  - **Word Clouds**: Display key terms associated with neighborhoods.
  - **Polar/Radar Plots**: Compare average property scores against city averages.
  - **Histograms and Bar Plots**: Show distributions of ratings and property types.
  - **Parallel Coordinate Plots**: Explore relationships between different factors like superhost status and pricing.
  - **Calendar Heatmaps**: Reveal trends in property availability through the year.
  - **Interactive Tables**: Filter and select properties, with direct links to Airbnb for booking.
- **Data Filters**: Users can adjust filters to refine their search based on preferences like price and accommodations.

## Technologies Used
- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **Plotly**: For creating interactive charts.
- **Streamlit**: To build and share the web application.
- **Overpy**: For querying OpenStreetMap data.
- **Matplotlib** and **CalPlot**: For calendar heatmap visualization.
- **WordCloud**: For generating word cloud images.

## Installation
To run this project locally, follow these steps:

```bash
git clone https://github.com/<your-github-username>/airbnb-data-viz.git
cd airbnb-data-viz
pip install -r requirements.txt
streamlit run app.py
'''

## Data
The data used in this project consists of Airbnb listings and calendar availability, structured by state and city in CSV format. GeoJSON files are used for mapping neighborhoods.

## Usage
Once the application is running, navigate through the sidebar to select different filters and view the visualizations that appear based on your selections. For neighborhood users may click directly on choropleth map to select paritcular neighborhood.
