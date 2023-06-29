import streamlit as st
import os
from zipfile import ZipFile
import datetime

import folium
from folium.features import DivIcon
from streamlit_folium import folium_static

import pandas as pd
import gpxpy
import geopy.distance

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from itertools import cycle


def string_hours_minutes_seconds_to_timedelta(string):
    return pd.to_timedelta(
        sum([int(e) * 60 ** index for index, e in enumerate(string.split(":")[::-1])]),
        unit="seconds"
    )


def timedelta_formatter(td):  # defining the function
    td_sec = td.seconds  # getting the seconds field of the timedelta
    hour_count, rem = divmod(td_sec, 3600)  # calculating the total hours
    minute_count, second_count = divmod(rem, 60)  # distributing the remainders
    if td.days > 0:
        msg = "{} days, {}:{}:{}".format(td.days, str(hour_count).zfill(2), str(minute_count).zfill(2),
                                         str(second_count).zfill(2))
    elif hour_count > 0:
        msg = "{}:{}:{}".format(str(hour_count).zfill(2), str(minute_count).zfill(2), str(second_count).zfill(2))
    else:
        msg = "{}:{}".format(str(minute_count).zfill(2), str(second_count).zfill(2))

    return msg


def get_color_palette(color_blind_palette=None):
    if color_blind_palette is None:
        color_blind_palette = False
    if color_blind_palette:
        color_palette = list(sns.color_palette(palette="colorblind", n_colors=15).as_hex())
    else:
        color_palette = list(sns.color_palette(palette="bright", n_colors=15).as_hex())
    return color_palette


def get_df_from_gpx_file(file):
    with open(file) as f:
        gpx = gpxpy.parse(f)
        points = [
            {
                "Date": point.time.date(),
                "Time": point.time,
                "Latitude": point.latitude,
                "Longitude": point.longitude,
                "Elevation": point.elevation,
            } for point in gpx.tracks[0].segments[0].points
        ]
        df = pd.DataFrame.from_records(points)
    return df


def parse_asics_zip_file(path):
    with ZipFile(path, "r") as zipfile:
        gpx_df_list = []

        file_list = zipfile.filelist

        for file in file_list:
            if file.filename.endswith("gpx"):
                zipfile.extract(file)
                gpx_df_list.append(get_df_from_gpx_file(file.filename))
                os.remove(file.filename)
            elif file.filename.startswith("cardioActivities"):
                zipfile.extract(file)
                cardioActivities_df = pd.read_csv(file.filename)
                os.remove(file.filename)
            elif file.filename.startswith("measurements"):
                zipfile.extract(file)
                measurements_df = pd.read_csv(file.filename)
                os.remove(file.filename)
            else:
                print(f"Unexpected file: {file.filename}")
    gpx_df = pd.concat(
        gpx_df_list,
        ignore_index=True,
    )

    # Feature Engineering
    cardioActivities_df = cardioActivities_df.sort_values("Date").reset_index(drop=True)
    cardioActivities_df["Duration"] = \
        cardioActivities_df["Duration"].apply(lambda x: string_hours_minutes_seconds_to_timedelta(x))
    cardioActivities_df["Average Pace"] = \
        cardioActivities_df["Average Pace"].apply(lambda x: string_hours_minutes_seconds_to_timedelta(x))
    cardioActivities_df["Total Distance (km)"] = cardioActivities_df["Distance (km)"].cumsum()

    mask = measurements_df.Type == "weight"
    measurements_df = measurements_df[mask].sort_values("Date").reset_index(drop=True)

    def custom_aggregation(df):
        # print(df)
        aggregation_dict = dict()

        aggregation_dict["Date"] = df["Date"]

        aggregation_dict["Time"] = df["Time"]
        aggregation_dict["Latitude"] = df["Latitude"]
        aggregation_dict["Longitude"] = df["Longitude"]
        aggregation_dict["Elevation"] = df["Elevation"]

        aggregation_dict["TimeDiff"] = \
            aggregation_dict["Time"].diff().dt.total_seconds().fillna(0)
        aggregation_dict["TimeAcc"] = aggregation_dict["TimeDiff"].cumsum()
        aggregation_dict["ElevationDiff"] = aggregation_dict["Elevation"].diff().fillna(0)
        aggregation_dict["ElevationAcc"] = aggregation_dict["ElevationDiff"].cumsum()

        aggregation_dict["LastLatitude"] = df["Latitude"].shift(1).fillna(df["Latitude"])
        aggregation_dict["LastLongitude"] = df["Longitude"].shift(1).fillna(df["Longitude"])
        aggregation_dict["Distance"] = \
            [
                geopy.distance.distance((lat, lon), (lastlat, lastlon)).m for lat, lon, lastlat, lastlon in zip(
                aggregation_dict["Latitude"],
                aggregation_dict["Longitude"],
                aggregation_dict["LastLatitude"],
                aggregation_dict["LastLongitude"],
            )
            ]
        aggregation_dict["DistanceAcc"] = pd.Series(aggregation_dict["Distance"]).cumsum().tolist()

        aggregation_dict["AveragePace"] = \
            pd.to_timedelta([
                (t / 60.0) / (m / 1000.0) if m > 0 else 0 for t, m in
                zip(aggregation_dict["TimeDiff"], aggregation_dict["Distance"])
            ], unit="minutes").round("s")

        return pd.DataFrame(aggregation_dict)

    gpx_df = (
        gpx_df
        .groupby(
            ["Date"],
            as_index=False,
        )
        .apply(custom_aggregation)
        .reset_index(drop=True)
    )

    gpx_df["Average Pace"] = gpx_df["AveragePace"].apply(
        lambda x: datetime.datetime.combine(datetime.datetime.today().date(), datetime.datetime.min.time()) + x)

    def custom_aggregation(df):
        aggregation_dict = {}

        km_list = []
        average_pace_list = []
        elevation_list = []
        lat_lon_list = []
        last_distance = None
        last_time = None
        last_elevation = None

        for km in range(1, int(df["DistanceAcc"].max() / 1000.0) + 1):
            mask = df["DistanceAcc"] / 1000.0 >= km
            aux_df = df[mask].sort_values(["TimeAcc"]).head(1)
            distance = aux_df["DistanceAcc"].values[0]
            time = aux_df["TimeAcc"].values[0]
            elevation = aux_df["Elevation"].values[0]
            lat_lon = aux_df[["Latitude", "Longitude"]].apply(lambda x: (x.Latitude, x.Longitude), axis=1).values[0]
            if last_distance is None:
                average_pace = (time / 60.0) / (distance / 1000.0)
                elevation_diff = elevation - df.sort_values(["TimeAcc"])["Elevation"].values[0]
            else:
                average_pace = ((last_time - time) / 60.0) / ((last_distance - distance) / 1000.0)
                elevation_diff = elevation - last_elevation

            km_list.append(km)
            average_pace_list.append(pd.to_timedelta(average_pace, unit="minutes").round("s"))
            elevation_list.append(int(round(elevation_diff)))
            lat_lon_list.append(lat_lon)

            last_distance = distance
            last_time = time
            last_elevation = elevation

        aggregation_dict["Date"] = [df["Date"].values[0]] * len(km_list)

        aggregation_dict["Km"] = km_list
        aggregation_dict["AveragePace"] = average_pace_list
        aggregation_dict["Elevation"] = elevation_list
        aggregation_dict["Points"] = lat_lon_list

        return pd.DataFrame(aggregation_dict)

    km_df = (
        gpx_df
        .groupby(
            ["Date"],
            as_index=False,
        )
        .apply(custom_aggregation)
        .reset_index(drop=True)
    )

    km_df["Average Pace"] = km_df["AveragePace"].apply(
        lambda x: datetime.datetime.combine(datetime.datetime.today().date(), datetime.datetime.min.time()) + x)

    return cardioActivities_df, measurements_df, gpx_df, km_df


def render_dashboard():
    st.title("Asics RunKeeper Summary")

    color_blind_palette = st.checkbox("Do you want to use a colorblind palette for the charts?")
    if color_blind_palette is None:
        color_blind_palette = False

    _color_palette = get_color_palette(color_blind_palette=color_blind_palette)

    uploaded_zip_file = st.file_uploader("Choose a file", type=["zip"])

    if uploaded_zip_file is None:
        return

    cardioActivities_df, measurements_df, gpx_df, km_df = parse_asics_zip_file(uploaded_zip_file)

    if cardioActivities_df is None or cardioActivities_df.empty:
        return

    st.header("Main KPIs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Max distance", value=f"{cardioActivities_df['Distance (km)'].max()} Km")
        st.metric(label="Accumulated distance", value=f"{cardioActivities_df['Distance (km)'].sum()} Km")

    with col2:
        st.metric(
            label="Min average pace",
            value=f"{timedelta_formatter(cardioActivities_df['Average Pace'].min())} min/Km"
        )
        st.metric(label="Accumulated time", value=f"{cardioActivities_df['Duration'].sum()}")

    with col3:
        st.metric(
            label="Weight",
            value=f"{measurements_df['Value'].values[-1]} Kg",
            delta=f"{measurements_df['Value'].values[-1] - measurements_df['Value'].values[0]} Kg",
            delta_color="inverse",
        )

    st.header("General Analysis")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=cardioActivities_df["Date"],
            y=cardioActivities_df["Average Pace"].apply(
                lambda x: datetime.datetime.combine(datetime.datetime.today().date(),
                                                    datetime.datetime.min.time()) + x),
            marker=dict(
                size=cardioActivities_df["Climb (m)"] / 10,
                color=cardioActivities_df["Distance (km)"],
                colorbar=dict(
                    title="Distance (km)",
                ),
                colorbar_x=-0.3,
                colorscale="Viridis"
            ),
            mode="markers",
            name="Average Pace",
            hovertemplate=
            "<b>Average Pace (min/km)</b>: %{y}" +
            "<br>%{text}" +
            "<br><b>Date</b>: %{x}<br>",
            text=[
                f"<b>Distance (km)</b>: {row['Distance (km)']}<br><b>Climb (m)</b>: {row['Climb (m)']}<br><b>Duration</b>: {row['Duration']}<br><b>Calories Burned</b>: {row['Calories Burned']}"
                for index, row in cardioActivities_df.iterrows()]
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=measurements_df["Date"],
            y=measurements_df["Value"],
            mode="lines+markers",
            name="Weight",
            line=go.scatter.Line(color="gray"),
            hovertemplate=
            "<b>Weight (kg)</b>: %{y:.2f}" +
            "<br><b>Date</b>: %{x}<br>",
        ),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Average Pace (size based on Climb) and Weight evolution"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="Average Pace (min/Km)", secondary_y=False)
    fig.update_yaxes(title_text="Weight (Kg)", range=[50, 100], secondary_y=True)

    fig.update_layout(
        yaxis_tickformat="%M:%S"
    )

    st.plotly_chart(fig, use_container_width=True)

    # fig = px.line(
    #     cardioActivities_df,
    #     x="Date",
    #     y="Total Distance (km)",
    #     markers=True,
    # )
    # st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        km_df,
        x="Date",
        y="Average Pace",
        markers=True,
        color="Km",
        title="Average Pace per Km evolution"
    )

    fig.update_layout(
        yaxis_tickformat="%M:%S"
    )
    st.plotly_chart(fig, use_container_width=True)

    # fig = px.scatter(
    #     gpx_df.sort_values(["Date", "ElevationDiff"]),
    #     x="ElevationDiff",
    #     y="AveragePace",
    #     color="Date",
    # )
    # st.plotly_chart(fig, use_container_width=True)
    #
    # fig = px.scatter(
    #     gpx_df.sort_values(["Date", "ElevationDiff"]),
    #     x="Date",
    #     y="AveragePace",
    #     color="ElevationDiff",
    # )
    # st.plotly_chart(fig, use_container_width=True)
    #
    # fig = px.violin(
    #     gpx_df.sort_values(["Date", "ElevationDiff"]),
    #     x="ElevationDiff",
    #     y="AveragePace",
    # )
    # st.plotly_chart(fig, use_container_width=True)

    st.header("Race Analysis")

    race_date = st.selectbox(label="Date", options=[""] + gpx_df["Date"].unique().tolist())

    if not race_date:
        return

    # mask = cardioActivities_df["Date"] == race_date
    # cardioActivities_df = cardioActivities_df[mask].reset_index(drop=True)
    # mask = measurements_df["Date"] == race_date
    # measurements_df = measurements_df[mask].reset_index(drop=True)
    mask = gpx_df["Date"] == race_date
    gpx_df = gpx_df[mask].reset_index(drop=True)
    mask = km_df["Date"] == race_date
    km_df = km_df[mask].reset_index(drop=True)

    # # st.write(cardioActivities_df)

    col1, col2 = st.columns(2)

    with col1:
        # fig = px.line_mapbox(
        #     gpx_df,
        #     lat="Latitude",
        #     lon="Longitude",
        #     #zoom=3,
        #     #height=300
        # )
        # st.plotly_chart(fig, use_container_width=True)
        map = folium.Map(
            location=[
                km_df["Points"].apply(lambda x: x[0]).mean(),
                km_df["Points"].apply(lambda x: x[1]).mean()
            ],
            zoom_start=14
        )

        points = gpx_df[["Latitude", "Longitude"]].apply(lambda x: (x.Latitude, x.Longitude), axis=1).tolist()
        # add a markers
        folium.Marker(
            points[0],
            icon=DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f'''
                    <div style="
                        background-color: #007bff;
                        color: #fff;
                        border-radius: 50%;
                        padding: 6px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 14px;
                        width: 30px;
                        height: 30px;
                        line-height: 30px;
                    ">Start</div>
                '''
            )
        ).add_to(map)

        for index, row in km_df.iterrows():
            folium.Marker(
                row["Points"],
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html=f'''
                    <div style="
                        background-color: #007bff;
                        color: #fff;
                        border-radius: 50%;
                        padding: 6px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 14px;
                        width: 30px;
                        height: 30px;
                        line-height: 30px;
                    ">{row["Km"]}</div>
                '''
                )
            ).add_to(map)

        folium.Marker(
            points[-1],
            icon=DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f'''
                    <div style="
                        background-color: #007bff;
                        color: #fff;
                        border-radius: 50%;
                        padding: 6px;
                        text-align: center;
                        font-weight: bold;
                        font-size: 14px;
                        width: 30px;
                        height: 30px;
                        line-height: 30px;
                    ">Finish</div>
                '''
            )
        ).add_to(map)

        # fadd lines
        folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(map)
        folium_static(map, width=350, height=450)

        fig = px.line(
            gpx_df,
            x="DistanceAcc",
            y="Elevation",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            km_df,
            x="Km",
            y="Average Pace",
            markers=True,
        )
        fig.update_layout(
            yaxis_tickformat="%M:%S"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            gpx_df,
            x="ElevationDiff",
            y="Average Pace",
        )
        fig.update_layout(
            yaxis_tickformat="%M:%S"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
