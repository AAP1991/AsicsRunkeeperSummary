import streamlit as st
import os
from zipfile import ZipFile
import datetime

import pandas as pd
import gpxpy
import geopy.distance

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from itertools import cycle


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
    cardioActivities_df["Average Pace"] = \
        cardioActivities_df["Average Pace"].apply(lambda x: pd.to_timedelta(60 * int(x.split(":")[0]) + int(x.split(":")[1]), unit="seconds").round("s"))
    cardioActivities_df["Total Distance (km)"] = cardioActivities_df["Distance (km)"].cumsum()

    mask = measurements_df.Type == "weight"
    measurements_df = measurements_df[mask].reset_index(drop=True)

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
            [
                (t / 60.0) / (m / 1000.0) if m > 0 else 0 for t, m in
                zip(aggregation_dict["TimeDiff"], aggregation_dict["Distance"])
            ]

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

    def custom_aggregation(df):
        aggregation_dict = {}

        km_list = []
        average_pace_list = []
        elevation_list = []
        last_distance = None
        last_time = None
        last_elevation = None

        for km in range(1, int(df["DistanceAcc"].max() / 1000.0) + 1):
            mask = df["DistanceAcc"] / 1000.0 >= km
            aux_df = df[mask].sort_values(["TimeAcc"]).head(1)
            distance = aux_df["DistanceAcc"].values[0]
            time = aux_df["TimeAcc"].values[0]
            elevation = aux_df["Elevation"].values[0]
            if last_distance is None:
                average_pace = (time / 60.0) / (distance / 1000.0)
                elevation_diff = elevation - df.sort_values(["TimeAcc"])["Elevation"].values[0]
            else:
                average_pace = ((last_time - time) / 60.0) / ((last_distance - distance) / 1000.0)
                elevation_diff = elevation - last_elevation

            km_list.append(km)
            average_pace_list.append(pd.to_timedelta(average_pace, unit="minutes").round("s"))
            elevation_list.append(int(round(elevation_diff)))

            last_distance = distance
            last_time = time
            last_elevation = elevation

        aggregation_dict["Date"] = [df["Date"].values[0]] * len(km_list)

        aggregation_dict["Km"] = km_list
        aggregation_dict["AveragePace"] = average_pace_list
        aggregation_dict["Elevation"] = elevation_list

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

    st.header("Plots Section")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=cardioActivities_df["Date"],
            y=cardioActivities_df["Average Pace"].apply(lambda x: datetime.datetime.combine(datetime.datetime.today().date(), datetime.datetime.min.time()) + x),
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
    fig.update_yaxes(title_text="Average Pace (min/km)", secondary_y=False)
    fig.update_yaxes(title_text="Weight (kg)", range=[50, 100], secondary_y=True)

    fig.update_layout(
        yaxis_tickformat = "%M:%S"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        cardioActivities_df,
        x="Date",
        y="Total Distance (km)",
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    km_df["Average Pace"] = km_df['AveragePace'].apply(lambda x: datetime.datetime.combine(datetime.datetime.today().date(), datetime.datetime.min.time()) + x)

    fig = px.line(km_df, x="Km", y="Average Pace", markers=True, color="Date")

    fig.update_layout(
        yaxis_tickformat="%M:%S"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        gpx_df.sort_values(["Date", "ElevationDiff"]),
        x="ElevationDiff",
        y="AveragePace",
        color="Date",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
