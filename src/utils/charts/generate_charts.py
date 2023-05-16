import os
import warnings
import inspect
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

from ..data_preparation.load_data import load_flights, load_airports
from .helpers import save_fig, finish
from .constants import REQUIRE, MONTHS, WEEK_DAYS, PLOTS_DIR


plt.set_loglevel("WARNING")
sns.set_style("whitegrid")
warnings.simplefilter(action="ignore")
np.random.seed(42)


def generate_charts(years: str | list = "all", dir: str = None):
    """
    Function that wraps all eda_Pawel code and generates its charts
    for given year. Charts are saved to dir.

    :param years: choice of years that will be passed to utils.load_flights()
    :param dir: directory to save charts. If None, the chart will be saved to "plots/{{year}}"
    """
    if dir is None:
        if isinstance(years, str):
            dir = os.path.join(PLOTS_DIR, "plots", years)
        else:
            dir = os.path.join(PLOTS_DIR, "plots", "_".join(years))
    os.makedirs(dir, exist_ok=True)

    flights = load_flights(years, cols=REQUIRE)

    for item in list(globals().keys()):
        if item.startswith("chart_"):
            globals()[item](flights, dir)


def chart_1(flights: pd.DataFrame, dir: str):
    """ "Total Planned Flight Time for each Carrier" chart"""
    title = "Total Planned Flight Time for each Carrier"

    dt = flights.groupby("UniqueCarrier")["CRSElapsedTime"].sum()
    dt = np.c_[dt.index, dt / (60 * 1000)]
    dt = pd.DataFrame(
        dt, columns=["UniqueCarrier", "Total CRSElapsedTime [hours * 10^3]"]
    )
    dt = dt.sort_values(
        by="Total CRSElapsedTime [hours * 10^3]", axis=0, ascending=False
    ).reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = 6 * np.ceil(len(dt["UniqueCarrier"].unique()) / 20)
    plt.figure(figsize=(w, 6))

    ax = sns.barplot(
        dt,
        x="UniqueCarrier",
        y="Total CRSElapsedTime [hours * 10^3]",
        color="lightblue",
    )
    for i in range(len(dt)):
        ax.text(
            i,
            10,
            round(dt.loc[i, "Total CRSElapsedTime [hours * 10^3]"], 1),
            color="#0f540f",
            ha="center",
            rotation=90,
            size=8,
        )

    finish(ax, title, plot=False, dir=dir)


def chart_2(flights: pd.DataFrame, dir: str):
    """ "Max Departure and Arrival Delay for each Carrier" chart"""
    title = "Max Departure and Arrival Delay for each Carrier"

    dt = flights.groupby("UniqueCarrier")[["DepDelay", "ArrDelay"]].max()
    dt_DepDelay = np.c_[
        dt.index, dt["DepDelay"] / 60, np.full(dt.index.shape, "DepDelay")
    ]
    dt_ArrDelay = np.c_[
        dt.index, dt["ArrDelay"] / 60, np.full(dt.index.shape, "ArrDelay")
    ]
    dt = np.r_[dt_ArrDelay, dt_DepDelay]
    dt = pd.DataFrame(dt, columns=["UniqueCarrier", "Delay [hours]", "Type"])
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = 6 * np.ceil(len(dt["UniqueCarrier"].unique()) / 20)
    plt.figure(figsize=(w, 6))

    ax = sns.barplot(
        dt,
        x="UniqueCarrier",
        y="Delay [hours]",
        hue="Type",
        palette=sns.color_palette("ch:s=.25,rot=-.25", 2),
    )

    finish(ax, title, plot=False, dir=dir)


def chart_3(flights: pd.DataFrame, dir: str):
    """ "Number of Aircrafts in fleet of each Carrier" chart"""
    title = "Number of Aircrafts in fleet of each Carrier"

    dt1 = flights.groupby(["UniqueCarrier"])["TailNum"].unique()
    dt2 = [len(dt1[i][~pd.isna(dt1[i])]) for i in range(len(dt1))]
    dt = np.c_[dt1.index, dt2]
    dt = pd.DataFrame(dt, columns=["UniqueCarrier", "Known Airplanes Count"])
    dt = dt.sort_values(
        by="Known Airplanes Count", axis=0, ascending=False
    ).reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = 6 * np.ceil(len(dt["UniqueCarrier"].unique()) / 20)
    plt.figure(figsize=(w, 6))

    ax = sns.barplot(
        dt, x="UniqueCarrier", y="Known Airplanes Count", color="lightblue"
    )
    for i in range(len(dt)):
        ax.text(
            i,
            15,
            round(dt.loc[i, "Known Airplanes Count"], 1),
            color="#0f540f",
            ha="center",
            rotation=90,
            size=8,
        )

    finish(ax, title, plot=False, dir=dir)


def chart_4(flights: pd.DataFrame, dir: str):
    """ "Cancelation Rate for each Carrier" chart"""
    title = "Cancelation Rate for each Carrier"

    dt1 = (
        flights[~(flights["Cancelled"] == 0)]
        .groupby(["UniqueCarrier"])["Cancelled"]
        .count()
    )
    dt2 = (
        flights[~(flights["Cancelled"].isna())]
        .groupby(["UniqueCarrier"])["Cancelled"]
        .count()
    )
    dt = np.c_[dt1.index, dt1, dt2, dt1 / dt2 * 100]
    dt = pd.DataFrame(
        dt,
        columns=[
            "UniqueCarrier",
            "Cancelled flights",
            "All flights",
            "Cancelation Rate [%]",
        ],
    )
    dt = dt.sort_values(
        by="Cancelation Rate [%]", axis=0, ascending=False
    ).reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = 6 * np.ceil(len(dt["UniqueCarrier"].unique()) / 20)
    plt.figure(figsize=(w, 6))

    ax = sns.barplot(dt, x="UniqueCarrier", y="Cancelation Rate [%]", color="lightblue")
    for i in range(len(dt)):
        ax.text(
            i,
            0.1,
            round(dt.loc[i, "Cancelation Rate [%]"], 1),
            color="#0f540f",
            ha="center",
            rotation=90,
            size=8,
        )

    finish(ax, title, plot=False, dir=dir)


def chart_5(flights: pd.DataFrame, dir: str):
    """ "Cancelation Causes" chart"""
    title = "Cancelation Causes"

    dt = (
        flights[~(flights["Cancelled"] == 0)]
        .groupby(["CancellationCode"])["CancellationCode"]
        .count()
    )
    dt.name = "Number"
    dt = pd.DataFrame(dt).reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(dt, x="CancellationCode", y="Number", color="lightblue")
    for i in range(len(dt)):
        ax.text(
            i, 200, round(dt.loc[i, "Number"], 1), color="#0f540f", ha="center", size=10
        )
    plt.xticks([0, 1, 2, 3], ["przewoźnik", "pogoda", "NAS", "bezpieczeńtwo"])
    finish(ax, title, plot=False, dir=dir)


def chart_6(flights: pd.DataFrame, dir: str):
    """ "Planned Flights over Time" chart (x2)"""
    title = "Planned Flights over Time"
    bins = [0, 1, 2, 3, 4, 5, 6, np.inf]

    dt = flights.groupby([flights["Arrival"].dt.date, "DayOfWeek"])["DayOfWeek"].count()
    dt.name = "Number of flights"
    dt = pd.DataFrame(dt).reset_index()
    dt["DayOfWeek"] = pd.cut(dt["DayOfWeek"], bins, labels=WEEK_DAYS)
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = int(np.ceil(len(dt["Arrival"].unique()) / 300))
    dt["Arrival"] = dt["Arrival"].astype(str)
    xmin, xmax = dt["Arrival"].min(), dt["Arrival"].max()
    dt["Smoothed"] = uniform_filter1d(dt["Number of flights"], w * 10)

    for nn in range(2):
        plt.figure(figsize=(20, 6))

        if nn == 0:
            sns.lineplot(
                dt, x="Arrival", y="Number of flights", color="red", alpha=0.05
            )
            sns.lineplot(
                dt,
                x="Arrival",
                y="Smoothed",
                color="red",
                alpha=0.5,
                label="Average number of flights",
            )
            ax = sns.pointplot(
                dt,
                x="Arrival",
                y="Number of flights",
                hue="DayOfWeek",
                scale=0.3,
                palette=sns.color_palette("husl", 7),
            )
        else:
            ax = sns.lineplot(
                dt,
                x="Arrival",
                y="Number of flights",
                hue="DayOfWeek",
                palette=sns.color_palette("husl", 7),
            )

        lgd = plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Day of the week",
            title_fontsize="x-large",
        )

        for handle in lgd.legendHandles:
            handle._sizes = [50]

        i = 5 * w - 1
        for label in ax.xaxis.get_ticklabels():
            i += 1
            if i == 5 * w:
                i = 0
                continue
            label.set_visible(False)

        plt.xticks(rotation=45)
        ax.xaxis.grid(False)
        sns.despine()
        plt.xlim([xmin, xmax])
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        plt.title(title, size=30)
        save_fig(f"{title}_{nn}", dir, dpi=max(min(w * 25, 400), 200))


def chart_7(flights: pd.DataFrame, dir: str):
    """ "Most popular routes" chart"""
    title = "Most popular routes"

    dt = (
        flights.groupby(["Origin", "Dest", "Cancelled"])["TailNum"]
        .count()
        .reset_index()
    )

    # treat flights from ABE to ATL and from ATL to ABE as same route
    dt1 = dt[dt["Origin"].astype("U3") < dt["Dest"].astype("U3")]
    dt2 = dt[dt["Origin"].astype("U3") > dt["Dest"].astype("U3")]
    dt2 = dt2.rename(
        columns={"Origin": "Dest", "Dest": "Origin", "TailNum": "Number of flights"}
    )
    dt = pd.concat([dt1, dt2])
    dt["Route"] = dt["Origin"].astype("U3") + " - " + dt["Dest"].astype("U3")

    dt1 = dt.groupby(["Route", "Cancelled"])["Number of flights"].sum().reset_index()
    dt2 = (
        dt1.groupby(["Route"])["Number of flights"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()[:15]
    )
    keep = dt1["Route"].isin(dt2["Route"])
    dt = dt1[keep].reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        dt,
        x="Route",
        y="Number of flights",
        hue="Cancelled",
        palette=["lightgreen", "#fa6666"],
    )

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title_fontsize="x-large",
        title="Cancelled",
    )
    plt.xticks(rotation=45)
    finish(ax, title, plot=False, dir=dir)


def chart_8(flights: pd.DataFrame, dir: str):
    """ "Total Delay Time for Each Month" and "Delay Coefficient for Each Month" charts"""
    title = "Total Delay Time for Each Month"

    dt3 = flights.groupby(flights["Departure"].dt.month)["DepDelay"].count() / 1000
    dt3.name = "All flights [x1000]"
    dt3 = dt3.reset_index()

    dt1 = flights.groupby(flights["Departure"].dt.month)["ArrDelay"].sum()
    dt1.name = "Delay [hr * 10^3]"
    dt1 = dt1.reset_index()
    dt1["type"] = "Arrival"
    dt1 = pd.merge(dt1, dt3)

    dt2 = flights.groupby(flights["Departure"].dt.month)["DepDelay"].sum()
    dt2.name = "Delay [hr * 10^3]"
    dt2 = dt2.reset_index()
    dt2["type"] = "Departure"
    dt2 = pd.merge(dt2, dt3)

    dt = pd.merge(dt1, dt2, how="outer")
    dt["Delay [hr * 10^3]"] = dt["Delay [hr * 10^3]"] / 60000
    dt.rename(columns={"Departure": "Month"}, errors="raise", inplace=True)
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    sns.barplot(
        dt,
        x="Month",
        y="Delay [hr * 10^3]",
        hue="type",
        palette=sns.color_palette("ch:s=.25,rot=-.25", 2),
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend_.remove()

    ax2 = ax1.twinx()
    sns.pointplot(
        dt,
        x="Month",
        y="All flights [x1000]",
        color="red",
        linestyles="--",
        label="Number of flights",
    )
    ax2.yaxis.label.set_size(13)
    ax2.set_yticklabels(ax2.get_yticks().astype(np.int64), size=9)
    ax2.grid(False)
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
        title_fontsize="x-large",
        title="Delay Type",
    )

    pick = [int(text.get_text()) - 1 for text in ax2.get_xticklabels()]
    plt.xticks(list(range(len(pick))), MONTHS[pick])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)  # rotate xticks
    finish(ax1, title, plot=False, dir=dir)

    """ "Delay Coefficient for Each Month" chart"""
    title = "Delay Coefficient for Each Month"

    dt["Coefficient"] = dt["Delay [hr * 10^3]"] / dt["All flights [x1000]"]
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    fig = plt.figure()
    ax = sns.barplot(
        dt,
        x="Month",
        y="Coefficient",
        hue="type",
        palette=sns.color_palette("ch:s=.25,rot=-.25", 2),
    )

    pick = [int(text.get_text()) - 1 for text in ax.get_xticklabels()]
    plt.xticks(list(range(len(pick))), MONTHS[pick], rotation=45)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title_fontsize="x-large",
        title="Delay Type",
    )
    sns.despine()
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    plt.title(title, size=20)
    # so the legend is put inside the box
    save_fig(title, dir, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")


def chart_9(flights: pd.DataFrame, dir: str):
    """ "Number of Departures over hours" chart"""
    title = "Number of flights over hours"

    dts = []
    for name in ["Departure", "Arrival"]:
        dt = (
            flights[~pd.isna(flights[name])][name].dt.hour.value_counts().sort_index()
            / 1000
        ).to_frame(name=name + "s")
        dt.index.name = "Hour"
        dt.reset_index(inplace=True)
        dts.append(dt)
    dt = pd.merge(*dts)
    dt.index = dt["Hour"]
    del dt["Hour"]

    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    ax = dt.plot(
        kind="bar", stacked=True, color=sns.color_palette("ch:s=.25,rot=-.25", 2)
    )

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title_fontsize="x-large",
        title="",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Number of flights [x1000]")
    ax.xaxis.grid(False)
    finish(ax, title, plot=False, dir=dir)


def chart_10(flights: pd.DataFrame, dir: str):
    """ "Flights Count for each Carrier" chart"""
    title = "Flights Count for each Carrier chart"

    dt = flights["UniqueCarrier"].value_counts()
    dt = np.c_[dt.index, dt / (60 * 1000)]
    dt = pd.DataFrame(dt, columns=["UniqueCarrier", "Number of flights [* 10^3]"])
    dt = dt.sort_values(
        by="Number of flights [* 10^3]", axis=0, ascending=False
    ).reset_index()
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = 6 * np.ceil(len(dt["UniqueCarrier"].unique()) / 20)
    plt.figure(figsize=(w, 6))

    ax = sns.barplot(
        dt,
        x="UniqueCarrier",
        y="Number of flights [* 10^3]",
        color="lightblue",
    )
    for i in range(len(dt)):
        ax.text(
            i,
            0.5,
            round(dt.loc[i, "Number of flights [* 10^3]"], 1),
            color="#0f540f",
            ha="center",
            rotation=90,
            size=8,
        )

    finish(ax, title, plot=False, dir=dir)


def chart_11(flights: pd.DataFrame, dir: str):
    """ "20 most popular Airports" and ""Airports and their popularity" charts"""
    title = "10 most popular Airports"

    dt1 = flights["Dest"].value_counts()
    dt2 = flights["Origin"].value_counts()
    dt1 = pd.DataFrame(
        np.c_[dt1.index, dt1 / 1000], columns=["Airport", "Number of flights [* 10^3]"]
    )
    dt2 = pd.DataFrame(
        np.c_[dt2.index, dt2 / 1000], columns=["Airport", "Number of flights [* 10^3]"]
    )
    dt1["Type"] = "Arrivals"
    dt2["Type"] = "Departures"

    dt = pd.merge(dt1, dt2, how="outer")
    dt.fillna(0.0, inplace=True)

    combined = dt.groupby(["Airport"])["Number of flights [* 10^3]"].sum().reset_index()
    combined.rename(columns={"Number of flights [* 10^3]": "Combined"}, inplace=True)
    dt = pd.merge(dt, combined).sort_values("Combined", ascending=False)
    dt_copy = dt.copy()

    if dt.shape[0] > 20:
        dt = dt.iloc[:20, :]
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        dt,
        x="Airport",
        y="Number of flights [* 10^3]",
        hue="Type",
        color="lightblue",
        palette=sns.color_palette("ch:s=.25,rot=-.25", 2),
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title_fontsize="x-large",
        title="Connection Type",
    )
    sns.despine()
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    plt.title(title, size=20)

    save_fig(title, dir, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight")

    """ "Airports and their popularity" chart """
    title = "Airports and their popularity"

    airports = load_airports()
    dt = pd.merge(dt_copy, airports, left_on="Airport", right_on="iata")
    dt.drop_duplicates(subset=["Airport"], inplace=True)

    fig, ax = plt.subplots(figsize=(15, 15))

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    usa = world[world["name"] == "United States of America"]
    usa.plot(ax=ax, color="white", edgecolor="black")

    points = gpd.GeoDataFrame(dt, geometry=gpd.points_from_xy(dt["long"], dt["lat"]))

    w = int(np.ceil(points["Combined"].max() / 300))
    points.plot(
        ax=ax,
        markersize=points["Combined"].astype(np.float32) / w,
        color="red",
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # set only mainland for visibility
    ax.set_xlim([-130, -65])
    ax.set_ylim([24, 50])

    i = 0
    for x, y, label in zip(points["long"], points["lat"], points["city"]):
        ax.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points")
        i += 1
        if i == 20:
            break

    finish(ax, title, plot=False, dir=dir)


def main():
    generate_charts(["1989", "2007"])
    generate_charts(["2000", "2001", "2002"])
    generate_charts(["1990", "1995", "2000", "2005"])


if __name__ == "__main__":
    main()
