import os
import warnings
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

from .load_data import load_flights
from .eda import save_fig, finish

sns.set_style("whitegrid")
warnings.simplefilter(action="ignore")
np.random.seed(42)

MONTHS = np.array(
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
)


def generate_charts(years: str | list, dir: str = None):
    """
    Function that wraps all eda_Pawel code and generates its charts
    for given year. Charts are saved to dir.

    :param years: choice of years that will be passed to utils.load_flights()
    :param dir: directory to save charts. If None, the chart will be saved to "plots/{{year}}"
    """
    if dir is None:
        dir = os.path.join("plots", "_".join(years))
    os.makedirs(dir, exist_ok=True)

    flights = load_flights(years)

    to_iter = list(globals().keys())
    for item in to_iter:
        if item.startswith("chart_") and not item.startswith("chart_8_1"):
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
    """ "Planned Flights over Time" chart"""
    title = "Planned Flights over Time"

    labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    bins = [0, 1, 2, 3, 4, 5, 6, np.inf]

    dt = flights.groupby(["Year", "Month", "DayofMonth", "DayOfWeek"])[
        "DayOfWeek"
    ].count()
    dt.name = "Number of flights"
    dt = pd.DataFrame(dt).reset_index()
    dt["Date"] = (
        dt["Year"].astype(str)
        + "/"
        + dt["Month"].astype(str)
        + "/"
        + dt["DayofMonth"].astype(str)
    )
    dt["DayOfWeek"] = pd.cut(dt["DayOfWeek"], bins, labels=labels)
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    w = np.ceil(len(dt["Date"].unique()) / 300)
    plt.figure(figsize=(w * 20, 6 * np.ceil(w / 2)))

    plt.figure(figsize=(20, 6))
    ax = sns.lineplot(dt, x="Date", y="Number of flights", color="red", alpha=0.3)
    ax = sns.pointplot(
        dt,
        x="Date",
        y="Number of flights",
        hue="DayOfWeek",
        scale=max(0.8 / w, 0.1),
        palette=sns.color_palette("husl", 7),
    )
    i = 5 * 2 - 1
    for label in ax.xaxis.get_ticklabels():
        i += 1
        if i == 5 * w:
            i = 0
            continue
        label.set_visible(False)

    lgd = plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Day of the week",
        title_fontsize="x-large",
    )
    for handle in lgd.legendHandles:
        handle._sizes = [50]
    plt.xticks(rotation=45)
    sns.despine()
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    plt.title(title, size=30)
    save_fig(title, dir, dpi=w * 75)


def chart_7(flights: pd.DataFrame, dir: str):
    """ "Most popular routes" chart"""
    title = "Most popular routes"

    dt = (
        flights.groupby(["Origin", "Dest", "Cancelled"])["FlightNum"]
        .count()
        .reset_index()
    )

    # treat flights from ABE to ATL and from ATL to ABE as same route
    dt1 = dt[dt["Origin"] < dt["Dest"]]
    dt2 = dt[dt["Origin"] > dt["Dest"]]
    # assert len(dt1) + len(dt2) == len(dt)
    dt2 = dt2.rename(
        columns={"Origin": "Dest", "Dest": "Origin", "FlightNum": "Number of flights"}
    )
    dt = pd.concat([dt1, dt2])
    dt["Route"] = dt["Origin"] + " - " + dt["Dest"]

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
    """ "Total Delay Time for Each Month" chart"""
    title = "Total Delay Time for Each Month"

    dt3 = flights.groupby(["Month"])["DepDelay"].count() / 1000
    dt3.name = "All flights [x1000]"
    dt3 = dt3.reset_index()

    dt1 = flights.groupby(["Month"])["ArrDelay"].sum()
    dt1.name = "Delay [hr * 10^3]"
    dt1 = dt1.reset_index()
    dt1["type"] = "Arrival"
    dt1 = pd.merge(dt1, dt3)

    dt2 = flights.groupby(["Month"])["DepDelay"].sum()
    dt2.name = "Delay [hr * 10^3]"
    dt2 = dt2.reset_index()
    dt2["type"] = "Departure"
    dt2 = pd.merge(dt2, dt3)

    dt = pd.merge(dt1, dt2, how="outer")
    dt["Delay [hr * 10^3]"] = dt["Delay [hr * 10^3]"] / 60000
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

    chart_8_1(dt, dir)


def chart_8_1(dt: pd.DataFrame, dir: str):
    """ "Delay Coefficient for Each Month" chart"""
    title = "Delay Coefficient for Each Month"

    dt["Coefficient"] = dt["Delay [hr * 10^3]"] / dt["All flights [x1000]"]
    if dt.empty:
        warnings.warn(f"Empty final data set: {inspect.currentframe().f_code.co_name}")
        return  # all values were nan

    ax = sns.barplot(
        dt,
        x="Month",
        y="Coefficient",
        hue="type",
        palette=sns.color_palette("ch:s=.25,rot=-.25", 2),
    )

    pick = [int(text.get_text()) - 1 for text in ax.get_xticklabels()]
    plt.xticks(list(range(len(pick))), MONTHS[pick], rotation=45)

    plt.legend(
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
    title = "Number of Departures over hours"

    dts = []
    for name in ["DepTime", "ArrTime"]:
        dt = (
            flights[~pd.isna(flights[name])][name]
            .astype(np.int16)
            .astype(str)
            .str.zfill(4)
        )

        # dt[dt == "2400"] = "0000"
        # somehow hhmm format exceeds 2400 in our date
        dt[dt >= "2400"] = (
            (dt[dt >= "2400"].astype(np.int16) - 2400).astype(str).str.zfill(4)
        )

        dt = pd.to_datetime(dt, format="%H%M").dt.strftime("%H")
        dt = dt.reset_index().groupby(name)["index"].count().sort_values().reset_index()
        dt["index"] /= 1000
        dt = dt.rename(columns={name: "Hour", "index": name})
        dts.append(dt)
    dt = pd.merge(*dts)

    dt = dt.sort_values("Hour")
    dt.index = dt["Hour"]
    dt = dt.rename(columns={"DepTime": "Departures", "ArrTime": "Arrivals"})
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
        title="Delay Type",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Number of flights [x1000]")
    ax.xaxis.grid(False)
    finish(ax, title, plot=False, dir=dir)


if __name__ == "__main__":
    generate_charts(["1989", "2007"])
    generate_charts(["2000", "2001", "2002"])
    generate_charts(["1990", "1995", "2000", "2005"])
