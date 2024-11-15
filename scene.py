"""
The Scene class

by
Gabriel Mesquida Masana
gabmm@stanford.edu

"""

import copy
from flight import Flight as Fl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Scene:
    """
    The Scene class
    ================
    Version: 1.00
    Last update: 15/11/24
    -----------------------
    Gabriel Mesquida Masana
    gabmm@stanford.edu

    A Scene object contains:

    * flights: a list of Flight objects
    * notes: a string of the manipulations done to the Scene

    Methods:
        * __init__
            - from_file
            - from_scene
            - from_flights_and_notes
        * __repr__

    for filtering flights
        * arrivals
        * departures
        * overflights
        * longer
        * split_c
        * remove
        * split_train_dev_test
        * aircraft

    for filtering tracks in flights
    (always makes deepcopy of modified flights)
        * descent
        * climb (refreshes alt_max)
        * higher_than
        * thin_tracks

    for querying scene
        * alt_max (option to refresh)
        * alt_min
        * c_max_max (first flight, second scene)
        * c_max_min
        * query_attribute (can produce pie chart)

    for rendering
        * plot
        * plot_elevation_profile
        * plot_c (histogram)
        * plot_alt_max (density)


    """

    def __init__(
        self,
        from_file: str = None,
        from_scene=None,
        from_flights_and_notes=None,
    ):
        """
        Creating objects:
            - from_file
            - from_scene
            - from_flights_and_notes
        """

        # Declaring all fields in __init__
        self.flights = None
        self.notes = None

        # If loading
        if from_file:
            Fl.load_dataframe(file_name=from_file)
            self.flights = Fl.create_flights()
            self.notes = f"loaded from {from_file}"

        # From scene
        elif from_scene:
            self.flights = copy.copy(from_scene.flights)
            self.notes = f"copy of object ({from_scene.notes})"

        # From flights and notes
        elif from_flights_and_notes:
            self.flights = copy.copy(from_flights_and_notes[0])
            self.notes = from_flights_and_notes[1]

    def arrivals(self, arriving_from: str = None):
        """
        Filters for arrivals
        """
        self.flights = list(
            filter(
                lambda flight: (
                    flight.ades == "WSSS"
                    if not arriving_from
                    else flight.ades == "WSSS" and flight.adep == arriving_from
                ),
                self.flights,
            )
        )
        self.notes += (
            ", arriving to WSSS"
            if not arriving_from
            else f", arriving to WSSS from {arriving_from}"
        )
        return self

    def departures(self, departing_to: str = None, clean_altitudes=False):
        """
        Filters for departures
        """
        self.flights = list(
            filter(
                lambda flight: (
                    flight.adep == "WSSS"
                    if not departing_to
                    else flight.adep == "WSSS" and flight.ades == departing_to
                ),
                self.flights,
            )
        )
        self.notes += (
            ", departing WSSS"
            if not departing_to
            else f", departing WSSS to {departing_to}"
        )

        if clean_altitudes == True:
            filtered = list(
                filter(
                    lambda flight: flight.tracks.iloc[0].Altitude < 500,
                    self.flights,
                )
            )
            if len(filtered) != len(self.flights):
                print(f"Removed {len(self.flights)-len(filtered)} flights")
                self.flights = filtered
                self.notes += (
                    f", removed {len(self.flights)-len(filtered)} flights"
                )

        return self

    def overflights(self):
        """
        Filters for overflights
        """
        self.flights = list(
            filter(
                lambda flight: flight.adep != "WSSS" and flight.ades != "WSSS",
                self.flights,
            )
        )
        self.notes += ", overflights"
        return self

    def longer(self, tracks=75):
        """
        Filters for minimum number of tracks
        """
        self.flights = list(
            filter(
                lambda flight: (len(flight.tracks) > tracks),
                self.flights,
            )
        )
        self.notes += f", with min {tracks} tracks"
        return self

    def aircraft(self, aircraft="A320"):
        """
        Filters for overflights
        """
        self.flights = list(
            filter(
                lambda flight: flight.aircraft == aircraft,
                self.flights,
            )
        )
        self.notes += f", aircraft is {aircraft}"
        return self

    def descent(self):
        """
        Filters the tracks for descent segment only
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            # 20 seconds same level
            f_.tracks["dA"] = f_.tracks["Altitude"].diff(-6)

            # block that is going down
            flat = f_.tracks[f_.tracks.dA > 40]
            if len(flat) > 0:
                f_.tracks = f_.tracks.iloc[flat.iloc[0].name + 1 :]
            f_.tracks.drop(columns=["dA"], inplace=True)
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += ", descent only"
        return self

    def thin_tracks(self, factor=2):
        """
        Thinning tracks by a factor
        Tracks will be every 4xfactor seconds
        """
        thin_flights = copy.deepcopy(self.flights)
        for f_ in thin_flights:
            f_.tracks = f_.tracks[::factor]
            f_.numtracks = len(f_.tracks)
            f_.additional["alt_max"] = f_.tracks["Altitude"].max()
        self.flights = thin_flights
        self.notes += f", thinned to every {4*factor}s"
        return self

    def climb(self):
        """
        Filters the tracks for climb segment only
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            # 20 seconds same level
            f_.tracks["dA"] = f_.tracks["Altitude"].diff(-6)

            # block that is flat or going down
            flat = f_.tracks[f_.tracks.dA >= 0]
            if len(flat) > 0:
                f_.tracks = f_.tracks.iloc[0 : flat.iloc[0].name + 10]
            # Clean
            f_.tracks.drop(columns=["dA"], inplace=True)
            f_.additional["alt_max"] = f_.tracks["Altitude"].max()
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += ", climb only"
        return self

    def higher_than(self, altitude):
        """
        Filters the tracks and removes segment lower than altitude threshold
        """
        cut_flights = copy.deepcopy(self.flights)
        for f_ in cut_flights:
            f_.tracks = f_.tracks[f_.tracks.Altitude >= altitude]
            f_.numtracks = len(f_.tracks)
        self.flights = cut_flights
        self.notes += f", higher than {altitude}"
        return self

    def alt_max(self, refresh=False):
        """
        Returns max altitude in scene
        """
        if refresh:
            for f_ in self.flights:
                f_.additional["alt_max"] = f_.tracks["Altitude"].max()

        return max(
            map(lambda flight: flight.additional["alt_max"], self.flights)
        )

    def plot_alt_max(self, clip=None):
        maxs = [flight.additional["alt_max"] for flight in self.flights]
        sns.kdeplot(maxs, bw=0.01, fill=True, clip=clip)
        plt.title(f"Alt Max for {len(self.flights)} flights")
        plt.yticks(None)
        plt.show()

    def alt_min(self):
        """
        Returns min altitude in scene
        """
        return min(
            map(
                lambda flight: min(flight.additional["alt_max"].to_list()),
                self.flights,
            )
        )

    def c_max_min(self):
        """
        Returns min c_max value in scene
        """
        return min(
            map(lambda flight: flight.additional["c_max"], self.flights)
        )

    def c_max_max(self):
        """
        Returns max c_max value in scene
        """
        return max(
            map(lambda flight: flight.additional["c_max"], self.flights)
        )

    def plot_c(self):
        """
        Plots c_max value density distribution from scene
        """
        cs = [flight.additional["c_max"] for flight in self.flights]
        clip = (self.c_max_min(), self.c_max_max())
        _, (ax1, ax2) = plt.subplots(2)
        sns.kdeplot(cs, bw=0.02, fill=True, ax=ax1, clip=clip)
        sns.boxplot(x=cs, ax=ax2)
        plt.suptitle(f"c_max distribution for {len(self.flights)} flights")
        plt.show()

    def split_c(self, c_threshold=2):
        """
        Split one scene into two based on c_max threshold value
        """
        flights_c_low = [
            flight
            for flight in self.flights
            if float(flight.additional["c_max"]) <= c_threshold
        ]
        flights_c_high = [
            flight
            for flight in self.flights
            if float(flight.additional["c_max"]) > c_threshold
        ]
        return (
            Scene(
                from_flights_and_notes=(
                    flights_c_low,
                    f"{self.notes}, c<={c_threshold}",
                )
            ),
            Scene(
                from_flights_and_notes=(
                    flights_c_high,
                    f"{self.notes}, c>{c_threshold}",
                )
            ),
        )

    def split_train_dev_test(self, trainpct=80, devpct=10):
        """
        Returns three different scenes for train, dev, and test
        """
        testpct = 100 - trainpct - devpct
        place = np.random.choice(
            3,
            len(self.flights),
            p=[trainpct / 100, devpct / 100, testpct / 100],
        )
        train, dev, test = ([], [], [])

        for n_, f_ in enumerate(self.flights):
            match place[n_]:
                case 0:
                    train.append(f_)
                case 1:
                    dev.append(f_)
                case 2:
                    test.append(f_)

        return (
            Scene(
                from_flights_and_notes=(
                    train,
                    f"{self.notes}, {trainpct}% for train",
                )
            ),
            Scene(
                from_flights_and_notes=(
                    dev,
                    f"{self.notes}, {devpct}% for dev",
                )
            ),
            Scene(
                from_flights_and_notes=(
                    test,
                    f"{self.notes}, {testpct}% for test",
                )
            ),
        )

    def remove(self, id):
        """
        Removes one or more flights from scene
        """
        new_list = []
        if isinstance(id, str):
            id = [id]

        for f_ in self.flights:
            if f_.flight_id in id:
                print(f"Removed {f_.flight_id}")
                self.notes = f"{self.notes}, removed {id}"
            else:
                new_list.append(f_)
        self.flights = new_list

    #
    # More specific queries
    #
    def query_attribute(
        self, attribute="aircraft", n=10, normalize=False, pie=False
    ):
        """
        Can retrieve values of an attribute or create pie chart
        Attributes allowed are ["Aircraft", "ADES", "ADEP", "Callsign"]
        """

        if attribute.lower() not in ["aircraft", "ades", "adep", "callsign"]:
            raise TypeError("This attribute is not allowed")

        airframes = pd.Series(
            [getattr(f_, attribute.lower()) for f_ in self.flights]
        )
        if pie:
            values = [
                str(val_[0]) + " (" + str(val_[1]) + ")"
                for val_ in zip(
                    airframes.value_counts().head(n).index,
                    (airframes.value_counts().head(n)),
                )
            ]
            airframes.value_counts().head(n).plot.pie(
                labels=values, autopct="%1.1f%%"
            )
            plt.title(f"{attribute} (main {n})")
        else:
            return airframes.value_counts().head(n)

    #
    # Wrappers
    #

    def plot(
        self,
        wide=140,
        ratio=0.6,
        move_east=-70,
        move_north=0,
        alpha=0.1,
        cmap="brg",
        title="",
    ):
        """
        Plot a scene
        """
        Fl.plot_very_long_list(
            flights=self.flights,
            title=title,
            wide=wide,
            alpha=alpha,
            cmap=cmap,
            move_east=move_east,
            move_north=move_north,
            ratio=ratio,
        )

    def plot_elevation_profile(
        self,
        xlim=500,
        ylim=None,
        factortime=1,
        endalign=True,
        alpha=0.1,
    ):
        """
        Plots the elevation profile of a scene
        """
        Fl.plot_elevation_profile(
            flights=self.flights,
            xlim=xlim,
            ylim=ylim,
            factortime=factortime,
            endalign=endalign,
            alpha=alpha,
        )

    def plot_elevation_histogram(
        self,
        bins=100,
        alpha_threshold=0,
        xlim=None,
        ylim=None,
        endalign=True,
        cmap="prism_r",
    ):
        """
        Plots the elevation profile of a scene
        """
        Fl.plot_elevation_histogram(
            flights=self.flights,
            bins=bins,
            alpha_threshold=alpha_threshold,
            xlim=xlim,
            ylim=ylim,
            endalign=endalign,
            cmap=cmap,
        )

    #
    # Other dunders
    #

    def __repr__(self):
        """
        Convert to string showing number of flights
        """
        return f"Scene with {len(self.flights)} flights, {self.notes}."
