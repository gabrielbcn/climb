"""
The Flight class

by
Gabriel Mesquida Masana
gabmm@stanford.edu

"""

import json
import math
import os
import warnings
import jsonpath_ng as jp
from datetime import datetime


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
import pandas as pd
import pymap3d
import matplotlib as mpl
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from svgpathtools import svg2paths
from svgpath2mpl import parse_path


# Ignoring warnings when rendering plots
warnings.filterwarnings("ignore")


class Flight:
    """
    The Flight class
    ================
    Version: 3.02
    Last update: 15/11/24
    -----------------------
    Gabriel Mesquida Masana
    gabmm@stanford.edu

    A Flight object contains:

    * flight_id : string unique to each flight
    * timebase : unix timestamp from first track data
    * callsign : flight callsign
    * adep : airport of departure
    * ades : airport of descent
    * aircraft : type of aircraft
    * numfixes : number of fixes included in fixes dataframe
    * numtracks : number of tracks included in tracks dataframe
    * continuous : true if the track data is continuous
    * c1mean : measure of pilot accuracy (data enhancement)
    * fixes : dataframe with flight fixes
    * tracks : dataframe with flight tracks
    * filename : filename including the raw data

    Methods:
        * __init__
            1. If name is indicated in file_name, get from rawdata folder
            2. From JSON content
            3. If flight_id is indicated get from main dataframe
            4. If flight is provided in from_flight, a deep copy of the
                flight object is provided
        * __repr__

    for data loading and saving
        * load_dataframe (static)
        * dataframe_magnitudes (static)
        * get_main_dataframe (static)
        * set_main_dataframe (static)
        * create_flights : from a pandas boolean vector (static)
        * save_flights : save dataframe from list of flights (static)
        * deep_copy : create duplicated list with
            deep copies of flights (static)
        * get_json : get json content from raw data file

    for interpolation
        * is_continuous : check for track continuity
        * grouped_discontinuities : return list of discontinuity ranges
        * interpolate : cover track gaps and remove sequences
            that are not long enough

    for track expansion including intent (based on Tran, et al. (2022))
        * get_lat_lng_fix : internal for calculations
        * get_lat_lng_track : internal for calculations
        * get_lat_lng_alt_track : internal for calculations
        * get_lat_lng_alt_sec_two_tracks : internal for calculations
        * get_closest_fix_the_hard_way : internal for calculations
        * get_sq_dist : internal for calculations
        * get_dist : internal for calculations
        * position_wps_expand : internal for calculations
            does the mathematical calculations for intent modelling
        * get_speed : internal for calculations
            calculates speed from position
        * expand_location : provides intent for a specific location
        * expand_closest_wp : expand track with nearest WP
        * expand_tracks : provides track enhancement with intent
        * remove_expansion

    for rendering
        * plot
        * plot_list (static)
        * plot_very_long_list (static)
        * plot_elevation_profile (static)
        * plot_expanded : plots flight profile with expanded
            parameters
        * plot_with_altitude : plots fixes and altitude profile
        * plot_histogram
        * stats


    Folders:
    * ./procedures contains WPs, SIDs & STARs
    * ./rawdata contains the tracks raw data with most likely fixes
    * ./data contains the dataframe with all flights once cleaned

    """

    # Domain constants
    MINTRACKS = 180  # 12 minutes
    MINCONTTRACKS = 60  # 4 minutes
    MAXGAP = 16  # up to one minute will be interpolated
    C1MEANTHRESHOLD = 0.2  # good behaviour metric
    WSSS_LAT = 1.3591666666666666
    WSSS_LNG = 103.99111111111111
    METER_2_FEET = 3.28084

    # retrieve Singapore FIR shape for rendering
    FIR_FILENAME = "./fir/SG_FIR.geojson"
    SINGAPORE_FIR = [gpd.read_file(FIR_FILENAME)]

    # retrieve Singapore FIR sectors for rendering
    SECTOR_FILENAME = "./sectors/SG_SECTOR_{}.geojson"
    SINGAPORE_FIR_SECTORS = []
    for i_ in range(1, 9):
        SINGAPORE_FIR_SECTORS.append(gpd.read_file(SECTOR_FILENAME.format(i_)))

    # retrieve waypoints for rendering
    WAYPOINTS = pd.read_pickle("procedures/SG_waypoints_FPL.pickle")
    PROCWAYPOINTS = pd.read_pickle("procedures/SG_waypoints.pickle")
    PROCWAYPOINTSNAMES = PROCWAYPOINTS["FixName"].values

    # Except for WSSS, assess the Fix which is used most often to be
    # represented in the biggest size
    WAYPOINTSMAXCOUNT = WAYPOINTS["Count"].nlargest(2).iloc[1]

    # WP rendering
    MAXWPSIZE = 30
    MINWPSIZE = 5

    # File constants
    DEFAULT_RAW_DIR_NAME = "raw_data/"
    DEFAULT_RAW_EXTENSION = ".json"
    DEFAULT_DATA_DIR_NAME = "./data/"
    DEFAULT_DATAFRAME = "all_flights.pkl"

    # Presentation constants
    DEFAULTWIDEFIGSIZE = (10, 6)  # For screen
    DEFAULTDPI = 100
    DEFAULTCMAP = plt.get_cmap("nipy_spectral")
    params = {
        "legend.fontsize": "large",
        "figure.figsize": DEFAULTWIDEFIGSIZE,
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    plt.rcParams.update(params)
    DEFAULTFACECOLOR = "w"
    DEFAULTEDGECOLOR = "g"
    DEFAULTTEXTCOLOR = "k"

    # Airplane marker
    # https://petercbsmith.github.io/marker-tutorial.html
    airplane_path, airplane_attributes = svg2paths("icons/airplane.svg")
    airplane_marker = parse_path(airplane_attributes[0]["d"])
    airplane_marker.vertices -= airplane_marker.vertices.mean(axis=0)
    airplane_marker = airplane_marker.transformed(
        mpl.transforms.Affine2D().rotate_deg(180 - 45)
    )
    AIRPLANE_MARKER = airplane_marker.transformed(
        mpl.transforms.Affine2D().scale(-1, 1)
    )

    # Class variables being declared
    # This one will be the main dataframe when loaded
    main_dataframe = pd.DataFrame()

    def __init__(
        self,
        file_name=None,  # For 2019 files only, 2024 use raw_set_ingest
        json_content=None,  # To use with raw_set_ingest
        flight_id=None,
        from_flight=None,
        from_dataframe=None,
    ):
        """
        Creating a flight object

        1. If name is indicated in file_name, get from rawdata folder
        2. From JSON content
        3. If flight_id is indicated get from main dataframe
        4. If flight is provided in from_flight, a deep copy of the
            flight object is provided
        """

        # Declaring all fields in __init__
        (
            self.flight_id,
            self.timebase,
            self.callsign,
            self.adep,
            self.ades,
            self.aircraft,
            self.numfixes,
            self.numtracks,
            self.continuous,
            self.fixes,
            self.tracks,
            self.filename,
        ) = [None] * 12

        # Plus empty dictionary in case  more info needs to be
        #  added in data transformations and kept
        self.additional = {}

        # From raw data file
        if file_name:
            self.filename = (
                Flight.DEFAULT_RAW_DIR_NAME
                + file_name
                + Flight.DEFAULT_RAW_EXTENSION
            )
            if not os.path.exists(self.filename):
                print(f"The file {self.filename} does not exist")
            else:
                # Loading data
                j = self.get_json()
                self.process_flight_json_original(j)

        # From content in package of raw flights
        elif json_content:
            self.process_flight_json_batch(json_content)

        # From dataframe by index
        elif flight_id:
            if len(Flight.main_dataframe) == 0 and from_dataframe is None:
                print(
                    "Dataframe not loaded, cannot create flights",
                    "from dataframe",
                )
            else:
                try:
                    if from_dataframe is None:
                        dataframe = Flight.main_dataframe
                    else:
                        dataframe = from_dataframe
                    item = dataframe.loc[flight_id]

                    self.flight_id = flight_id
                    self.timebase = item["Timebase"]
                    self.callsign = item["Callsign"]
                    self.adep = item["ADEP"]
                    self.ades = item["ADES"]
                    self.aircraft = item["Aircraft"]
                    self.numfixes = item["NumFixes"]
                    self.numtracks = item["NumTracks"]
                    self.continuous = item["Continuous"]
                    self.fixes = item["Fixes"]
                    self.tracks = item["Tracks"]

                    if "Additional" in item.index.values:
                        self.additional = item["Additional"]

                except pd.errors.InvalidIndexError:
                    print("Index not found in dataframe")

        # From another instance
        elif isinstance(from_flight, Flight):
            self.flight_id = from_flight.flight_id + " Copy"
            self.timebase = from_flight.timebase
            self.callsign = from_flight.callsign
            self.adep = from_flight.adep
            self.ades = from_flight.ades
            self.aircraft = from_flight.aircraft
            self.numfixes = from_flight.numfixes
            self.numtracks = from_flight.numtracks
            self.continuous = from_flight.continuous
            self.fixes = from_flight.fixes.copy(deep=True)
            self.tracks = from_flight.tracks.copy(deep=True)
            self.additional = self.additional.copy()

        else:
            print("Flight cannot be created without parameters")

    def __repr__(self):
        """
        Convert to string showing id
        """
        return self.flight_id

    #
    # DATAFRAME INGESTION BLOCK
    #
    #

    @staticmethod
    def load_dataframe(file_name=None):
        """
        Loads the full dataset from main dataframe
        """
        if file_name:
            Flight.main_dataframe = pd.read_pickle(
                Flight.DEFAULT_DATA_DIR_NAME + file_name
            )
        else:
            Flight.main_dataframe = pd.read_pickle(
                Flight.DEFAULT_DATA_DIR_NAME + Flight.DEFAULT_DATAFRAME
            )

        # Just to double check
        initial_len = len(Flight.main_dataframe)
        Flight.main_dataframe = Flight.main_dataframe[
            ~Flight.main_dataframe.index.duplicated(keep="first")
        ]
        uniq_len = len(Flight.main_dataframe)
        if initial_len != uniq_len:
            print(f"Filtered {initial_len-uniq_len} duplicates in dataframe")

        print(f"Loaded {len(Flight.main_dataframe)} flights")

    @staticmethod
    def dataframe_magnitudes():
        """
        Shows the key magnitudes on the dataframe contents
        """
        # Flights
        print(
            f"Dataframe contains {len(Flight.main_dataframe)} flights",
            f"with {Flight.main_dataframe.size} values",
        )

        # Total datapoints
        ntracks = (
            Flight.main_dataframe["Tracks"].apply(lambda item: len(item)).sum()
        )
        ptracks = (
            Flight.main_dataframe["Tracks"].apply(lambda item: item.size).sum()
        )
        print(f"Dataframe contains {ntracks} tracks with {ptracks} values")

        # Total fixes
        nfixes = (
            Flight.main_dataframe["Fixes"].apply(lambda item: len(item)).sum()
        )
        pfixes = (
            Flight.main_dataframe["Fixes"].apply(lambda item: item.size).sum()
        )
        print(f"Dataframe contains {nfixes} fixes with {pfixes} values")

        # Total parameters
        nparams = Flight.main_dataframe.size + pfixes + ptracks
        print(f"In total, dataframe contains {nparams} values")

    @staticmethod
    def get_main_dataframe():
        """
        Returns the main dataframe
        """
        return Flight.main_dataframe

    @staticmethod
    def set_main_dataframe(from_dataframe):
        """
        Sets the main dataframe
        """
        Flight.main_dataframe = from_dataframe

    @staticmethod
    def create_flights(df_filter=None, from_dataframe=None):
        """
        Creates a list of flights from a dataframe filter
        or the list of all flights if no parameter is given
        """

        if from_dataframe is None:
            dataframe = Flight.main_dataframe
        else:
            dataframe = from_dataframe

        list_of_flights = []
        if df_filter is None:
            for id_ in dataframe.index:
                list_of_flights.append(
                    Flight(flight_id=id_, from_dataframe=from_dataframe)
                )
        else:
            for id_ in dataframe.index[df_filter]:
                list_of_flights.append(
                    Flight(flight_id=id_, from_dataframe=from_dataframe)
                )

        return list_of_flights

    @staticmethod
    def save_flights(flights, filename):
        """
        Saves a set of flights in a pickle file
        """
        df_flights = pd.DataFrame(
            columns=[
                "Id",
                "Timebase",
                "Callsign",
                "ADEP",
                "ADES",
                "Aircraft",
                "NumFixes",
                "NumTracks",
                "Continuous",
                "Fixes",
                "Tracks",
                "Additional",
            ]
        )
        df_flights = df_flights.astype(
            {
                "Timebase": "int64",
                "NumFixes": "int64",
                "NumTracks": "int64",
            },
            copy=False,
        )

        for flight_ in flights:
            # Only if enough tracks
            if flight_.tracks.empty:
                print(f"{flight_.flight_id} dropped as empty")
            else:
                df_flights = pd.concat(
                    [
                        df_flights,
                        pd.DataFrame.from_records(
                            [
                                {
                                    "Id": flight_.flight_id,
                                    "Timebase": flight_.timebase,
                                    "Callsign": flight_.callsign,
                                    "ADEP": flight_.adep,
                                    "ADES": flight_.ades,
                                    "Aircraft": flight_.aircraft,
                                    "NumFixes": len(flight_.fixes),
                                    "NumTracks": len(flight_.tracks),
                                    "Continuous": flight_.is_continuous(),
                                    "Fixes": flight_.fixes,
                                    "Tracks": flight_.tracks,
                                    "Additional": flight_.additional,
                                }
                            ]
                        ),
                    ]
                )

        # Set flight_id as index of dataframe
        df_flights.set_index("Id", inplace=True, drop=True)

        # Check for duplicates, keep highest NumTracks
        initial_len = len(df_flights)
        df_flights = df_flights.sort_values(by="NumTracks", ascending=False)
        df_flights = df_flights[~df_flights.index.duplicated(keep="first")]
        df_flights = df_flights.sort_values(by="Timebase", ascending=True)
        uniq_len = len(df_flights)
        if initial_len != uniq_len:
            print(f"Filtered {initial_len-uniq_len} duplicates in dataframe")

        # Save the full dataset in dataframe
        df_flights.to_pickle(Flight.DEFAULT_DATA_DIR_NAME + filename)
        print(f"Successfully saved {len(df_flights)} flights")

    @staticmethod
    def deep_copy(original_flights_list):
        """
        Creates a new list with deep copied flights
        """
        return [
            Flight(from_flight=flight_) for flight_ in original_flights_list
        ]

    #
    # RAW DATA INGESTION BLOCK
    #
    #

    @staticmethod
    def raw_set_ingest(file_name):
        """
        Loads a set of objects from a raw data file
        """
        file = open(file_name, encoding="utf-8")
        j = json.load(file)

        flights = []
        for match in jp.parse("$[*]").find(j):
            flights.append(Flight(json_content=match.value))
        return flights

    def process_flight_json_batch(self, j):
        """
        Clean the raw data up
        """
        # General info
        info = {}
        json_items = ["callsign", "aircraftType", "adep", "ades"]
        for item in json_items:
            info[item] = jp.parse(f"$.{item}").find(j)[0].value

        # Populating the fields
        self.callsign = info["callsign"]
        self.adep = info["adep"]
        self.ades = info["ades"]
        self.aircraft = info["aircraftType"]

        # Creating ID
        self.flight_id = " ".join(
            [self.callsign, self.adep, self.ades, self.aircraft]
        )

        # Getting fixes
        list_fixes = jp.parse("$.fixes").find(j)[0].value
        self.fixes = pd.json_normalize(list_fixes)
        if not self.fixes.empty:
            self.fixes.rename(
                columns={
                    "eto": "ETO",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "fixName": "FixName",
                    "fixType": "FixType",
                    "computedFL": "ComputedFL",
                    "computedSpeed": "ComputedSpeed",
                },
                inplace=True,
            )

        # Getting tracks
        list_tracks = jp.parse("$.coupledTrkUpd").find(j)[0].value
        self.tracks = pd.json_normalize(list_tracks)
        if list_tracks:
            self.tracks.drop(
                ["trkNumber", "coupledFplNumber"],
                axis=1,
                inplace=True,
            )
            if not self.tracks.empty:

                self.tracks.rename(
                    columns={
                        "time": "Time",
                        "latitude": "Latitude",
                        "longitude": "Longitude",
                        "altitude": "Altitude",
                        "speedX": "Vx",
                        "speedY": "Vy",
                    },
                    inplace=True,
                )

                # Make time relative and round, and altitude round
                self.tracks["Time"] = self.tracks["Time"] // 1000
                self.tracks["Unix"] = self.tracks["Time"]
                timebase = self.tracks.loc[0, "Time"]
                self.tracks["Time"] = (
                    self.tracks["Time"].add(-timebase).div(4).round()
                )

                # remove time duplicates and redo index
                self.tracks = self.tracks.groupby(
                    "Time", as_index=False
                ).mean()

                # Types to integer
                self.tracks = self.tracks.astype(
                    {"Time": "int32", "Unix": "int64"},
                    copy=True,
                    errors="ignore",
                )

                # use time as index -> required for successful interpolation
                self.tracks.set_index("Time", inplace=True, drop=False)

                # Store the timebase
                self.timebase = int(timebase)

                # Altitude in metres
                self.tracks.Altitude = (
                    self.tracks.Altitude / Flight.METER_2_FEET
                )
            else:
                print(f"{self.flight_id}: No track updates left after filter")
                self.timebase = 0

        else:
            print(f"{self.flight_id}: All tracks dropped for lack of coupling")
            self.timebase = 0

        # Complete flight ID based on tracks timebase if any
        if self.timebase > 0:
            self.flight_id = (
                datetime.fromtimestamp(self.timebase).strftime(
                    "%d/%m/%Y-%H:%M"
                )
                + " "
                + self.flight_id
            )

    def get_json(self):
        """
        Loads the flight object from a raw data file
        """
        file = open(self.filename, encoding="utf-8")
        j = json.load(file)
        return j

    def process_flight_json_original(self, j):
        """
        Clean the raw data up
        """
        # General info
        info = {}
        json_items = ["Callsign", "AircraftType", "ADEP", "ADES"]
        for item in json_items:
            info[item] = jp.parse(f"$.{item}").find(j)[0].value

        # Populating the fields
        self.callsign = info["Callsign"]
        self.adep = info["ADEP"]
        self.ades = info["ADES"]
        self.aircraft = info["AircraftType"]

        # Creating ID
        self.flight_id = " ".join(
            [self.callsign, self.adep, self.ades, self.aircraft]
        )

        # Getting fixes
        list_fixes = jp.parse("$.Fixes").find(j)[0].value
        self.fixes = pd.json_normalize(list_fixes)
        if not self.fixes.empty:
            self.fixes = self.fixes[
                self.fixes["FixType"].eq("RP")
                | self.fixes["FixType"].eq("AIRPORT")
            ]
            self.fixes.reset_index(inplace=True, drop=True)
            self.fixes.drop(
                ["ETO", "FixType", "SectorIndex"], axis=1, inplace=True
            )

        # Getting tracks
        list_tracks = jp.parse("$.CoupledTrkUpd").find(j)[0].value
        self.tracks = pd.json_normalize(list_tracks)
        if list_tracks:
            self.tracks.drop(
                ["TrkNumber", "CoupledFplNumber", "ModeS", "ModeA"],
                axis=1,
                inplace=True,
            )

            # Filtering by Callsign or Ident to avoid extraneous data
            # Note that the callsign field is padded with spaces
            self.tracks["Callsign"] = self.tracks.apply(
                lambda row: row["Callsign"].strip(), axis=1
            )
            self.tracks["AircraftIdent"] = self.tracks.apply(
                lambda row: row["AircraftIdent"].strip(), axis=1
            )

            # Dropping non-linked track updates
            previous_size = len(self.tracks)
            self.tracks = self.tracks[
                self.tracks["Callsign"].eq(self.callsign)
                | self.tracks["AircraftIdent"].eq(self.callsign)
            ]
            self.tracks.drop(
                ["AircraftIdent", "Callsign"], axis=1, inplace=True
            )

            # Resetting the index and assessing how many have dropped
            if len(self.tracks) < previous_size:
                self.tracks.reset_index(inplace=True, drop=True)
                print(
                    f"{self.flight_id}: Discarded "
                    + f"{previous_size - len(self.tracks)} track updates"
                )

            if not self.tracks.empty:
                # Make time relative and round, and altitude round
                self.tracks["Unix"] = self.tracks["Time"]
                timebase = self.tracks.loc[0, "Time"]
                self.tracks["Time"] = (
                    self.tracks["Time"].add(-timebase).div(4).round()
                )

                # remove time duplicates and redo index
                self.tracks = self.tracks.groupby(
                    "Time", as_index=False
                ).mean()

                # Types to integer
                self.tracks = self.tracks.astype(
                    {"Time": "int32", "Unix": "int64"},
                    copy=True,
                    errors="ignore",
                )

                # use time as index -> required for successful interpolation
                self.tracks.set_index("Time", inplace=True, drop=False)

                # Store the timebase
                self.timebase = int(timebase)
            else:
                print(f"{self.flight_id}: No track updates left after filter")
                self.timebase = 0

        else:
            print(f"{self.flight_id}: All tracks dropped for lack of coupling")
            self.timebase = 0

        # Complete flight ID based on tracks timebase if any
        if self.timebase > 0:
            self.flight_id = (
                datetime.fromtimestamp(self.timebase).strftime(
                    "%d/%m/%Y-%H:%M"
                )
                + " "
                + self.flight_id
            )

    #
    # INTERPOLATION BLOCK
    #
    #

    def is_continuous(self):
        """
        Checks the continuity of the track updates
        """
        if not self.tracks.empty:
            self.continuous = (
                len(self.tracks) - 1 == self.tracks.iloc[-1]["Time"]
            )
            return self.continuous
        else:
            print(f"Empty tracks in flight {self.flight_id}")
            return False

    def grouped_discontinuities(self):
        """
        Provides a list of grouped consecutive discontinuities in the track
        updates
        """
        if self.tracks.empty:
            return []

        list_tracks = sorted(
            set(range(round(self.tracks.iloc[-1]["Time"] + 1)))
            - set(self.tracks["Time"].tolist())
        )
        slow, fast = 0, 0
        ans, temp = [], []
        while fast < len(list_tracks):
            if fast - slow == list_tracks[fast] - list_tracks[slow]:
                temp.append(list_tracks[fast])
                fast += 1
            else:
                slow = fast
                ans.append(temp)
                temp = []
        if fast > slow:
            ans.append(temp)
        return ans

    def interpolate(self, trace=True):
        """
        Filling the track gaps with interpolated values

        Note that the interpolation must be produced before the expansion,
        otherwise raises an exception
        """
        if self.tracks.empty:
            if trace:
                print(f"{self.flight_id}: no tracks to interpolate")
            return

        elif len(self.tracks) < Flight.MINTRACKS:
            if trace:
                print(
                    f"Discarding flight as too short ({len(self.tracks)*4}s)"
                )
            self.tracks.drop(self.tracks.index, inplace=True)
            self.timebase = 0
            return

        # The work section
        if self.is_continuous():
            # Nothing to be done but the final checkings
            self.continuous = True

        else:
            if len(self.tracks.columns) != 6:
                # If we interpolate after expansion, expansion gets removed
                if trace:
                    print(
                        f"{self.flight_id}: interpolating removes previous expansion"
                    )
                self.remove_expansion()

            # Interpolate
            groupofgroups_ = self.grouped_discontinuities()

            # First fill in the small discontinuities with NaNs and sort
            for group_ in groupofgroups_:
                if len(group_) <= Flight.MAXGAP:
                    if group_[0] == group_[-1]:
                        if trace:
                            print(
                                f"{self.flight_id}: interpolating for",
                                f"[{group_[0]}]",
                            )
                    else:
                        if trace:
                            print(
                                f"{self.flight_id}: interpolating for",
                                f"[{group_[0]},{group_[-1]}]",
                            )
                    for item in group_:
                        filling_ = pd.DataFrame.from_records(
                            data=[
                                {
                                    "Time": item,
                                    "Latitude": None,
                                    "Longitude": None,
                                    "Altitude": None,
                                    "Vx": None,
                                    "Vy": None,
                                    "Unix": None,
                                }
                            ],
                        ).set_index(keys="Time", drop=False)
                        self.tracks = pd.concat(
                            [self.tracks, filling_],
                            sort=False,
                        ).sort_index()
                    self.tracks.sort_index(inplace=True)

            # Split into blocks only based on big discontinuities
            previous_ = 0
            trackupdateblocks_ = []
            # Create a last fake group
            groupofgroups_ += [
                [self.tracks.iloc[-1]["Time"]] * (Flight.MAXGAP + 1)
            ]
            for group_ in groupofgroups_:
                if len(group_) > Flight.MAXGAP:
                    # Gap too long, make break
                    if group_[0] != self.tracks.iloc[-1]["Time"]:
                        if trace:
                            print(
                                f"{self.flight_id}: found relevant gap [{group_[0]}, {group_[-1]}]"
                            )
                    # Interpolation process
                    if (
                        group_[0] - previous_
                    ) > Flight.MAXGAP:  # Only if long enough
                        block_ = self.tracks[
                            self.tracks.index.get_loc(
                                previous_
                            ) : self.tracks.index.get_loc(round(group_[0] - 1))
                        ]
                        block_ = block_.astype(float)
                        block_.index = block_.index.astype(float)
                        block_ = block_.interpolate(method="spline", order=2)
                        block_ = block_.astype(
                            {"Time": "int32", "Unix": "int64"},
                            copy=True,
                            errors="ignore",
                        )
                        block_.index = block_.index.astype("int32")
                        trackupdateblocks_.append(block_)
                    previous_ = group_[-1] + 1

            # If blocks are there, assemble the pieces
            if trackupdateblocks_:
                self.tracks = pd.concat(trackupdateblocks_)
                if len(trackupdateblocks_) > 1:
                    self.continuous = False
                else:
                    self.continuous = True

                # Remove start gap
                newtimebase = self.tracks.iloc[0]["Time"]
                if newtimebase != 0:
                    self.tracks["Time"] = self.tracks["Time"].add(-newtimebase)
                    self.tracks = self.tracks.astype(
                        {"Time": "int32", "Unix": "int64"},
                        copy=True,
                        errors="ignore",
                    )
                self.tracks.set_index("Time", inplace=True, drop=False)

            else:
                # Nothing left, but only realised after grouping
                if trace:
                    print(
                        "Not enough tracks to interpolate detected after grouping"
                    )
                self.tracks.drop(self.tracks.index, inplace=True)
                self.timebase = 0
                return

        # Final checks
        while self.tracks.iloc[0].isnull().any():
            self.tracks.drop(self.tracks.head(1).index, inplace=True)
            if trace:
                print("Dropped first track because of NaN")

        while self.tracks.iloc[-1].isnull().any():
            self.tracks.drop(self.tracks.tail(1).index, inplace=True)
            if trace:
                print("Dropped last track because of NaN")

        if len(self.tracks[self.tracks.isnull().any(axis=1)]) > 0:
            self.tracks = self.tracks.interpolate(method="spline", order=3)
            if trace:
                print("Tried to interpolate central NaNs")

        if self.tracks.iloc[-1].Time - self.tracks.iloc[-2].Time > 10:
            self.tracks.drop(self.tracks.tail(1).index, inplace=True)
            if trace:
                print("Dropped final inconsistent altitude value")

    #
    # TRACK UPDATES EXPANSION BLOCK
    #
    #

    def get_lat_lng_fix(self, index):
        """
        Get the latitude and longitude of a fix
        Helper internal method
        """
        return (
            self.fixes.iloc[index]["Latitude"],
            self.fixes.iloc[index]["Longitude"],
        )

    def get_lat_lng_track(self, index):
        """
        Get the latitude and longitude of a track update
        Helper internal method
        """
        return (
            self.tracks.iloc[index]["Latitude"],
            self.tracks.iloc[index]["Longitude"],
        ), self.tracks.iloc[index]["Altitude"]

    def get_lat_lng_alt_track(self, index):
        """
        Get the latitude, longitude and altitude of one track update
        Helper internal method
        """
        return (
            self.tracks.iloc[index]["Latitude"],
            self.tracks.iloc[index]["Longitude"],
            self.tracks.iloc[index]["Altitude"],
        )

    def get_lat_lng_alt_two_tracks(self, index):
        """
        Get the latitude, longitude and altitude of two track updates
        Helper internal method
        """
        return (
            self.tracks.iloc[index]["Latitude"],
            self.tracks.iloc[index]["Longitude"],
            self.tracks.iloc[index]["Altitude"],
            self.tracks.iloc[index - 1]["Latitude"],
            self.tracks.iloc[index - 1]["Longitude"],
            self.tracks.iloc[index - 1]["Altitude"],
        )

    def get_closest_fix_the_hard_way(self, track):
        """
        Get the closest fix doing the full calculation
        Helper internal method
        """
        latfixes = self.fixes["Latitude"].to_numpy()
        lngfixes = self.fixes["Longitude"].to_numpy()
        vector = (latfixes - track[0]) ** 2 + (lngfixes - track[1]) ** 2
        closest_fix = np.where(vector == np.amin(vector))[0][0]

        return (
            closest_fix - 1 if closest_fix > 0 else 0,
            closest_fix,
            (
                closest_fix + 1
                if closest_fix < len(latfixes) - 1
                else len(latfixes) - 1
            ),
        )

    def get_sq_dist(self, point1, point2):
        """
        Get the squared distance between two points
        Helper internal method
        """
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    def get_dist(self, point1, point2):
        """
        Get the distance between two points
        Helper internal method
        """
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        )

    @staticmethod
    def geo_to_enu(point, reference):
        """
        Convert from geodetic coordinates to ENU
        """
        return pymap3d.geodetic2enu(
            point[0],
            point[1],
            point[2],
            reference[0],
            reference[1],
            reference[2],
        )

    @staticmethod
    def enu_len(e_, n_, u_):
        """
        Find the ENU length
        """
        return math.sqrt(e_**2 + n_**2 + u_**2)

    def position_wps_expand(
        self,
        lat_lng_aircraft,
        alt_aircraft,
        lat_lng_previous_fix,
        lat_lng_closest_fix,
        lat_lng_next_fix,
        v_x,
        v_y,
    ):
        """
        Given a position and the set of WPs return expanded
        parameters by doing all calculations
        Helper internal method following Tran, et al. (2022):

        b_1 : angle between aircraft movement direction and line from aircraft
            to closest waypoint
        b_2 : angle between aircraft movement direction and line from aircraft
            to next waypoint
        cos_b_1 : cos angle between aircraft movement direction and line from
            aircraft to closest waypoint, this is the one actually stored and
            not the angle
        cos_b_2 : cos angle between aircraft movement direction and line from
            aircraft to next waypoint, this is the one actually stored and not
            the angle
        c_1 : distance to current airway
        c_2 : distance to next airway
        d_1 : distance to closest waypoint
        d_2 : distance to next waypoint
        l_1_l, l_1_t: longitudinal and transverse speed projected
            towards current waypoint
        l_1_l, l_1_t: longitudinal and transverse speed projected
            towards next waypoint
        """

        # d1 : distance to closest fix
        # d_1 = self.get_dist(lat_lng_aircraft, lat_lng_closest_fix)
        enu_wp1 = Flight.geo_to_enu(
            lat_lng_closest_fix + tuple([alt_aircraft]),
            lat_lng_aircraft + tuple([alt_aircraft]),
        )
        d_1 = Flight.enu_len(*enu_wp1)

        # d2 : distance to next fix
        # d_2 = self.get_dist(lat_lng_aircraft, lat_lng_next_fix)
        enu_wp2 = Flight.geo_to_enu(
            lat_lng_next_fix + tuple([alt_aircraft]),
            lat_lng_aircraft + tuple([alt_aircraft]),
        )
        d_2 = Flight.enu_len(*enu_wp2)

        # theta : angle of aircraft velocity vector
        theta_aircraft = math.atan2(v_x, v_y)
        if theta_aircraft < 0:
            theta_aircraft += 2 * math.pi

        # vlen : length of aircraft velocity vector
        vlen = math.sqrt(v_y**2 + v_x**2)

        # alpha1 : angle of line between aircraft progress and
        # current waypoint
        alpha_1 = math.atan(enu_wp1[0] / enu_wp1[1])
        if enu_wp1[0] < 0 and enu_wp1[1] < 0:
            alpha_1 += math.pi
        elif enu_wp1[0] < 0 and enu_wp1[1] > 0:
            alpha_1 += 2 * math.pi
        if alpha_1 < 0:
            alpha_1 += math.pi

        # b1 : angle between aircraft movement direction and line
        # from aircraft to closest waypoint
        b_1 = alpha_1 - theta_aircraft

        # alpha2 : angle of line between aircraft and next waypoint
        alpha_2 = math.atan(enu_wp2[0] / enu_wp2[1])
        if enu_wp2[0] < 0 and enu_wp2[1] < 0:
            alpha_2 += math.pi
        elif enu_wp2[0] < 0 and enu_wp2[1] > 0:
            alpha_2 += 2 * math.pi
        if alpha_2 < 0:
            alpha_2 += math.pi

        # b2 : angle between aircraft movement direction and line
        # from aircraft to next waypoint
        b_2 = alpha_2 - theta_aircraft

        # c1 : distance to current airway
        # https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # d = abs( (x2-x1)(y1-y0) - (x1-x0)(y2-y1) )
        #           / sqrt( (x2-x1)^2 + (y2-y1)^2 )
        # where 1 =closest_fix,  2 =next_fix,
        # 0 =aircraft x=longitude=1 y=latitude=0
        y_0, x_0 = lat_lng_aircraft
        y_1, x_1 = lat_lng_previous_fix
        y_2, x_2 = lat_lng_closest_fix
        c_1 = abs(
            (x_2 - x_1) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_1)
        ) / math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        # c2 : distance to next airway
        # https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # d = abs( (x2-x1)(y1-y0) - (x1-x0)(y2-y1) )
        #           / sqrt( (x2-x1)^2 + (y2-y1)^2 )
        # where 1 =closest_fix,  2 =next_fix,
        # 0 =aircraft x=longitude=1 y=latitude=0
        y_0, x_0 = lat_lng_aircraft
        y_1, x_1 = lat_lng_closest_fix
        y_2, x_2 = lat_lng_next_fix
        c_2 = abs(
            (x_2 - x_1) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_1)
        ) / math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        # Movement vector projected towards current waypoint
        cos_b_1 = math.cos(b_1)
        l_1_l = vlen * cos_b_1
        l_1_t = vlen * math.sin(b_1)

        # Movement vector projected towards next waypoint
        cos_b_2 = math.cos(b_2)
        l_2_l = vlen * cos_b_2
        l_2_t = vlen * math.sin(b_2)

        return (
            theta_aircraft,
            alpha_1,
            b_1,
            cos_b_1,
            alpha_2,
            b_2,
            cos_b_2,
            c_1,
            c_2,
            d_1,
            d_2,
            l_1_l,
            l_1_t,
            l_2_l,
            l_2_t,
        )

    def get_speed(
        self,
        latitude,
        longitude,
        altitude,
        ref_latitude,
        ref_longitude,
        ref_altitude,
        seconds,
    ):
        """
        Provides the speed based on two geodetic coordinates
        and the seconds that have passed
        """
        e_, n_, _ = pymap3d.geodetic2enu(
            latitude,
            longitude,
            altitude,
            ref_latitude,
            ref_longitude,
            ref_altitude,
        )
        v_x = e_ / seconds
        v_y = n_ / seconds
        return v_x, v_y

    def expand_location(
        self,
        latitude,
        longitude,
        altitude,
        previous_latitude,
        previous_longitude,
        previous_altitude,
        seconds,
    ):
        """
        Provides the expanded parameters or
        pilot intent for a specific location
        """
        lat_lng_aircraft = (latitude, longitude)
        (
            previous_fix,
            closest_fix,
            next_fix,
        ) = self.get_closest_fix_the_hard_way(lat_lng_aircraft)
        lat_lng_previous_fix = self.get_lat_lng_fix(previous_fix)
        lat_lng_closest_fix = self.get_lat_lng_fix(closest_fix)
        lat_lng_next_fix = self.get_lat_lng_fix(next_fix)
        v_x, v_y = self.get_speed(
            latitude,
            longitude,
            altitude,
            previous_latitude,
            previous_longitude,
            previous_altitude,
            seconds,
        )

        # Return expanded values
        return pd.Series(
            self.position_wps_expand(
                lat_lng_aircraft,
                lat_lng_previous_fix,
                lat_lng_closest_fix,
                lat_lng_next_fix,
                v_x,
                v_y,
            ),
            index=[
                "theta",
                "b_1",
                "cos_b_1",
                "b_2",
                "cos_b_2",
                "c_1",
                "c_2",
                "d_1",
                "d_2",
                "l_1_l",
                "l_1_t",
                "l_2_l",
                "l_2_t",
            ],
        ).fillna(value=0)

    def expand_closest_wp(self):
        """
        Calculates which waypoint is closest to a track
        and provides the expansion in the tracks
        dataframe

        ClosestFix : closest fix to the track update
        """

        # Initialise
        closest_fix_list = []

        for item in range(0, len(self.tracks)):
            # Aircraft position and movement angle
            lat_lng_aircraft, _ = self.get_lat_lng_track(item)

            # Initially and regularly get the fix the hard way
            if item % 20 == 0:
                (
                    previous_fix,
                    closest_fix,
                    next_fix,
                ) = self.get_closest_fix_the_hard_way(lat_lng_aircraft)
                lat_lng_closest_fix = self.get_lat_lng_fix(closest_fix)
                lat_lng_next_fix = self.get_lat_lng_fix(next_fix)

            # Check the distance compared to the next fix
            elif closest_fix < len(self.fixes) - 1 and self.get_sq_dist(
                lat_lng_closest_fix, lat_lng_aircraft
            ) > self.get_sq_dist(lat_lng_next_fix, lat_lng_aircraft):
                # And move on
                closest_fix, lat_lng_closest_fix = next_fix, lat_lng_next_fix
                next_fix = (
                    closest_fix + 1
                    if closest_fix < len(self.fixes) - 1
                    else len(self.fixes) - 1
                )
                lat_lng_next_fix = self.get_lat_lng_fix(next_fix)

            closest_fix_list.append(closest_fix)

        # Put into dataframe
        #
        # closest fixes
        self.tracks["ClosestFix"] = closest_fix_list
        return self

    def expand_tracks(self):
        """
        Calculates which waypoint is closest to a track
        and provides the expanded parameters in the tracks
        dataframe

        ClosestFix : closest fix to the track update
        And, following Tran, et al. (2022)
        cos_b_1, cos_b_2, c_1, c_2, d_1, d_2, l_1_l, l_1_t, l_2_l, l_2_t
        obtained by internal method position_wps_expand
        """
        if not self.tracks.empty:

            # Initialise
            (
                theta_list,
                closest_fix_list,
                l_1_l_list,
                l_1_t_list,
                l_2_l_list,
                l_2_t_list,
                d_1_list,
                d_2_list,
                alpha_1_list,
                b_1_list,
                cos_b_1_list,
                alpha_2_list,
                b_2_list,
                cos_b_2_list,
                c_1_list,
                c_2_list,
            ) = ([] for _ in range(16))

            for item in range(0, len(self.tracks)):
                # Aircraft position and movement angle
                lat_lng_aircraft, alt_aircraft = self.get_lat_lng_track(item)

                # Initially and regularly get the fix the hard way
                if item % 20 == 0:
                    (
                        previous_fix,
                        closest_fix,
                        next_fix,
                    ) = self.get_closest_fix_the_hard_way(lat_lng_aircraft)
                    lat_lng_previous_fix = self.get_lat_lng_fix(previous_fix)
                    lat_lng_closest_fix = self.get_lat_lng_fix(closest_fix)
                    lat_lng_next_fix = self.get_lat_lng_fix(next_fix)

                # Check the distance compared to the next fix
                elif closest_fix < len(self.fixes) - 1 and self.get_sq_dist(
                    lat_lng_closest_fix, lat_lng_aircraft
                ) > self.get_sq_dist(lat_lng_next_fix, lat_lng_aircraft):
                    # And move on
                    previous_fix, lat_lng_previous_fix = (
                        closest_fix,
                        lat_lng_closest_fix,
                    )
                    closest_fix, lat_lng_closest_fix = (
                        next_fix,
                        lat_lng_next_fix,
                    )
                    next_fix = (
                        closest_fix + 1
                        if closest_fix < len(self.fixes) - 1
                        else len(self.fixes) - 1
                    )
                    lat_lng_next_fix = self.get_lat_lng_fix(next_fix)

                # Get the expanded values
                (
                    theta,
                    alpha_1,
                    b_1,
                    cos_b_1,
                    alpha_2,
                    b_2,
                    cos_b_2,
                    c_1,
                    c_2,
                    d_1,
                    d_2,
                    l_1_l,
                    l_1_t,
                    l_2_l,
                    l_2_t,
                ) = self.position_wps_expand(
                    lat_lng_aircraft,
                    alt_aircraft,
                    lat_lng_previous_fix,
                    lat_lng_closest_fix,
                    lat_lng_next_fix,
                    self.tracks.iloc[item]["Vx"],
                    self.tracks.iloc[item]["Vy"],
                )

                closest_fix_list.append(closest_fix)
                theta_list.append(theta)
                alpha_1_list.append(alpha_1)
                b_1_list.append(b_1)
                cos_b_1_list.append(cos_b_1)
                alpha_2_list.append(alpha_2)
                b_2_list.append(b_2)
                cos_b_2_list.append(cos_b_2)
                c_1_list.append(c_1)
                c_2_list.append(c_2)
                d_1_list.append(d_1)
                d_2_list.append(d_2)
                l_1_l_list.append(l_1_l)
                l_1_t_list.append(l_1_t)
                l_2_l_list.append(l_2_l)
                l_2_t_list.append(l_2_t)

            # Put into dataframe
            #
            # closest fixes
            self.tracks["ClosestFix"] = closest_fix_list
            # expansions
            self.tracks["theta"] = theta_list
            self.tracks["alpha_1"] = alpha_1_list
            self.tracks["b_1"] = b_1_list
            self.tracks["cos_b_1"] = cos_b_1_list
            self.tracks["alpha_2"] = alpha_2_list
            self.tracks["b_2"] = b_2_list
            self.tracks["cos_b_2"] = cos_b_2_list
            self.tracks["c_1"] = c_1_list
            # when too close may give error because of divided by zero,
            # this fills the "not a number"s with zero
            self.tracks["c_1"] = self.tracks["c_1"].fillna(0)
            self.tracks["c_2"] = c_2_list
            # when too close may give error because of divided by zero,
            # this fills the "not a number"s with zero
            self.tracks["c_2"] = self.tracks["c_2"].fillna(0)
            self.tracks["d_1"] = d_1_list
            self.tracks["d_2"] = d_2_list
            self.tracks["l_1_l"] = l_1_l_list
            self.tracks["l_1_t"] = l_1_t_list
            self.tracks["l_2_l"] = l_2_l_list
            self.tracks["l_2_t"] = l_2_t_list

            self.tracks["c_min"] = self.tracks[["c_1", "c_2"]].min(axis=1)

            self.additional["alt_max"] = self.tracks["Altitude"].max()
            self.additional["c_max"] = self.tracks["c_min"].max()
            self.additional["c_mean"] = self.tracks["c_min"].mean()

        else:
            print(f"{self.flight_id}: no tracks to expand")

        return self

    def is_expanded(self):
        """
        Quick check if expaned
        """
        return "ClosestFix" in self.tracks.tail(
            2
        ) and "c_1" in self.tracks.tail(2)

    def remove_expansion(self):
        """
        Cleans the expanded parameters
        """
        if "ClosestFix" in self.tracks.columns:
            self.tracks.drop(
                columns=[
                    "ClosestFix",
                    "cos_b_1",
                    "cos_b_2",
                    "c_1",
                    "c_2",
                    "d_1",
                    "d_2",
                    "l_1_l",
                    "l_1_t",
                    "l_2_l",
                    "l_2_t",
                ],
                inplace=True,
            )
        return self

    #
    # RENDERING BLOCK
    #
    #
    @staticmethod
    def decdeg2dms(dd):
        mult = -1 if dd < 0 else 1
        mnt, sec = divmod(abs(dd) * 3600, 60)
        deg, mnt = divmod(mnt, 60)
        return f"{round(mult*deg)}°{round(mult*mnt)}'{round(mult*sec)}\""

    def plot(self, flights=None, **kwargs):
        """
        Plotting the flight object

        Parameters
        ----------
        flights : the list of flights to render (optional)
        wide : horizontal length of the window in km (default 480km)
        flightplans : render the fixes
        labels : display fixes' labels
        rainbow : each flight a different colour
        fixesrainbow : each track update a different colour
            (requires track expansion)
        allwaypoints : render all the existing waypoints
            (requires loading database)
        move_east: to move the centre of the pic towards east
            (default 0km - centered)
        move_north: to move the centre of the pic towards north
            (default 0km - centered)
        ratio: to make the render window wider
            (default 1 - square)
        marks: to render a series of list of specific points
        title: to add text to the title

        Class method: if no list is produced then
        will default to the object. Calls the static method.
        """
        if flights is None:
            flights = [self]

        Flight.plot_list(flights, **kwargs)

    @staticmethod
    def plot_list(
        flights,
        wide=300,
        ratio=1,
        move_east=0,
        move_north=0,
        flightplans=True,
        labels=True,
        rainbow=False,
        fixesrainbow=False,
        allwaypoints=False,
        ha=None,
        va=None,
        stop=None,
        marks=None,
        drawing_function=None,
        title="",
    ):
        """
        Plotting a list of flight objects

        Parameters
        ----------
        flights : the list of flights to render (required)
        wide : horizontal length of the window in km (default 480km)
        ratio: to make the render window wider
            (default 1 - square)
        move_east: to move the centre of the pic towards east
            (default 0km - centered)
        move_north: to move the centre of the pic towards north
            (default 0km - centered)
        flightplans : render the fixes
        labels : display fixes' labels
        rainbow : each flight a different colour
        fixesrainbow : each track update a different colour
            (requires track expansion)
        allwaypoints : render all the existing waypoints
            (requires loading database)
        marks: to render a series of list of specific points
        title: to add text to the title
        Static method for Flight class
        """

        # Plot helper
        plot_ax, min_lat, max_lat, min_lng, max_lng = Flight.plot_helper(
            wide, move_east, move_north, ratio
        )
        widedot = wide / 600

        # Represent the waypoints
        if allwaypoints:
            for fix in Flight.WAYPOINTS.iterrows():
                if fix[1]["FixName"] != "WSSS":
                    markersize = max(
                        [
                            Flight.MAXWPSIZE
                            * fix[1]["Count"]
                            * 100
                            / wide
                            / Flight.WAYPOINTSMAXCOUNT,
                            Flight.MINWPSIZE,
                        ]
                    )
                    if fix[1]["FixName"] in Flight.PROCWAYPOINTSNAMES:
                        color = "xkcd:orange"
                    else:
                        color = Flight.DEFAULTTEXTCOLOR
                else:
                    markersize = Flight.MAXWPSIZE

                plot_ax.plot(
                    fix[1]["Longitude"],
                    fix[1]["Latitude"],
                    color=color,
                    alpha=0.5,
                    marker=".",
                    markersize=markersize,
                )

        # Flight list traversal
        for num, flight_ in enumerate(flights):
            # Fixes
            if flightplans:
                for item in range(0, len(flight_.fixes)):
                    if fixesrainbow:
                        color_ = Flight.DEFAULTCMAP(
                            item / (len(flight_.fixes) - 1)
                        )
                        linecolor_ = "xkcd:grey"
                    elif rainbow:
                        color_ = "xkcd:grey"
                        linecolor_ = Flight.DEFAULTCMAP(
                            num / (len(flights) - 1)
                        )
                    else:
                        color_ = "xkcd:grey"
                        linecolor_ = "xkcd:grey"

                    plot_ax.plot(
                        flight_.fixes.loc[item, "Longitude"],
                        flight_.fixes.loc[item, "Latitude"],
                        color=color_,
                        mec=color_,
                        marker="x",
                        markersize=10,
                        alpha=1,
                    )
                    plot_ax.plot(
                        flight_.fixes.loc[item, "Longitude"],
                        flight_.fixes.loc[item, "Latitude"],
                        color=color_,
                        mec=color_,
                        marker="o",
                        markersize=6,
                        alpha=1,
                    )

                    if item > 0:
                        plot_ax.plot(
                            [
                                flight_.fixes.loc[item - 1, "Longitude"],
                                flight_.fixes.loc[item, "Longitude"],
                            ],
                            [
                                flight_.fixes.loc[item - 1, "Latitude"],
                                flight_.fixes.loc[item, "Latitude"],
                            ],
                            "--",
                            color=linecolor_,
                            mec=linecolor_,
                        )

            # Fixes labels
            if flightplans and labels:
                halign_, halign__ = "left", "right"
                valign_, valign__ = "bottom", "top"
                for i in range(0, len(flight_.fixes)):
                    if (
                        min_lat < flight_.fixes.loc[i, "Latitude"]
                        and max_lat > flight_.fixes.loc[i, "Latitude"]
                        and min_lng < flight_.fixes.loc[i, "Longitude"]
                        and max_lng > flight_.fixes.loc[i, "Longitude"]
                    ):
                        plot_ax.text(
                            s=" " + flight_.fixes.loc[i, "FixName"] + " ",
                            x=flight_.fixes.loc[i, "Longitude"],
                            y=flight_.fixes.loc[i, "Latitude"],
                            color=Flight.DEFAULTTEXTCOLOR,
                            rotation=30,
                            ha=halign_,
                            va=valign_,
                            fontsize=10,
                        )
                        halign_, halign__ = halign__, halign_
                        valign_, valign__ = valign__, valign_

            # Surveillance
            if len(flight_.tracks) == 0:
                print(f"{flight_.flight_id}: No tracks to plot")
            else:
                if stop is not None:
                    tracks_ = flight_.tracks.iloc[: stop + 1]

                    theta = round(math.degrees(tracks_.theta.iloc[-1]), 1)
                    alpha_1 = round(math.degrees(tracks_.alpha_1.iloc[-1]), 1)
                    alpha_2 = round(math.degrees(tracks_.alpha_2.iloc[-1]), 1)

                    plt.plot(
                        tracks_.Longitude.iloc[-1],
                        tracks_.Latitude.iloc[-1],
                        color=Flight.DEFAULTTEXTCOLOR,
                        marker=MarkerStyle(
                            Flight.AIRPLANE_MARKER,
                            transform=Affine2D().rotate_deg(-theta),
                        ),
                        markersize=28,
                    )

                    plot_ax.text(
                        s=f"$t=${tracks_.Time.iloc[-1]}\n"
                        + f"$\\theta_A${theta}°"
                        + f" $v_x${tracks_.Vx.iloc[-1]} "
                        + f" $v_y${tracks_.Vy.iloc[-1]}\n"
                        + f" $\\phi${Flight.decdeg2dms(tracks_.Latitude.iloc[-1])} "
                        + f" $\\lambda${Flight.decdeg2dms(tracks_.Longitude.iloc[-1])} "
                        + f" $h${round(tracks_.Altitude.iloc[-1])}m\n"
                        # FIX 1
                        + f"{flight_.fixes.FixName.iloc[tracks_.ClosestFix.iloc[-1]]} "
                        + f" $\\phi${Flight.decdeg2dms(flight_.fixes.Latitude.iloc[tracks_.ClosestFix.iloc[-1]])} "
                        + f" $\\lambda${Flight.decdeg2dms(flight_.fixes.Longitude.iloc[tracks_.ClosestFix.iloc[-1]])}\n"
                        + f"$\\theta_{{WP1}}${alpha_1}°"
                        + f" $d${round(tracks_.d_1.iloc[-1]):,}m\n"
                        # FIX 2
                        + f"{flight_.fixes.FixName.iloc[tracks_.ClosestFix.iloc[-1]+1]} "
                        + f" $\\phi${Flight.decdeg2dms(flight_.fixes.Latitude.iloc[tracks_.ClosestFix.iloc[-1]+1])} "
                        + f" $\\lambda${Flight.decdeg2dms(flight_.fixes.Longitude.iloc[tracks_.ClosestFix.iloc[-1]+1])}\n"
                        + f"$\\theta_{{WP2}}${alpha_2}°"
                        + f" $d${round(tracks_.d_2.iloc[-1]):,}m\n",
                        x=tracks_.Longitude.iloc[-1]
                        + (
                            -widedot
                            if ha == "right"
                            else widedot if va == "left" else 0
                        ),
                        y=tracks_.Latitude.iloc[-1]
                        + (
                            -widedot / 2
                            if va == "top"
                            else widedot / 2 if va == "bottom" else 0
                        ),
                        color=Flight.DEFAULTTEXTCOLOR,
                        ha=ha if ha is not None else "center",
                        va=va if va is not None else "center",
                        fontsize=8,
                    )
                else:
                    tracks_ = flight_.tracks

                if rainbow:
                    if len(flights) > 1:
                        color_ = Flight.DEFAULTCMAP(num / (len(flights) - 0))
                    else:
                        color_ = "xkcd:green"

                    plot_ax.plot(
                        tracks_["Longitude"],
                        tracks_["Latitude"],
                        color=color_,
                        marker="s",
                        markersize=1.5,
                        alpha=0.2,
                    )
                elif fixesrainbow:
                    for _, row in tracks_.iterrows():
                        color_ = Flight.DEFAULTCMAP(
                            row["ClosestFix"] / (len(flight_.fixes) - 0)
                        )
                        plot_ax.plot(
                            row["Longitude"],
                            row["Latitude"],
                            color=color_,
                            mec=color_,
                            marker="s",
                            markersize=1.5,
                            alpha=1,
                        )

                else:
                    plot_ax.plot(
                        tracks_["Longitude"],
                        tracks_["Latitude"],
                        color="xkcd:green",
                        mec="xkcd:green",
                        marker="s",
                        markersize=1.5,
                        alpha=0.5,
                    )

            # Represent the marks
            if marks is not None:
                style_ = (
                    ("xkcd:azure", "dotted"),
                    ("xkcd:azure", "solid"),
                    ("xkcd:azure", "solid"),
                    ("xkcd:aquamarine", "solid"),
                    ("xkcd:aquamarine", "solid"),
                )
                for nm_, mark_ in enumerate(marks):
                    if isinstance(mark_, np.ndarray) or (
                        isinstance(mark_, list) and len(mark_) > 0
                    ):
                        long = mark_[:, 1]
                        lat = mark_[:, 0]
                        plot_ax.plot(
                            long,  # Longitude
                            lat,  # Latitude
                            color=style_[nm_ % 5][0],
                            linestyle=style_[nm_ % 5][1],
                        )

            # Use the insert
            if drawing_function is not None:
                drawing_function[0](**drawing_function[1] | {"ax": plot_ax})

        # Figure title
        if len(flights) == 1:
            if "Copy" in flights[0].flight_id:
                id_ = flights[0].flight_id.split("Copy")[0]
                plt.title(f"{title} {id_}", fontsize=12)
            else:
                plt.title(f"{title} {flights[0].flight_id}", fontsize=12)
        else:
            plt.title(f"{title} {len(flights)} flights", fontsize=12)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_very_long_list(
        flights,
        wide=300,
        ratio=1,
        move_east=0,
        move_north=0,
        alpha=0.01,
        cmap="brg",
        title="",
    ):
        """
        Plotting a very long list of flight objects

        Will use rainbow mode with very low alpha,
        thinner line and with less code for speed,
        also no printed warnings

        Parameters
        ----------
        flights : the list of flights to render (required)
        wide : horizontal length of the window in km (default 300km)
        alpha : alpha for tracks in matplotlib rendering
            (default min 0.01)
        cmap : color map (default "brg")
        move_east: to move the centre of the pic towards east
            (default 0km - centered)
        move_north: to move the centre of the pic towards north
            (default 0km - centered)
        ratio: to make the render window wider
            (default 1 - square)
        title: to add text to the title

        Static method for Flight class
        """

        # Plot helper
        plot_ax, _, _, _, _ = Flight.plot_helper(
            wide, move_east, move_north, ratio
        )

        # Color range
        many_cmap = plt.get_cmap(cmap)

        # Flight list traversal
        for num, flight_ in enumerate(flights):
            if len(flight_.tracks) > 0:
                # Surveillance rainbow only
                color_ = many_cmap(num / (len(flights) - 0))
                plot_ax.plot(
                    flight_.tracks["Longitude"],
                    flight_.tracks["Latitude"],
                    color=color_,
                    marker="s",
                    markersize=0.1,
                    alpha=alpha,
                )

        # Figure title
        plt.title(f"{title} {len(flights)} flights", fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_helper(wide, move_east, move_north, ratio):
        """
        Internal static method to provide common
        background graphics and geometry to the
        track rendering functions

        Will return the latitude and longitude range
        """

        # Calculating plotting range
        kmdegreelat = 110.574
        min_lat = Flight.WSSS_LAT - (wide - move_north) * ratio / kmdegreelat
        max_lat = Flight.WSSS_LAT + (wide + move_north) * ratio / kmdegreelat

        kmdegreelng = 111.320 * math.cos(Flight.WSSS_LAT / 180 * math.pi)
        min_lng = Flight.WSSS_LNG - (wide - move_east) / kmdegreelng
        max_lng = Flight.WSSS_LNG + (wide + move_east) / kmdegreelng

        fig = plt.figure(
            # figsize=(
            #     Flight.DEFAULTWIDEFIGSIZE[0],
            #     Flight.DEFAULTWIDEFIGSIZE[1] * ratio,
            # ),
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )

        plot_ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plot_ax.set_extent([min_lng, max_lng, min_lat, max_lat])

        plot_ax.add_feature(cfeature.COASTLINE, alpha=0.25)
        plot_ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.25)
        plot_ax.add_feature(cfeature.OCEAN, alpha=0.15)
        # plot_ax.add_feature(cfeature.LAND, alpha=0.5)
        plot_ax.add_feature(cfeature.LAKES, alpha=0.5)
        plot_ax.add_feature(cfeature.RIVERS, alpha=0.5)

        # # Plot Singapore FIR sectors
        # for sector in Flight.SINGAPORE_FIR_SECTORS:
        #     sector.plot(ax=plot_ax, linestyle=":", color="xkcd:grey")

        # Plot Singapore FIR
        Flight.SINGAPORE_FIR[0].plot(
            ax=plot_ax,
            linestyle=":",
            facecolor="none",
            edgecolor="xkcd:grey",
        )

        return plot_ax, min_lat, max_lat, min_lng, max_lng

    @staticmethod
    def render_svg():
        """Change rendering mode to svg"""
        backend_inline.set_matplotlib_formats("svg")

    @staticmethod
    def render_png():
        """Change rendering mode to retina"""
        backend_inline.set_matplotlib_formats("png")

    @staticmethod
    def render_retina():
        """Change rendering mode to retina"""
        backend_inline.set_matplotlib_formats("retina")

    @staticmethod
    def plot_elevation_profile(
        flights,
        xlim=500,
        ylim=None,
        factortime=1,
        endalign=True,
        alpha=0.1,
        marks=None,
        startmarks=None,
    ):
        """
        Plots the elevation profile of a set of flights
        Can add marks but if adding marks a position to start them in the
        graphic is required (n where prediction started)
        Factortime is default 1 but does need to be changed if tracks
        have been downsampled
        """
        plt.figure(
            figsize=Flight.DEFAULTWIDEFIGSIZE,
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )
        ax1 = plt.subplot()

        # Traverse the flights
        for flight_ in flights:
            ax1.plot(
                (
                    flight_.tracks["Time"] - flight_.tracks.iloc[-1]["Time"]
                    if endalign
                    else flight_.tracks["Time"]
                ),
                flight_.tracks["Altitude"],
                "-",
                color="darkgreen",
                alpha=alpha,
                linewidth=2,
            )

        # Represent the marks
        if marks is not None:
            colo_ = (
                "xkcd:tangerine",
                "xkcd:cerise",
                "xkcd:azul",
                "xkcd:aqua blue",
            )
            for nm_, mark_ in enumerate(marks):
                for np_, point_ in enumerate(mark_):
                    ax1.plot(
                        factortime(startmarks[nm_] + np_),
                        point_[2],  # Altitude
                        color=colo_[nm_ % 4],
                        alpha=0.5,
                        marker="X",
                        markersize=5,
                    )
        if endalign:
            ax1.set_xlim(-xlim, 0)
        else:
            ax1.set_xlim(0, xlim)
        if ylim is not None:
            if type(ylim) is tuple:
                ax1.set_ylim(ylim[0], ylim[1])
            else:
                ax1.set_ylim(0, ylim)
        ax1.grid(axis="y", zorder=-1)
        plt.title(f"Profiles for {len(flights)} flights", fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_elevation_histogram(
        flights,
        bins=100,
        alpha_threshold=0,
        xlim=None,
        ylim=None,
        endalign=True,
        cmap="rainbow",
    ):
        """
        Plots the elevation histogram density of a set of flights
        """
        plt.figure(
            figsize=Flight.DEFAULTWIDEFIGSIZE,
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )
        ax1 = plt.subplot()

        time_list = []
        altitude_list = []

        # Traverse the flights
        for flight_ in flights:
            time_list += (
                (
                    flight_.tracks["Time"] - flight_.tracks.iloc[-1]["Time"]
                ).to_list()
                if endalign
                else flight_.tracks["Time"].to_list()
            )

            altitude_list += flight_.tracks["Altitude"].to_list()

        time_list = np.array(time_list)
        altitude_list = np.array(altitude_list)

        H, yedges, xedges = np.histogram2d(altitude_list, time_list, bins=bins)
        alpha = (H > alpha_threshold).astype(int)

        ax1.pcolormesh(xedges, yedges, H, alpha=alpha, cmap=cmap)

        if endalign and xlim is not None:
            ax1.set_xlim(-xlim, 0)
        elif not endalign and xlim is not None:
            ax1.set_xlim(0, xlim)
        else:
            ax1.set_xlim(min(time_list), max(time_list))

        if ylim is not None:
            ax1.set_ylim(altitude_list.min(), ylim)
        else:
            ax1.set_ylim(altitude_list.min(), max(altitude_list))

        ax1.grid(axis="y", zorder=-1)
        plt.title(f"Profiles for {len(flights)} flights", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_expanded(self):
        """
        Shows a graphic of the evolution of the main flight
        parameters within the track update including the expanded
        parameters

        It represents:
        * x axis is time: one sample each four seconds

        * y axis on the left:
            * cos_b_1: cos angle from heading direction to current fix
            * cos_b_2: cos angle from heading direction to next fix
            * c_1: distance to current airway (gold)
            * c_2: distance to next airway (orange)
            * d_1: distance to current or closest fix (blue)
            * d_2: distance to next fix (dark violet)

        * y axis on the right:
            * l_1_l: longitudinal speed current fix (turquoise dotted)
            * l_1_t: transverse speed current fix (turquoise)
            * l_2_l: longitudinal speed next fix (lime green dotted)
            * l_2_t: transverse speed next fix (lime green)
            * altitude (tomato dashed)
            * level 290 (8839m) as a reference (tomato dotted horizontal line)

        """
        if len(self.tracks) == 0:
            print(f"{self.flight_id}: No track updates to plot")
            return
        elif "ClosestFix" not in self.tracks:
            print(f"{self.flight_id}: No expanded details to plot")
            return

        plt.figure(
            figsize=Flight.DEFAULTWIDEFIGSIZE,
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )
        colors = [
            Flight.DEFAULTCMAP((n) / (len(self.fixes) - 1))
            for n in self.tracks["ClosestFix"]
        ]

        # Left side
        ax1 = plt.subplot()
        # drawing the grid and fix labels
        ax1.plot(
            [
                self.tracks.iloc[0]["Time"],
                self.tracks.iloc[len(self.tracks) - 1]["Time"],
            ],
            [0, 0],
            "--",
            color="lightgrey",
        )

        for row in (
            self.tracks[["Time", "ClosestFix"]]
            .drop_duplicates(subset="ClosestFix")
            .itertuples()
        ):
            ax1.plot(
                [row.Time, row.Time],
                [-math.pi, math.pi],
                "--",
                color="lightgrey",
            )
            ax1.text(
                s=self.fixes.loc[row.ClosestFix, "FixName"],
                x=row.Time,
                y=1.75,
                color=Flight.DEFAULTTEXTCOLOR,
                rotation=30,
                horizontalalignment="left",
            )

        # Left side
        ax1.scatter(
            self.tracks["Time"],
            self.tracks["cos_b_1"],
            marker="_",
            c=colors,
            label="cos_b_1: cos angle dir to current fix",
        )
        ax1.scatter(
            self.tracks["Time"],
            self.tracks["cos_b_2"],
            marker="|",
            c=colors,
            label="cos_b_2: cos angle dir to next fix",
        )
        ax1.plot(
            self.tracks["Time"],
            self.tracks["c_1"] * 5,
            "-",
            color="gold",
            label="c_1: dist to current airway (scaled 5:1)",
        )
        ax1.plot(
            self.tracks["Time"],
            self.tracks["c_2"] * 5,
            ":",
            color="orange",
            label="c_2: dist to next airway (scaled 5:1)",
        )
        ax1.plot(
            self.tracks["Time"],
            self.tracks["d_1"],
            "-",
            color="blue",
            label="d_1: dist to current fix",
        )
        ax1.plot(
            self.tracks["Time"],
            self.tracks["d_2"],
            ":",
            color="darkviolet",
            label="d_2: dist to next fix",
        )

        ax1.set_ylim(-2, 2)
        plt.legend(loc="lower left")

        # Right side
        ax2 = ax1.twinx()

        # Level 290 reference
        # (29000ft - 8839.2m - 1:50)
        ax2.plot(
            [
                self.tracks.iloc[0]["Time"],
                self.tracks.iloc[len(self.tracks) - 1]["Time"],
            ],
            [177, 177],
            ":",
            color="tomato",
            label="level 290 (8839.2m) (scaled 1:50)",
        )
        ax2.plot(
            self.tracks["Time"],
            self.tracks["l_1_l"],
            ":",
            color="turquoise",
            label="l_1_l: longitudinal speed current fix",
        )
        ax2.plot(
            self.tracks["Time"],
            self.tracks["l_1_t"],
            "-",
            color="turquoise",
            label="l_1_t: transverse speed current fix",
        )
        ax2.plot(
            self.tracks["Time"],
            self.tracks["l_2_l"],
            ":",
            color="limegreen",
            label="l_2_l: longitudinal speed next fix",
        )
        ax2.plot(
            self.tracks["Time"],
            self.tracks["l_2_t"],
            "-",
            color="limegreen",
            label="l_2_t: transverse speed next fix",
        )
        ax2.plot(
            self.tracks["Time"],
            self.tracks["Altitude"] / 50,
            "--",
            color="tomato",
            label="altitude (scaled 1:50)",
        )

        ax2.set_ylim(-250, 250)
        ax2.set_xlim(
            self.tracks.iloc[0]["Time"],
            self.tracks.iloc[len(self.tracks) - 1]["Time"],
        )
        plt.legend(loc="lower right")

        plt.title(self.flight_id, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_profile(self, start=0, end=-1):
        """
        Shows a graphic of the evolution of the main flight
        parameters within the track update including the expanded
        parameters

        It represents:
        * x axis is time: one sample each four seconds

        * y axis on the left:
            * d_1: distance to current or closest fix (blue)
            * d_2: distance to next fix (dark violet)

        * y axis on the right:
            * altitude (tomato dashed)
            * level 290 (8839m) as a reference (tomato dotted horizontal line)

        """
        if len(self.tracks) == 0 or start >= len(self.tracks) or start == end:
            print(f"{self.flight_id}: No track updates to plot")
            return
        elif "ClosestFix" not in self.tracks:
            print(f"{self.flight_id}: No expanded details to plot")
            return

        plt.figure(
            figsize=Flight.DEFAULTWIDEFIGSIZE,
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )

        max_wp_dist = max(self.tracks["d_1"].max(), self.tracks["d_2"].max())
        # Left side
        ax1 = plt.subplot()
        # drawing the grid and fix labels

        tracks_ = self.tracks.iloc[start:end]

        separations = (
            tracks_[["Time", "ClosestFix"]]
            .drop_duplicates(subset="ClosestFix")
            .itertuples()
        )
        wp_positions, wp_closestfix = [], []
        for row in separations:
            ax1.plot(
                [row.Time, row.Time],
                [0, max_wp_dist],
                ":",
                color="darkgrey",
            )
            wp_positions.append(row.Time)
            wp_closestfix.append(row.ClosestFix)

        wp_positions += [tracks_.Time.iloc[-1]]
        wp_centre = [
            (a + b) / 2 for a, b in zip(wp_positions[:-1], wp_positions[1:])
        ]
        wp_space = [
            (b - a) / tracks_.Time.iloc[-1] > 0.08
            for a, b in zip(wp_positions[:-1], wp_positions[1:])
        ]

        for n_, close_ in enumerate(wp_closestfix):
            ax1.text(
                s=self.fixes.FixName.iloc[close_],
                x=wp_centre[n_],
                y=max_wp_dist * 1.05 if wp_space[n_] else max_wp_dist * 1.09,
                color=Flight.DEFAULTTEXTCOLOR,
                rotation=0 if wp_space[n_] else 90,
                ha="center",
                va="center" if wp_space[n_] else "top",
            )

        # Left side
        ax1.plot(
            tracks_["Time"],
            tracks_["d_1"],
            "-",
            color="blue",
            label="d_1: dist to closest fix",
        )
        ax1.plot(
            tracks_["Time"],
            tracks_["d_2"],
            ":",
            color="darkviolet",
            label="d_2: dist to next fix",
        )

        ax1.set_ylim(-max_wp_dist * 0.19, max_wp_dist * 1.1)
        plt.yticks(fontsize=10, rotation=0)
        plt.xticks(fontsize=10, rotation=0)
        plt.legend(loc="lower left", fontsize=10)

        # Right side
        ax2 = ax1.twinx()

        max_alt = tracks_["Altitude"].max()

        # Level 290 reference
        # (29000ft - 8839.2m - 1:50)
        ax2.plot(
            [
                tracks_.iloc[0]["Time"],
                tracks_.iloc[len(tracks_) - 1]["Time"],
            ],
            [8839.2, 8839.2],
            ":",
            color="tomato",
            label="FL290",
        )
        ax2.plot(
            tracks_["Time"],
            tracks_["Altitude"],
            "-",
            color="tomato",
            label="altitude",
        )

        ax2.plot(
            tracks_["Time"],
            tracks_["theta"] / math.pi * 180 * 100 / Flight.METER_2_FEET,
            "-",
            color="yellow",
            label="TRK",
        )
        ax2.plot(
            tracks_["Time"],
            tracks_["b_1"] / math.pi * 180 * 100 / Flight.METER_2_FEET,
            "--",
            color="orange",
            label="$\\beta$ close WP",
        )
        ax2.plot(
            tracks_["Time"],
            tracks_["b_2"] / math.pi * 180 * 100 / Flight.METER_2_FEET,
            ":",
            color="orange",
            label="$\\beta$ next WP",
        )

        ax2.set_ylim(-max_alt * 0.19, max_alt * 1.1)
        ax2.set_xlim(
            tracks_.iloc[0]["Time"],
            tracks_.iloc[len(tracks_) - 1]["Time"],
        )

        labels = [item.get_text() for item in ax2.get_yticklabels()]
        labels = [
            (
                label_
                + " (FL"
                + str(
                    round(
                        int(label_.replace("\u2212", "-"))
                        * Flight.METER_2_FEET
                        / 100
                    )
                )
                + ")"
                if int(label_.replace("\u2212", "-")) >= 0
                else ""
            )
            for label_ in labels
        ]
        ax2.set_yticklabels(labels)

        plt.legend(loc="lower right", fontsize=10, ncol=3)
        plt.yticks(fontsize=10, rotation=0)
        plt.title(self.flight_id, fontsize=16)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_histogram(data, bins=20, title=""):
        """
        Plot a histogram with specific data
        can specify bins and title as well
        """

        plt.figure(
            figsize=Flight.DEFAULTWIDEFIGSIZE,
            dpi=Flight.DEFAULTDPI,
            facecolor=Flight.DEFAULTFACECOLOR,
            edgecolor=Flight.DEFAULTEDGECOLOR,
        )

        plt.subplot()
        plt.hist(data, bins=bins)

        plt.title(f"{title} {len(data)} flights", fontsize=16)
        plt.tight_layout()
        plt.show()

    def stats(self):
        """
        Shows some additional stats describing:

        * altitude
        * cos_b_1, cos_b_2
        * c_1, c_2
        * d_1, d_2
        * l_1_l, l_1_t, l_2_l, l_2_t
        """
        if len(self.tracks) == 0:
            print(f"{self.flight_id}: No track updates to describe")
            return

        if "ClosestFix" not in self.tracks:
            print(f"{self.flight_id}: No expanded details to describe")
            return self.tracks[["Altitude"]].describe().T

        return (
            self.tracks[
                [
                    "Altitude",
                    "cos_b_1",
                    "cos_b_2",
                    "c_1",
                    "c_2",
                    "d_1",
                    "d_2",
                    "l_1_l",
                    "l_1_t",
                    "l_2_l",
                    "l_2_t",
                ]
            ]
            .describe()
            .T
        )
