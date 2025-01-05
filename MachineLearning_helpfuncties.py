# Help functions.py

# libraries
import gpxpy
import gpxpy.gpx
import os
import pandas as pd
from fitparse import FitFile
import xml.etree.ElementTree as ET
import json

# Import function FIT-files
def import_fit(folder_path):
    """
    Imports all FIT files in a specified folder, converts them to DataFrames, and returns a dictionary
    where the keys are file names (without extensions) and the values are DataFrames.

    Args:
        folder_path (str): Path to the folder containing FIT files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    fit_dataframes = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.fit'):
            file_path = os.path.join(folder_path, file_name)

            # Open and parse the FIT file
            fitfile = FitFile(file_path)

            # Extract data from FIT file
            data = []
            for record in fitfile.get_messages('record'):
                record_data = {}
                for field in record:
                    record_data[field.name] = field.value
                data.append(record_data)

            # Convert the extracted data to a DataFrame
            df = pd.DataFrame(data)

            # Use the file name (without extension) as the key
            file_key = os.path.splitext(file_name)[0]
            fit_dataframes[file_key] = df

    return fit_dataframes



# Import function GPX-files
def import_gpx(folder_path):
    """
    Imports all GPX files in a specified folder, converts them to DataFrames, and returns a dictionary
    where the keys are file names (without extensions) and the values are DataFrames.

    Args:
        folder_path (str): Path to the folder containing GPX files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    gpx_dataframes = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.gpx'):
            file_path = os.path.join(folder_path, file_name)

            # Open and parse the GPX file
            with open(file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)

                # Initialize data list for storing extracted points
                data = []

                # Check if there are waypoints and extract data
                if gpx.waypoints:
                    for waypoint in gpx.waypoints:
                        data.append({
                            'type': 'waypoint',
                            'latitude': waypoint.latitude,
                            'longitude': waypoint.longitude,
                            'elevation': waypoint.elevation,
                            'time': waypoint.time
                        })

                # Check if there are trackpoints and extract data
                if gpx.tracks:
                    for track in gpx.tracks:
                        for segment in track.segments:
                            for point in segment.points:
                                data.append({
                                    'type': 'trackpoint',
                                    'latitude': point.latitude,
                                    'longitude': point.longitude,
                                    'elevation': point.elevation,
                                    'time': point.time
                                })

                # Convert the extracted data to a DataFrame
                if data:
                    df = pd.DataFrame(data)

                    # Use the file name (without extension) as the key
                    file_key = os.path.splitext(file_name)[0]
                    gpx_dataframes[file_key] = df

    return gpx_dataframes


# Import function TCX-files
import os
import xml.etree.ElementTree as ET
import pandas as pd

def import_tcx(folder_path):
    """
    Imports all TCX files in a specified folder, converts them to DataFrames, and returns a dictionary
    where the keys are file names (without extensions) and the values are DataFrames.

    Args:
        folder_path (str): Path to the folder containing TCX files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    tcx_dataframes = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tcx'):
            file_path = os.path.join(folder_path, file_name)

            # Parse the TCX file
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract data from TCX file (assuming Garmin TCX namespace)
            namespace = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
            data = []

            for trackpoint in root.findall('.//ns:Trackpoint', namespace):
                # Initialize a dictionary for each trackpoint
                point_data = {
                    'time': trackpoint.find('ns:Time', namespace).text if trackpoint.find('ns:Time', namespace) is not None else None,
                    'latitude': float(trackpoint.find('.//ns:LatitudeDegrees', namespace).text) if trackpoint.find('.//ns:LatitudeDegrees', namespace) is not None else None,
                    'longitude': float(trackpoint.find('.//ns:LongitudeDegrees', namespace).text) if trackpoint.find('.//ns:LongitudeDegrees', namespace) is not None else None,
                    'elevation': float(trackpoint.find('ns:AltitudeMeters', namespace).text) if trackpoint.find('ns:AltitudeMeters', namespace) is not None else None,
                    'heart_rate': int(trackpoint.find('.//ns:HeartRateBpm/ns:Value', namespace).text) if trackpoint.find('.//ns:HeartRateBpm/ns:Value', namespace) is not None else None
                }

                # Check and add extra fields (cadence, speed, distance, etc.)
                # Cadence
                cadence = trackpoint.find('.//ns:Cadence', namespace)
                if cadence is not None:
                    point_data['cadence'] = int(cadence.text)
                else:
                    point_data['cadence'] = None

                # Speed
                speed = trackpoint.find('.//ns:Speed', namespace)
                if speed is not None:
                    point_data['speed'] = float(speed.text)
                else:
                    point_data['speed'] = None

                # Distance
                distance = trackpoint.find('.//ns:DistanceMeters', namespace)
                if distance is not None:
                    point_data['distance'] = float(distance.text)
                else:
                    point_data['distance'] = None

                # Power (for cycling, if available)
                power = trackpoint.find('.//ns:Power', namespace)
                if power is not None:
                    point_data['power'] = int(power.text)
                else:
                    point_data['power'] = None

                # Append the trackpoint data to the data list
                data.append(point_data)

            # Convert the extracted data to a DataFrame
            df = pd.DataFrame(data)

            # Use the file name (without extension) as the key
            file_key = os.path.splitext(file_name)[0]
            tcx_dataframes[file_key] = df

    return tcx_dataframes


# Import function JSON-details-files
def import_details_json(folder_path):
    """
    Imports all JSON files einding with _details in a specified folder, converts them to DataFrames, and returns a dictionary
    where the keys are file names (without extensions) and the values are DataFrames.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    json_details_dataframes = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_details.json'):
            file_path = os.path.join(folder_path, file_name)

            # Read and parse the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                # Convert the JSON data to a DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)

                # Use the file name (without extension) as the key
                file_key = file_name.split('+')[0]
                #file_key = os.path.splitext(file_name)[0]
                json_details_dataframes[file_key] = df

    return json_details_dataframes


# Import function JSON-details-files
def import_summary_json(folder_path):
    """
    Imports all JSON files einding with _summary in a specified folder, converts them to DataFrames, and returns a dictionary
    where the keys are file names (without extensions) and the values are DataFrames.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    json_summary_dataframes = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_details.json'):
            file_path = os.path.join(folder_path, file_name)

            # Read and parse the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                # Convert the JSON data to a DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)

                # Use the file name (without extension) as the key
                #file_key = os.path.splitext(file_name)[0]
                file_key = file_name.split('+')[0]
                json_summary_dataframes[file_key] = df

    return json_summary_dataframes




    # Loop over the dictionary of DataFrames and flatten each DataFrame
    for file_key, df in json_dataframes.items():
        json_dataframes[file_key] = flatten_dataframe(df)

    return json_dataframes