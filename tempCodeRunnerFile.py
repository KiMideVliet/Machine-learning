# Main script
!pip install gpxpy

# libraries
import gpxpy
import gpxpy.gpx
import os
import pandas as pd
import MachineLearning_helpfuncties as Mhelp

files = os.listdir('/activities')  #files and directories
print(files)

dict_fit = Mhelp.import_fit('/activities')
dict_gpx = Mhelp.import_gpx('/activities')
dict_tcx = Mhelp.import_tcx('/activities')
dict_json_details = Mhelp.import_details_json('/activities')
dict_json_summ = Mhelp.import_summary_json('/activities')