# Main script
#!pip install gpxpy

# libraries
import gpxpy
import gpxpy.gpx
import os
import sys

sys.path.append('C:/Users/michel.marien_icarew/Documents/GitHub/Machine-learning')

import pandas as pd
import MachineLearning_helpfuncties as Mhelp



files = os.listdir()  #files and directories
print(files)

dict_fit = Mhelp.import_fit('/activities')
dict_gpx = Mhelp.import_gpx('/activities')
dict_tcx = Mhelp.import_tcx('/activities')
dict_json_details = Mhelp.import_details_json('/activities')
dict_json_summ = Mhelp.import_summary_json('/activities')
