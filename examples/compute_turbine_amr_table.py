import pandas as pd
from scipy.interpolate import interp1d
from whoc import __file__ as whoc_file
import os

df = pd.read_csv(os.path.join(os.path.dirname(whoc_file), "..", "examples", "NREL5MW_WindSpeedRelationships.txt"), delimiter=" ")
df = df.iloc[1:] # remove units
df = df.astype("float")
target_wind_speeds = [3.0, 4.0, 5.0, 6.0, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 9.0, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
rot_speed_f = interp1d(df["WindVxi"], df["RotSpeed"])
target_rot_speeds = rot_speed_f(target_wind_speeds)
print(" ".join([str(d) for d in target_rot_speeds]))