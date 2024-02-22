This example runs a lookup table-based wake steering controller, using FLORIS as a simulation
testbed rather than AMR-Wind. Steps to run the example:
0. Run construct_yaw_offests.py. To run in full, requires FLORIS v3; 
   however, you may switch off the optimize_yaw_offsets flag, in which case the provided offsets
   are used. Once FLORIS v4 is formally released, these options and the provided offsets will be
   removed from the example.
1. run bash_script.sh
2. run plot_output_data.py