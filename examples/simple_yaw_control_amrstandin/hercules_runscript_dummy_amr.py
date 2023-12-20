# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import sys

from hercules.dummy_amr_wind import launch_dummy_amr_wind

# Check that one command line argument was given
if len(sys.argv) != 2:
    raise Exception("Usage: python emu_runscript_dummy_amr.py <amr_input_file>")

# # Get the first command line argument
# This is the name of the file to read
amr_input_file = sys.argv[1]
print(f"Running AMR-Wind dummy with input file: {amr_input_file}")


launch_dummy_amr_wind(amr_input_file)
