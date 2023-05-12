# Copyright 2022 Guy’s and St Thomas’ NHS Foundation Trust
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner

simulator = SimulatorRunner(
    job_folder="app",
    workspace="workspace-folder",
    n_clients=2,
    threads=2
)

run_status = simulator.run()

print(f"Simulator has finished running. Status: {run_status}")

if run_status == 1:
    raise RuntimeError("Error detected during initialisation - Check logs for error details")
    pass

if run_status == 2:
    raise RuntimeError("Error detected while training - Check logs for error details")
    pass

if run_status == 0:
    print("Run finished successfully")
