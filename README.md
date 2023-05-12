<!--
  ~ Copyright 2022 Guy’s and St Thomas’ NHS Foundation Trust
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->





# Sample FLIP Application

![example workflow](https://github.com/AI4VBH/flip-sample-application/actions/workflows/nvflare-test.yml/badge.svg)


The purpose of this repository is to provide a sample application that can be developed and tested locally before being uploaded onto FLIP.
## **How to use this repository**

This repository contains a simplified sample application that replicates a running application on FLIP. This application runs on [NVIDIA Flare](https://github.com/NVIDIA/NVFlare).
Download or clone this repository and use the `./app` directory as a sample application to run in NVIDIA Flare.

## Setup NVFlare Workspace without Docker 💻 
Follow the nvflare [Installation](https://nvflare.readthedocs.io/en/2.0/installation.html) instructions.

Install the required dependencies from `requirements.txt`

``python3 -m pip install -r requirements.txt``

## Setup NVFlare workspace with Docker 🐳 
A dockerfile has been provided that contains all the required dependencies installed.

Copy any NVFlare Applications you wish to use to the ``/apps`` directory. The nvflare fl-simulator can then be used in 
the same manner described above.

### 1. Build Image & Run Container
Use the docker build commands and run the container

```shell 
docker build . -t nvflare-in-one``
docker run nvflare-in-one``
```

### 2. Inserting Files into the container
If you wish to test the utilization of resources in your model E.g. DICOMS 
It's recommended to use the copy command to copy the files to the container

```
docker cp <path_to_your_resource> <container_name>:/dir
```

You can then edit the flip.py module and change the response to the parent directory of where your files were copied to
e.g. ``/dir``

Within FLIP the files are identifiable by the accession number, as returned within the dataframe

``/dir/<accession_number>``

# Running the Sample Application flip-app
The `./apps/flip-app` directory contains a replica of an application that can be run on FLIP. Some modules are stubbed with only a return type set. There are two main files that FLIP requires before running any training - `trainer.py` and `validator.py`. Both of these files you will find within `flip-app/app/custom` and contain a working example application that can be used as a starting point.
This example uses [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

These two files are where you should add your own application code. The `./samples` directory contains empty templates of both `trainer.py` and `validator.py`.

### Run the fl-simulator

``nvflare simulator ./app -w ./workspace-folder -n 2 -t 2``

If running from a docker container then you'll need to exec into that container first.

```shell 
docker exec -it <name> bash``
nvflare simulator ./app -w ./workspace-folder -n 2 -t ``
```

Assuming the `./app` directory is set up in the same way as the sample app repository. The fl-simulator will create a local
nvflare server with two clients and then proceed to run the sample application. Changes can be quickly tested by using
this command. Simply update the parts of the sample app you wish to change and run the same command. (The app will need
to be copied back into the docker container if running from docker)

### Output
The output of training can be found within the designated workspace folder passed in through the `-w` argument. The
simulator will create this directory for you if it doesn't exist.

This directory will contain the trained server and local models along with the cross validation results and log file

### Using GPUs
The fl-simulator command is setup to run on cpu but can be configured to use gpus if available. See more information in the nvflare documentation
[FL-Simulator 2.2 docs](https://nvflare.readthedocs.io/en/2.2/getting_started.html#the-fl-simulator)

## **FLIP methods**
The following methods are available to be used in training, located in `flip.py`:

- `get_dataframe(self, project_id: str, query: str) -> DataFrame`
  This retrieves data in the form of a Dataframe containing, at the minimum, accession IDs.
  The method takes in the project ID and the project query as parameters. These values are
  already passed in as parameters to the trainer to be used.

- `get_by_accession_number(self, project_id: str, accession_id: str) -> Path`
  This downloads scans and places them in a directory made available for NVFlare to utilise.
  The method takes in the project ID as a parameter as well as an accession ID, which can be 
  obtained from `get_dataframe`. It returns the path to where the scans are stored.

- `add_resource(self, project_id: str, accession_id: str, scan_id: str, resource_id: str, files: List[str])`
  This allows uploading scans to XNAT under the project that the model to. Scans are to be placed 
  in the `uploads` directory.
  The method does not have a return type. It supports the following required parameters: 
    - project ID
    - accession ID 
    - scan ID (ID/label of the directory at the scan level)
    - resource ID (ID/label of the directory at the resource level) 
    - a list of files corresponding to the names of the files that reside within the `uploads` 
    directory that you wish to upload, e.g. [`scan-1.dcm`, `scan-2.dcm`, ...].

  The list of files could also point to locations in subfolders relative to the uploads directory, 
  e.g. [`subfolder/scans/scan-1.dcm`, `scan-2.dcm`],
  where `scan-1` has the path `uploads/subfolder/scans/scan-1.dcm` and `scan-2` has the path `uploads/scan-2.dcm`.

- `update_status(self, model_id: str, new_model_status: ModelStatus)`
  This method is for internal use only and is not to be called by the trainer.

- `send_metrics_value(self, label: str, value: float, fl_ctx: FLContext)`
  This method raises an event which allows the sending of metrics data back to the central hub.
  The FL Server workflow component listens for these events and populates the data with the current global round and
  model id before storing.
  The method has no return type. It supports the following required parameters:
  - label (Any string is valid. The value will be stored against this label)
  - value
  Some constant values are provided under `FlipMetricsLabel` in `utils/flip_constants` but is not required to use these

- `handle_metrics_event(self, event_data: Shareable, global_round: int, model_id: str)`
  This method is for internal use only and is not to be called by the trainer.
- `send_handled_exception(self, formatted_exception: str, client_name: str, model_id: str)`
  This method is for internal use only and is not to be called by the trainer.

### Import FLIP and call methods
- Import the module: `from flip import FLIP`
- Make an instance of the class: `flip = FLIP()`
- Use the instance to call one of the methods: `dataframe = flip.get_dataframe(project_id, query)`

This will allow successful calls to any the methods in `flip.py`.

## **Load config into trainer**
The `config.json` file allows variables to be defined and utilised within the trainer files.

An example of a config file:
```
{
	"GLOBAL_ROUNDS": 1,
	"LOCAL_ROUNDS": 1,
	"ROUND_HALF_UP": true,
	"LOSS_FUNCTION_START_VALUE": 1.0,
	"DAYS_OF_WEEK": [
		"mon",
		"tue",
		"wed",
		"thu",
		"fri",
		"sat",
		"sun"
	]
}
```

To use the config file within the trainer:

```
import json


self.config = {}

current_dir = os.path.dirname(__file__)
config_file = os.path.join(current_dir, "config.json")

with open(config_file) as file:
   self.config = json.load(file)
```

NOTE: As the sample application is a proof of concept, updating the global and local rounds in 
the config file will not dynamically update the global and local round values.

## Note for server config
In the file `config_fed_client.json` under the Cross Site Validation workflow, a parameter named
`participating_clients` is passed in with the values `site-1` and `site-2` in a list. This may need
modifying depending on what clients you perform the training at locally. For example, if you only run
the training at `site-1`, then the list should should reflect that. Failure to do so could end up in a
loop where the server is waiting for `site-2`'s response.
