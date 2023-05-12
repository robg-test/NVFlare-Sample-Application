# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class FlipConstants(object):
    CLEANUP: str = "cleanup"
    POST_VALIDATION: str = "post_validation"
    MIN_CLIENTS: int = 1
    INIT_TRAINING: str = "init_training"
    NIFTI_RESOURCE: str = "NIFTI"
    DICOM_RESOURCE: str = "DICOM"
    ALL_RESOURCE: str = "ALL"


class FlipEvents(object):
    TRAINING_INITIATED: str = "_training_initiated"
    RESULTS_UPLOAD_STARTED: str = "_results_upload_started"
    RESULTS_UPLOAD_COMPLETED: str = "_results_upload_completed"
    ABORTED: str = "_aborted"
    SEND_RESULT: str = "_send_result"


class ModelStatus(object):
    PENDING: str = "PENDING",
    INITIATED: str = "INITIATED",
    PREPARED: str = "PREPARED",
    TRAINING_STARTED: str = "TRAINING_STARTED",
    RESULTS_UPLOADED: str = "RESULTS_UPLOADED",
    ERROR: str = "ERROR",
    STOPPED: str = "STOPPED"


class FlipMetricsLabel(object):
    LOSS_FUNCTION: str = "LOSS_FUNCTION"
    DL_RESULT: str = "DL_RESULT"
    AVERAGE_SCORE: str = "AVERAGE_SCORE"
