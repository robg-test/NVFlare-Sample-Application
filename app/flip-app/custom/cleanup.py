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

import os
import shutil
from pathlib import Path

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, EventScope, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.security.logging import secure_format_traceback

from utils.flip_constants import FlipConstants, FlipEvents


class CleanupImages(Executor):
    def __init__(self):
        """CleanupImages takes place at the start and end of the run.
        All the images used for the training are deleted to prevent the build-up of unnecessary
        files on the storage space. Executing at the start of a run ensures that any training code
        is executed with a clean slate.

        Args:

        Raises:
        """

        super().__init__()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        try:
            if task_name == FlipConstants.INIT_TRAINING or task_name == FlipConstants.POST_VALIDATION:
                self.log_info(fl_ctx, "Cleanup task received but is not executed as part of the sample app")

                return make_reply(ReturnCode.OK)

            return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:

            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
