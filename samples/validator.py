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


from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class FLIP_VALIDATOR(Executor):
    def __init__(
        self, 
        validate_task_name=AppConstants.TASK_VALIDATION,
        project_id="",
        query=""
    ):
        """A blank validator that will handle a "validate" task.

        Args:
            validate_task_name (str, optional): Task name for validate. Defaults to "validate".
            project_id (str, optional): The ID of the project the model belongs to.
            query (str, optional): The cohort query that is associated with the project.
        """
        super(FLIP_VALIDATOR, self).__init__()
        self._validate_task_name = validate_task_name
        self._project_id = project_id
        self._query = query

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == self._validate_task_name:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
