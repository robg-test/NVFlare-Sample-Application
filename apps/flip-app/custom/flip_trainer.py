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


from utils.flip_constants import FlipEvents
from flip import FLIP
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_traceback
from trainer import FLIP_TRAINER as UPLOADED_TRAINER


class RUN_TRAINER(Executor):
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        project_id="",
        query="",
    ):
        """Executes the uploaded trainer and handles any errors.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            project_id (str, optional): The ID of the project the model belongs to.
            query (str, optional): The cohort query that is associated with the project.
        """
        super(RUN_TRAINER, self).__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars
        self._project_id = project_id
        self._query = query
        self._flip_trainer = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if self._flip_trainer == None:
                    self._flip_trainer = UPLOADED_TRAINER(
                    self._lr,
                    self._epochs,
                    self._train_task_name,
                    self._submit_model_task_name,
                    self._exclude_vars,
                    self._project_id,
                    self._query,
                )
            
            return self._flip_trainer.execute(task_name, shareable, fl_ctx, abort_signal)
        except Exception as e:
            engine = fl_ctx.get_engine()
            if engine is None:
                self.logger.error("Error: no engine in fl_ctx, cannot fire log exception event")
                return

            formatted_exception = secure_format_traceback()

            dxo = DXO(data_kind=DataKind.ANALYTIC, data={"exception": formatted_exception})
            event_data = dxo.to_shareable()

            fl_ctx.set_prop(FLContextKey.EVENT_DATA, event_data, private=True, sticky=False)

            fl_ctx.set_prop(
                FLContextKey.EVENT_SCOPE,
                value=EventScope.FEDERATION,
                private=True,
                sticky=False,
            )

            fl_ctx.set_prop(
                FLContextKey.EVENT_ORIGIN, "flip_client", private=True, sticky=False
            )

            engine.fire_event(FlipEvents.LOG_EXCEPTION, fl_ctx)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)