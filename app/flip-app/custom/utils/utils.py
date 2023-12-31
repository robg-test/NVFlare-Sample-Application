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

from uuid import UUID


class Utils(object):
    @staticmethod
    def is_valid_uuid(val):
        try:
            UUID(str(val))
            return True
        except ValueError:
            return False

    @staticmethod
    def is_string_empty(val):
        return val.strip() == ""
