# Copyright 2021, 2022, 2023 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022, 2023 Huawei Technologies Co., Ltd
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
# ============================================================================

from luojianet_ms.train.callback import Callback
import time
from luojianet_ms.communication.management import get_rank
import os


class BenchmarkTraining(Callback):
    """BenchmarkTraining callback util"""

    def __init__(self, eval_model, eval_dataset):
        super(BenchmarkTraining, self).__init__()
        # self.train_threshold = 0
        self.eval_model = eval_model
        self.eval_dataset = eval_dataset
        self.tic = 0
        self.res = []
        pass

    """Callback base class"""

    def begin(self, run_context):
        """Called once before the network executing."""
        self.tic = time.time()
        print("!!!!!!!!!!start time: " + str(self.tic))
        pass

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        pass

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        cb_params = run_context.original_args()
        atic = time.time()
        eval_result = self.eval_model.eval(self.eval_dataset, dataset_sink_mode = False)
        btime = time.time()
        dtime = btime - atic
        total_time = btime - self.tic
        print(f"----------epoch: {cb_params.cur_epoch_num}, loss: "
              f"{eval_result['0']} eval_result: {eval_result['1']}----------")
        print('One eval elapsed:' + str(dtime))
        self.res.append({'epoch': cb_params.cur_epoch_num, 'loss': eval_result['0'], 'eval_result': eval_result['1'],
                         'eval_elapsed': dtime, 'total_elapsed': total_time})
        # if eval_result > self.train_threshold:
        #     print("================================")
        #     print(f"Stop at epoch: {cb_params.cur_epoch_num}, which eval
        #     result is {eval_result}!")
        #     run_context.request_stop()

    def step_begin(self, run_context):
        """Called before each step beginning."""
        pass

    def step_end(self, run_context):
        """Called after each step finished."""
        pass

    def end(self, run_context):
        """Called once after network training."""
        toc = time.time()
        dtime = toc - self.tic
        print('Training Elapsed: %s. \n ' % dtime)

        # Save the eval data
        folder = '/cache/eval/'
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except Exception:
            print('WARNING: Could not create directory！')
        f = open(f'/cache/eval/eval_result_list_{get_rank()}.txt', 'w')
        f.write('Training Elapsed: %s. \n ' % dtime)
        f.write("=========Epoch Data========\n")
        f.write("EpochNum   Loss         EvalResult   EvalElapsed   TotalElapsed\n")
        for i in self.res:
            f.write("{0[epoch]: <4d}       {0[loss]:0<10.8f}   {0[eval_result]:0<10.8f}   {0[eval_elapsed]:0<11.8f}   {0[total_elapsed]:0<15.8f}\n".format(
                    i))
        f.close()
