// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

class CommAsyncRecorder;

class CommAsyncTimeProfiler {
 public:
  CommAsyncTimeProfiler();

  ~CommAsyncTimeProfiler();

  static CommAsyncTimeProfiler& GetInstance() {
    static CommAsyncTimeProfiler instance;
    return instance;
  }

  void Stop();
  void AddRecorder(std::shared_ptr<CommAsyncRecorder> recorder);

  std::unordered_map<int, float> GetProfiles();

 private:
  void RecordTimeLoop();
  void CheckAndUpdateProfilingInfos();
  void UpdateGroupInfo(int gid, float recorder_time);
  void ResetGroupInfo();

 private:
  std::atomic<bool> terminated_;
  std::mutex recoders_mutex_;
  std::thread loop_thread_;
  std::unordered_map<int, float> profiling_infos_;  // key:gid, value:time(ms)

  static thread_local std::list<std::shared_ptr<CommAsyncRecorder>>
      recorders_list_;  // to lock free
};

}  // namespace distributed
}  // namespace phi
