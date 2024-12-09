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

#include <chrono>
#include <thread>

#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/comm_async_recorder.h"
#include "paddle/phi/core/distributed/comm_async_time_profiler.h"

namespace phi {
namespace distributed {

thread_local std::list<std::shared_ptr<CommAsyncRecorder>>
    CommAsyncTimeProfiler::recorders_list_;

CommAsyncTimeProfiler::CommAsyncTimeProfiler() : terminated_(false) {
  loop_thread_ = std::thread(
      &phi::distributed::CommAsyncTimeProfiler::RecordTimeLoop, this);
}

CommAsyncTimeProfiler::~CommAsyncTimeProfiler() { Stop(); }

void CommAsyncTimeProfiler::UpdateGroupInfo(int gid, float recorder_time) {
  profiling_infos_[gid] += recorder_time;
}

void CommAsyncTimeProfiler::ResetGroupInfo() {
  for (auto& p : profiling_infos_) {
    p.second = 0.f;
  }
}

std::unordered_map<int, float> CommAsyncTimeProfiler::GetProfiles() {
  CommAsyncRecorder::SynchronizeAllRecorders();
  while (!recorders_list_.empty()) {
    CheckAndUpdateProfilingInfos();
  }

  std::unique_lock<std::mutex> recoders_lk(recoders_mutex_);
  std::unordered_map<int, float> ret(profiling_infos_);
  ResetGroupInfo();
  return ret;
}

void CommAsyncTimeProfiler::AddRecorder(
    std::shared_ptr<CommAsyncRecorder> recorder) {
  if (!terminated_.load()) {
    recorders_list_.push_back(std::move(recorder));
  }
}

void CommAsyncTimeProfiler::Stop() {
  terminated_.store(true);
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void CommAsyncTimeProfiler::CheckAndUpdateProfilingInfos() {
  std::unique_lock<std::mutex> recoders_lk(recoders_mutex_);
  for (auto iter = recorders_list_.begin(); iter != recorders_list_.end();) {
    auto recorder = *iter;
    if (!recorder->IsStart() && recorder->QueryStart()) {
      // event start
      recorder->Start();
    }
    if (recorder->IsStart() && recorder->QueryEnd()) {
      // event end
      float recorder_time = recorder->RecordTime();
      UpdateGroupInfo(recorder->GetGid(), recorder_time);
      recorder->EventDestroy();
      iter = recorders_list_.erase(iter);
    } else {
      ++iter;
    }
  }
}

void CommAsyncTimeProfiler::RecordTimeLoop() {
  while (!terminated_.load()) {
    CheckAndUpdateProfilingInfos();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

}  // namespace distributed
}  // namespace phi
