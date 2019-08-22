// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CC_DUAL_NET_RANDOM_DUAL_NET_H_
#define CC_DUAL_NET_RANDOM_DUAL_NET_H_

#include <array>

#include "absl/synchronization/mutex.h"
#include "cc/dual_net/dual_net.h"
#include "cc/random.h"

namespace minigo {

class RandomDualNet : public DualNet {
 public:
  RandomDualNet(std::string name, uint64_t seed, float policy_stddev,
                float value_stddev);

  // Output policy is a normal distribution with a mean of 0.5 and a standard
  // deviation of policy_stddev, followed by a softmax.
  // Output value is a normal distribution with a mean of 0 and a standard
  // deviation of value_stddev. The output value is repeatedly sampled from the
  // normal distribution until a value is found in the range [-1, 1].
  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

 private:
  Random rnd_;
  const float policy_stddev_;
  const float value_stddev_;
};

class RandomDualNetFactory : public DualNetFactory {
 public:
  RandomDualNetFactory(uint64_t seed, float policy_stddev, float value_stddev);

  // Model specifies the policy and value standard deviation as a
  // colon-separated string, e.g. "0.4:0.4".
  std::unique_ptr<DualNet> NewDualNet(const std::string& model) override;

 private:
  absl::Mutex mutex_;
  Random rnd_;
  const float policy_stddev_;
  const float value_stddev_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_RANDOM_DUAL_NET_H_
