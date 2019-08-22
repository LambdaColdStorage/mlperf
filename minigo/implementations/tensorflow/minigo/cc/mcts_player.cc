// Copyright 2018 Google LLC
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

#include "cc/mcts_player.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "cc/logging.h"
#include "cc/random.h"
#include "cc/symmetries.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, const MctsPlayer::Options& options) {
  os << " inject_noise:" << options.inject_noise
     << " soft_pick:" << options.soft_pick
     << " random_symmetry:" << options.random_symmetry
     << " value_init_penalty:" << options.value_init_penalty
     << " policy_softmax_temp:" << options.policy_softmax_temp
     << " virtual_losses:" << options.virtual_losses
     << " num_readouts:" << options.num_readouts
     << " seconds_per_move:" << options.seconds_per_move
     << " time_limit:" << options.time_limit
     << " decay_factor:" << options.decay_factor
     << " random_seed:" << options.random_seed << std::flush;
  return os;
}

float TimeRecommendation(int move_num, float seconds_per_move, float time_limit,
                         float decay_factor) {
  // Divide by two since you only play half the moves in a game.
  int player_move_num = move_num / 2;

  // Sum of geometric series maxes out at endgame_time seconds.
  float endgame_time = seconds_per_move / (1.0f - decay_factor);

  float base_time;
  int core_moves;
  if (endgame_time > time_limit) {
    // There is so little main time that we're already in 'endgame' mode.
    base_time = time_limit * (1.0f - decay_factor);
    core_moves = 0;
  } else {
    // Leave over endgame_time seconds for the end, and play at
    // seconds_per_move for as long as possible.
    base_time = seconds_per_move;
    core_moves = (time_limit - endgame_time) / seconds_per_move;
  }

  return base_time *
         std::pow(decay_factor, std::max(player_move_num - core_moves, 0));
}

MctsPlayer::MctsPlayer(std::unique_ptr<DualNet> network,
                       std::unique_ptr<InferenceCache> inference_cache,
                       Game* game, const Options& options)
    : network_(std::move(network)),
      game_root_(&root_stats_, {&bv_, &gv_, Color::kBlack}),
      game_(game),
      rnd_(options.random_seed),
      options_(options),
      inference_cache_(std::move(inference_cache)) {
  // When to do deterministic move selection: 30 moves on a 19x19, 6 on 9x9.
  // divide 2, multiply 2 guarentees that white and black do even number.
  temperature_cutoff_ = !options_.soft_pick ? -1 : (((kN * kN / 12) / 2) * 2);
  root_ = &game_root_;

  if (options_.verbose) {
    MG_LOG(INFO) << "MctsPlayer options: " << options_;
    MG_LOG(INFO) << "Game options: " << game_->options();
    // A time-based seed will be used when options_.random_seed == 0, so log
    // that explicitly.
    MG_LOG(INFO) << "Random seed used: " << rnd_.seed();
  }

  InitializeGame({&bv_, &gv_, Color::kBlack});

  no_copy_ = (network_->GetInput() != NULL);

  if (no_copy_){
      no_copy_features_ = (DualNet::BoardFeatures*) network_->GetInput() + options_.thread_id * options_.virtual_losses;
      no_copy_outputs_ = network_->GetOutput() + options_.thread_id * options_.virtual_losses;
  }
}

MctsPlayer::~MctsPlayer() {
  if (options_.verbose) {
    MG_LOG(INFO) << "Inference history:";
    for (const auto& info : inferences_) {
      MG_LOG(INFO) << info.model << " [" << info.first_move << ", "
                   << info.last_move << "]";
    }
  }
}

void MctsPlayer::InitializeGame(const Position& position) {
  root_stats_ = {};
  game_root_ = MctsNode(&root_stats_, Position(&bv_, &gv_, position));
  ResetRoot();
}

void MctsPlayer::NewGame() {
  root_stats_ = {};
  game_root_ = MctsNode(&root_stats_, {&bv_, &gv_, Color::kBlack});
  ResetRoot();
}

void MctsPlayer::ResetRoot() {
  root_ = &game_root_;
  game_->NewGame();
}

bool MctsPlayer::UndoMove() {
  if (root_ == &game_root_) {
    return false;
  }
  root_ = root_->parent;
  game_->UndoMove();
  return true;
}

Coord MctsPlayer::SuggestMove() {
  auto start = absl::Now();

  // In order to correctly count the number of reads performed, the root node
  // must be expanded. The root will always be expanded unless this is the first
  // time SuggestMove has been called for a game, or PlayMove was called without
  // a prior call to SuggestMove.
  if (!root_->HasFlag(MctsNode::Flag::kExpanded)) {
    tree_search_paths_.clear();
    SelectLeaves(root_, 1, &tree_search_paths_);
    ProcessLeaves(absl::MakeSpan(tree_search_paths_), options_.random_symmetry);
  }

  if (options_.inject_noise) {
    std::array<float, kNumMoves> noise;
    rnd_.Dirichlet(kDirichletAlpha, &noise);
    root_->InjectNoise(noise, options_.noise_mix);
  }
  int current_readouts = root_->N();

  if (options_.seconds_per_move > 0) {
    // Use time to limit the number of reads.
    float seconds_per_move = options_.seconds_per_move;
    if (options_.time_limit > 0) {
      seconds_per_move =
          TimeRecommendation(root_->position.n(), seconds_per_move,
                             options_.time_limit, options_.decay_factor);
    }
    while (absl::Now() - start < absl::Seconds(seconds_per_move)) {
      TreeSearch();
    }
  } else {
    // Use a fixed number of reads.
    while (root_->N() < current_readouts + options_.num_readouts) {
      TreeSearch();
    }
  }
  int num_readouts = root_->N() - current_readouts;
  auto elapsed = absl::Now() - start;
  elapsed = elapsed * 100 / num_readouts;
  if (options_.verbose) {
    MG_LOG(INFO) << "Milliseconds per 100 reads: "
                 << absl::ToInt64Milliseconds(elapsed) << "ms"
                 << " over " << num_readouts
                 << " readouts (vlosses: " << options_.virtual_losses << ")";
    MG_LOG(INFO) << root_->CalculateTreeStats().ToString();
  }

  if (ShouldResign()) {
    return Coord::kResign;
  }

  return PickMove();
}

Coord MctsPlayer::PickMove() {
  if (root_->position.n() >= temperature_cutoff_) {
    Coord c = root_->GetMostVisitedMove();
    if (options_.verbose) {
      MG_LOG(INFO) << "Picked arg_max " << c;
    }
    return c;
  }

  // Select from the first kN * kN moves (instead of kNumMoves) to avoid
  // randomly choosing to pass early on in the game.
  std::array<float, kN * kN> cdf;

  // For moves before the temperature cutoff, exponentiate the probabilities by
  // a temperature slightly larger than unity to encourage diversity in early
  // play and hopefully to move away from 3-3s.
  for (size_t i = 0; i < cdf.size(); ++i) {
    cdf[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
  }
  for (size_t i = 1; i < cdf.size(); ++i) {
    cdf[i] += cdf[i - 1];
  }

  if (cdf.back() == 0) {
    // It's actually possible for an early model to put all its reads into pass,
    // in which case the SearchSorted call below will always return 0. In this
    // case, we'll just let the model have its way and allow a pass.
    return Coord::kPass;
  }

  float e = rnd_();
  Coord c = SearchSorted(cdf, e * cdf.back());
  if (options_.verbose) {
    MG_LOG(INFO) << "Picked rnd(" << e << ") " << c;
  }
  MG_DCHECK(root_->child_N(c) != 0);
  return c;
}

void MctsPlayer::TreeSearch() {
  tree_search_paths_.clear();
  SelectLeaves(root_, options_.virtual_losses, &tree_search_paths_);
  ProcessLeaves(absl::MakeSpan(tree_search_paths_), options_.random_symmetry);
}

void MctsPlayer::SelectLeaves(MctsNode* root, int num_leaves,
                              std::vector<MctsPlayer::TreePath>* paths) {
  int max_iterations = num_leaves * 2;
  int num_selected = 0;
  for (int i = 0; i < max_iterations; ++i) {
    auto* leaf = root->SelectLeaf();
    if (leaf->game_over() || leaf->at_move_limit()) {
      float value =
          leaf->position.CalculateScore(game_->options().komi) > 0 ? 1 : -1;
      leaf->IncorporateEndGameResult(value, root);
    } else {
      leaf->AddVirtualLoss(root);
      paths->emplace_back(root, leaf);
      if (++num_selected == num_leaves) {
        break;
      }
    }
  }
}

bool MctsPlayer::ShouldResign() const {
  return game_->options().resign_enabled &&
         root_->Q_perspective() < game_->options().resign_threshold;
}

std::string MctsPlayer::GetModelsUsedForInference() const {
  std::vector<std::string> parts;
  parts.reserve(inferences_.size());
  for (const auto& info : inferences_) {
    parts.push_back(absl::StrCat(info.model, "(", info.first_move, ",",
                                 info.last_move, ")"));
  }
  return absl::StrJoin(parts, ", ");
}

bool MctsPlayer::PlayMove(Coord c) {
  if (root_->game_over()) {
    MG_LOG(ERROR) << "can't play move " << c << ", game is over";
    return false;
  }

  // Handle resignations.
  if (c == Coord::kResign) {
    game_->SetGameOverBecauseOfResign(OtherColor(root_->position.to_play()));
    return true;
  }

  if (!root_->position.legal_move(c)) {
    MG_LOG(ERROR) << "Move " << c << " is illegal";
    // We're probably about to crash. Dump the player's options and the moves
    // that got us to this point.
    MG_LOG(ERROR) << "MctsPlayer options: " << options_;
    MG_LOG(ERROR) << "Game options: " << game_->options();
    for (int i = 0; i < game_->num_moves(); ++i) {
      const auto* move = game_->GetMove(i);
      MG_LOG(ERROR) << move->color << "  " << move->c;
    }
    return false;
  }

  UpdateGame(c);

  root_ = root_->MaybeAddChild(c);
  if (options_.prune_orphaned_nodes) {
    // Don't need to keep the parent's children around anymore because we'll
    // never revisit them during normal play.
    root_->parent->PruneChildren(c);
  }

  if (options_.verbose) {
    MG_LOG(INFO) << absl::StreamFormat("%s Q: %0.5f", name(), root_->Q());
    MG_LOG(INFO) << "Played >>" << c;
  }

  // Handle consecutive passing or termination by move limit.
  if (root_->at_move_limit()) {
    game_->SetGameOverBecauseMoveLimitReached(
        root_->position.CalculateScore(game_->options().komi));
  } else if (root_->game_over()) {
    game_->SetGameOverBecauseOfPasses(
        root_->position.CalculateScore(game_->options().komi));
  }

  return true;
}

void MctsPlayer::UpdateGame(Coord c) {
  // Record which model(s) were used when running tree search for this move.
  std::vector<std::string> models;
  if (!inferences_.empty()) {
    for (auto it = inferences_.rbegin(); it != inferences_.rend(); ++it) {
      if (it->last_move < root_->position.n()) {
        break;
      }
      models.push_back(it->model);
    }
    std::reverse(models.begin(), models.end());
  }

  // Build a comment for the move.
  auto comment = root_->Describe();
  if (!models.empty()) {
    comment =
        absl::StrCat("models:", absl::StrJoin(models, ","), "\n", comment);
  }

  // Convert child visit counts to a probability distribution, pi.
  std::array<float, kNumMoves> search_pi;
  if (root_->position.n() < temperature_cutoff_) {
    // Squash counts before normalizing to match softpick behavior in PickMove.
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = std::pow(root_->child_N(i), options_.policy_softmax_temp);
    }
  } else {
    for (int i = 0; i < kNumMoves; ++i) {
      search_pi[i] = root_->child_N(i);
    }
  }
  // Normalize counts.
  float sum = 0;
  for (int i = 0; i < kNumMoves; ++i) {
    sum += search_pi[i];
  }
  for (int i = 0; i < kNumMoves; ++i) {
    search_pi[i] /= sum;
  }

  // Update the game history.
  game_->AddMove(root_->position.to_play(), c, root_->position.stones(),
                 std::move(comment), root_->Q(), search_pi, std::move(models));
}

void MctsPlayer::ProcessLeaves(absl::Span<TreePath> paths,
                               bool random_symmetry) {
  if (paths.empty()) {
    return;
  }

  // Select symmetry operations to apply.
  symmetries_used_.resize(0);
  if (random_symmetry) {
    symmetries_used_.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
      symmetries_used_.push_back(static_cast<symmetry::Symmetry>(
          rnd_.UniformInt(0, symmetry::kNumSymmetries - 1)));
    }
  } else {
    symmetries_used_.resize(paths.size(), symmetry::kIdentity);
  }

  // Build input features for each leaf, applying random symmetries if
  // requested.
  DualNet::BoardFeatures raw_features;
  features_.resize(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto* leaf = paths[i].leaf;
    MG_CHECK(leaf->num_virtual_losses_applied > 0)
        << "Don't forget to add a virtual loss before calling ProcessLeaves";
    leaf->GetMoveHistory(DualNet::kMoveHistory, &recent_positions_);
    DualNet::SetFeatures(recent_positions_, leaf->position.to_play(),
                         &raw_features);

    if (network_->GetInputLayout() == DualNet::InputLayout::kNCHW) {
      using OutIter =
          symmetry::NchwOutputIterator<kN, DualNet::kNumStoneFeatures, float>;
      symmetry::ApplySymmetry<kN, DualNet::kNumStoneFeatures>(
          symmetries_used_[i], raw_features.data(),
          no_copy_ ? OutIter((float*)(no_copy_features_+i)) : OutIter(features_[i].data()));
    } else {
      symmetry::ApplySymmetry<kN, DualNet::kNumStoneFeatures>(
          symmetries_used_[i], raw_features.data(), no_copy_ ? (float*)(no_copy_features_+i) : features_[i].data());
    }
  }

  std::vector<const DualNet::BoardFeatures*> feature_ptrs;
  std::vector<DualNet::Output*> output_ptrs;

  if(!no_copy_){
      feature_ptrs.reserve(features_.size());
      for (const auto& feature : features_) {
          feature_ptrs.push_back(&feature);
      }
      outputs_.resize(paths.size());
      output_ptrs.reserve(outputs_.size());
      for (auto& output : outputs_) {
          output_ptrs.push_back(&output);
      }
  }

  // Run inference.
  network_->RunMany(std::move(feature_ptrs), std::move(output_ptrs),
                    &inference_model_);

  // Record some information about the inference.
  if (!inference_model_.empty()) {
    if (inferences_.empty() || inference_model_ != inferences_.back().model) {
      inferences_.emplace_back(inference_model_, root_->position.n());
    }
    inferences_.back().last_move = root_->position.n();
    inferences_.back().total_count += paths.size();
  }

  // Incorporate the inference outputs back into tree search, undoing any
  // previously applied random symmetries.
  std::array<float, kNumMoves> raw_policy;
  for (size_t i = 0; i < paths.size(); ++i) {
    auto* root = paths[i].root;
    auto* leaf = paths[i].leaf;
    if(no_copy_){
        const auto output = no_copy_outputs_[i];
        symmetry::ApplySymmetry<kN, 1>(symmetry::Inverse(symmetries_used_[i]),
                                       output.policy.data(), raw_policy.data());
        raw_policy[Coord::kPass] = output.policy[Coord::kPass];
        leaf->IncorporateResults(options_.value_init_penalty, raw_policy,
                                 output.value, root);
        leaf->RevertVirtualLoss(root);
    } else {
        const auto& output = outputs_[i];
        symmetry::ApplySymmetry<kN, 1>(symmetry::Inverse(symmetries_used_[i]),
                                       output.policy.data(), raw_policy.data());
        raw_policy[Coord::kPass] = output.policy[Coord::kPass];
        leaf->IncorporateResults(options_.value_init_penalty, raw_policy,
                                 output.value, root);
        leaf->RevertVirtualLoss(root);
    }
  }
}

}  // namespace minigo
