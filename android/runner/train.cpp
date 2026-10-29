#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::aten::ScalarType;

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
    throw std::runtime_error("Missing or incorrectly sized input: " + path);
  f.seekg(0);
  std::vector<T> data(count);
  if (!f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T)))
    throw std::runtime_error("Read failed: " + path);
  return data;
}

// Checkpoint: JSON header followed by FP32 B.
void save_checkpoint(const std::string& path, const json& header,
                     const std::vector<float>& state) {
  std::string temp = path + ".tmp";
  std::ofstream f(temp, std::ios::binary | std::ios::trunc);
  f << header.dump() << '\n';
  f.write(reinterpret_cast<const char*>(state.data()), state.size()*sizeof(float));
  f.flush();
  if (!f) throw std::runtime_error("Checkpoint write failed");
  f.close();
  if (std::rename(temp.c_str(), path.c_str()))
    throw std::runtime_error("Checkpoint rename failed");
}

int main(int argc, char** argv) try {
  std::string bundle, checkpoint_in, checkpoint_out;
  int steps = 5, warmup = 0;
  for (int i=1; i<argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--generated-noise") continue;  // Legacy flag.
    if (i+1 >= argc) throw std::runtime_error("Missing value for " + arg);
    std::string value = argv[++i];
    if (arg == "--bundle") bundle = value;
    else if (arg == "--steps") steps = std::stoi(value);
    else if (arg == "--warmup") warmup = std::stoi(value);
    else if (arg == "--checkpoint-in") checkpoint_in = value;
    else if (arg == "--checkpoint-out") checkpoint_out = value;
    else throw std::runtime_error("Unknown option: " + arg);
  }
  if (bundle.empty() || steps <= 0 || warmup < 0)
    throw std::runtime_error("Usage: mobizo_train --bundle DIR --steps N [--warmup N] [--checkpoint-in FILE] [--checkpoint-out FILE]");
  std::ifstream manifest_file(bundle + "/manifest.json");
  json m; manifest_file >> m;
  const int batch = m.at("effective_batch"), seq = m.at("seq_len");
  const int n = m.at("adapter_numel");
  const int vocab = m.at("params").at("vocab_size");
  float lr = m.at("learning_rate"), eps = m.at("eps");
  if (batch <= 0 || seq < 2 || n <= 0 || !std::isfinite(lr) ||
      !std::isfinite(eps) || eps <= 0)
    throw std::runtime_error("Invalid manifest dimensions or optimizer parameters");
  auto tokens = read_raw<int32_t>(bundle+"/tokens.bin", batch*seq);
  auto labels = read_raw<int32_t>(bundle+"/labels.bin", batch*seq);
  int valid = 0;
  for (size_t i=0; i<tokens.size(); ++i) {
    if (tokens[i] < 0 || tokens[i] >= vocab ||
        (labels[i] != -100 && (labels[i] < 0 || labels[i] >= vocab)))
      throw std::runtime_error("Token or label outside vocabulary");
    if (i % seq && labels[i] != -100) ++valid;
  }
  if (!valid) throw std::runtime_error("No unmasked next-token targets");
  auto state = read_raw<float>(bundle+"/initial_state.bin", n);
  int next_step = 0;
  unsigned seed = m.at("seed");
  if (!checkpoint_in.empty()) {
    std::ifstream f(checkpoint_in, std::ios::binary);
    std::string line; std::getline(f, line);
    auto h = json::parse(line);
    if (h.at("format") != "mobizo-state-v1" || h.at("pte_sha256") != m.at("pte_sha256") ||
        h.at("adapter_numel") != n || h.at("fixture") != false ||
        h.at("learning_rate") != lr || h.at("eps") != eps || h.at("seed") != seed)
      throw std::runtime_error("Checkpoint does not match graph or training configuration");
    next_step = h.at("next_step");
    if (next_step < 0 || !f.read(reinterpret_cast<char*>(state.data()), n*sizeof(float)) ||
        f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Invalid checkpoint payload");
  }
  for (float value: state)
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite input state");
  Module module(bundle+"/training.pte", Module::LoadMode::Mmap);
  auto token_tensor = make_tensor_ptr({batch,seq}, tokens.data(), ScalarType::Int);
  auto label_tensor = make_tensor_ptr({batch,seq}, labels.data(), ScalarType::Int);
  auto lr_tensor = make_tensor_ptr({1}, &lr);
  auto eps_tensor = make_tensor_ptr({1}, &eps);
  double total_ms = 0;
  for (int iteration=0; iteration<steps+warmup; ++iteration, ++next_step) {
    auto start = std::chrono::steady_clock::now();
    std::vector<float> noise(n);
    // Per-step seeds make checkpoint restart independent of prior RNG draws.
    std::mt19937 rng(seed + static_cast<unsigned>(next_step));
    std::normal_distribution<float> normal(0.f, 1.f);
    for (auto& value: noise) value = normal(rng);
    auto state_tensor = make_tensor_ptr({n}, state.data());
    auto noise_tensor = make_tensor_ptr({n}, noise.data());
    std::vector<EValue> inputs{*token_tensor, *label_tensor, *state_tensor,
                              *noise_tensor, *lr_tensor, *eps_tensor};
    auto execute_start = std::chrono::steady_clock::now();
    auto result = module.forward(inputs);
    auto execute_end = std::chrono::steady_clock::now();
    if (!result.ok())
      throw std::runtime_error("Training execution failed at step " + std::to_string(next_step) +
                               ", error=" + std::to_string(static_cast<int>(result.error())));
    if (result->size() != 3) throw std::runtime_error("Unexpected output count");
    auto losses = result->at(0).toTensor();
    auto grad = result->at(1).toTensor();
    auto updated = result->at(2).toTensor();
    if (losses.numel() != 2 || grad.numel() != 1 || updated.numel() != n ||
        losses.scalar_type() != ScalarType::Float || grad.scalar_type() != ScalarType::Float ||
        updated.scalar_type() != ScalarType::Float)
      throw std::runtime_error("Unexpected output shape/type");
    float plus=losses.const_data_ptr<float>()[0], minus=losses.const_data_ptr<float>()[1];
    float g=grad.const_data_ptr<float>()[0];
    if (!std::isfinite(plus) || !std::isfinite(minus) || !std::isfinite(g))
      throw std::runtime_error("Nonfinite loss/gradient");
    double norm=0, delta=0;
    for (int j=0; j<n; ++j) {
      float value=updated.const_data_ptr<float>()[j];
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite updated state");
      norm += static_cast<double>(value)*value;
      delta += static_cast<double>(value-state[j])*(value-state[j]);
    }
    // Copy before the next execute: runtime output buffers may be reused.
    std::copy(updated.const_data_ptr<float>(), updated.const_data_ptr<float>()+n, state.begin());
    auto end = std::chrono::steady_clock::now();
    double execute_ms=std::chrono::duration<double,std::milli>(execute_end-execute_start).count();
    double step_ms=std::chrono::duration<double,std::milli>(end-start).count();
    if (iteration >= warmup) total_ms += step_ms;
    std::cout << json({{"step",next_step},{"loss_plus",plus},{"loss_minus",minus},
                       {"projected_grad",g},{"state_norm",std::sqrt(norm)},
                       {"state_delta_norm",std::sqrt(delta)},
                       {"execute_ms",execute_ms},{"step_ms",step_ms},
                       {"warmup",iteration<warmup}}).dump() << std::endl;
  }
  if (!checkpoint_out.empty()) {
    save_checkpoint(checkpoint_out, {{"format","mobizo-state-v1"},
      {"pte_sha256",m.at("pte_sha256")},{"adapter_numel",n},{"next_step",next_step},
      {"seed",seed},{"fixture",false},{"learning_rate",lr},{"eps",eps}}, state);
  }
  std::cout << json({{"status","ok"},{"completed_steps",steps},
                     {"warmup_steps",warmup},{"next_step",next_step},
                     {"average_step_ms",total_ms/steps}}).dump() << std::endl;
  return 0;
} catch (const std::exception& e) {
  std::cerr << "MobiZO: " << e.what() << std::endl;
  return 1;
}
