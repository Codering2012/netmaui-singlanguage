// ==============================================================================
// ASL FOUNDATION MODEL — REAL-TIME C++ ONNX RUNTIME INFERENCE ENGINE
// Speculative Draft-Target Decoder on Intel i5 CPU / iGPU (DirectML / OpenVINO)
//
// Architecture:
//   ┌──────────────────────────────────────────────────────────┐
//   │  Live Keypoints  →  RingBuffer (zero-copy circular)      │
//   │  RestStateDetector + FrameDiffGate (skip idle frames)    │
//   │  ASL Encoder ONNX → Memory (B, T, 320)                  │
//   │  Speculative Loop:                                        │
//   │    Draft decoder proposes γ=4 tokens autoregressively    │
//   │    Target decoder verifies all γ in ONE parallel pass    │
//   │    Accept/reject using speculative sampling theorem      │
//   └──────────────────────────────────────────────────────────┘
// ==============================================================================

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <memory>
#include <algorithm>
#include <string>
#include <random>
#include <cassert>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

// ── Constants ─────────────────────────────────────────────────────────────────
constexpr int MAX_LEN           = 120;  // Sliding window length (frames)
constexpr int NUM_KEYPOINTS     = 60;
constexpr int CHANNELS_PER_KP   = 9;
constexpr int FRAME_SIZE        = NUM_KEYPOINTS * CHANNELS_PER_KP; // 540 floats/frame
constexpr int ENCODER_DIM       = 512;  // ASLFoundationModel d_model
constexpr int GAMMA             = 4;    // Draft tokens per speculative step
constexpr int MAX_DECODE_TOKENS = 128;  // Max generated sequence length
constexpr int BOS_TOKEN_ID      = 1;
constexpr int EOS_TOKEN_ID      = 2;

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::mt19937 rng(42);

static inline float rand_uniform() {
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
}

static std::vector<float> softmax(const float* logits, int vocab_size) {
    float max_val = *std::max_element(logits, logits + vocab_size);
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}

static int sample_from_probs(const std::vector<float>& probs) {
    float r = rand_uniform();
    float cumul = 0.0f;
    for (int i = 0; i < (int)probs.size(); ++i) {
        cumul += probs[i];
        if (r <= cumul) return i;
    }
    return (int)probs.size() - 1;
}

static int argmax(const std::vector<float>& probs) {
    return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// ==============================================================================
// 1. ZERO-COPY CIRCULAR RING BUFFER
// ==============================================================================
class RingBuffer {
public:
    explicit RingBuffer(int max_len = MAX_LEN)
        : max_len_(max_len), ptr_(0),
          buf_(max_len * FRAME_SIZE, 0.0f) {}

    void push(const float* frame_data) {
        int idx = ptr_ % max_len_;
        std::copy(frame_data, frame_data + FRAME_SIZE, buf_.data() + idx * FRAME_SIZE);
        ++ptr_;
    }

    // Returns (frames_flat [max_len * FRAME_SIZE], mask [max_len], n_filled)
    void get_padded(std::vector<float>& out_frames, std::vector<float>& out_mask) const {
        int n_filled = std::min(ptr_, max_len_);
        out_frames.assign(max_len_ * FRAME_SIZE, 0.0f);
        out_mask.assign(max_len_, 0.0f);
        if (n_filled == 0) return;

        int start = max_len_ - n_filled;
        std::fill(out_mask.begin() + start, out_mask.end(), 1.0f);

        int first_src = (ptr_ - n_filled) % max_len_;
        if (first_src < 0) first_src += max_len_;

        int len1 = std::min(n_filled, max_len_ - first_src);
        std::copy(buf_.data() + first_src * FRAME_SIZE,
                  buf_.data() + (first_src + len1) * FRAME_SIZE,
                  out_frames.data() + start * FRAME_SIZE);

        int len2 = n_filled - len1;
        if (len2 > 0) {
            std::copy(buf_.data(),
                      buf_.data() + len2 * FRAME_SIZE,
                      out_frames.data() + (start + len1) * FRAME_SIZE);
        }
    }

    int n_filled() const { return std::min(ptr_, max_len_); }
    int ptr() const { return ptr_; }

private:
    int max_len_, ptr_;
    std::vector<float> buf_;
};

// ==============================================================================
// 2. REST STATE & FRAME DIFFERENCE GATING
// ==============================================================================
class RestStateDetector {
public:
    explicit RestStateDetector(float energy_threshold = 0.008f)
        : threshold_(energy_threshold) {}

    bool is_resting(const float* frame) const {
        float e = 0.0f;
        #pragma omp simd reduction(+:e)
        for (int kp = 0; kp < NUM_KEYPOINTS; ++kp) {
            float vx = frame[kp * 9 + 3];
            float vy = frame[kp * 9 + 4];
            float vz = frame[kp * 9 + 5];
            e += vx*vx + vy*vy + vz*vz;
        }
        return (e / (NUM_KEYPOINTS * 3)) < threshold_;
    }
private:
    float threshold_;
};

class FrameDiffGate {
public:
    explicit FrameDiffGate(float diff_threshold = 0.003f)
        : threshold_(diff_threshold), has_prev_(false), prev_(FRAME_SIZE, 0.0f) {}

    bool should_skip(const float* frame) {
        if (!has_prev_) {
            std::copy(frame, frame + FRAME_SIZE, prev_.begin());
            has_prev_ = true;
            return false;
        }
        float diff = 0.0f;
        #pragma omp simd reduction(+:diff)
        for (int i = 0; i < FRAME_SIZE; ++i)
            diff += std::abs(frame[i] - prev_[i]);
        std::copy(frame, frame + FRAME_SIZE, prev_.begin());
        return (diff / FRAME_SIZE) < threshold_;
    }
private:
    float threshold_;
    bool has_prev_;
    std::vector<float> prev_;
};

// ==============================================================================
// 1b. RIEMANNIAN SE(3) PROCRUSTES CANONICALIZER
// ==============================================================================
class RiemannianSE3Aligner {
public:
    RiemannianSE3Aligner() = default;

    // Aligns keypoint array (60 x 3) in 3D: translation centroid centering + scale normalization
    void align_in_place(float* pts_3d, int num_pts = NUM_KEYPOINTS) const {
        if (num_pts != NUM_KEYPOINTS) return;

        // 1. Centroid Translation
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        for (int i = 0; i < num_pts; ++i) {
            cx += pts_3d[i * 3 + 0];
            cy += pts_3d[i * 3 + 1];
            cz += pts_3d[i * 3 + 2];
        }
        cx /= num_pts; cy /= num_pts; cz /= num_pts;

        for (int i = 0; i < num_pts; ++i) {
            pts_3d[i * 3 + 0] -= cx;
            pts_3d[i * 3 + 1] -= cy;
            pts_3d[i * 3 + 2] -= cz;
        }

        // 2. Scale Normalization
        float sq_sum = 0.0f;
        for (int i = 0; i < num_pts * 3; ++i) {
            sq_sum += pts_3d[i] * pts_3d[i];
        }
        float scale = std::sqrt(sq_sum / num_pts);
        if (scale > 1e-6f) {
            for (int i = 0; i < num_pts * 3; ++i) {
                pts_3d[i] /= scale;
            }
        }
    }
};

// ==============================================================================
// 3. THIN ONNX SESSION WRAPPER WITH OPENVINO & DIRECTML CPU/GPU ACCELERATION
// ==============================================================================
class OrtSession {
public:
    OrtSession(Ort::Env& env, const ORTCHAR_T* model_path, int intra_threads = 4, bool enable_openvino = false, bool enable_dml = false) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        #ifdef USE_OPENVINO
        if (enable_openvino) {
            try {
                OrtOpenVINOProviderOptions ov_options;
                ov_options.device_type = "CPU_FP32";
                opts.AppendExecutionProvider_OpenVINO(ov_options);
                std::cout << "[+] OpenVINO Execution Provider enabled for Intel CPU/iGPU.\n";
            } catch (const std::exception& e) {
                std::cout << "[!] OpenVINO EP failed to load, falling back to CPU: " << e.what() << "\n";
            }
        }
        #endif

        #ifdef USE_DIRECTML
        if (enable_dml) {
            try {
                // DirectML GPU Acceleration for Intel UHD Graphics 620
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DirectML(opts, 0));
                std::cout << "[+] DirectML Execution Provider enabled (GPU 0).\n";
            } catch (const std::exception& e) {
                std::cout << "[!] DirectML EP failed to load, falling back to CPU: " << e.what() << "\n";
            }
        }
        #endif

        session_ = std::make_unique<Ort::Session>(env, model_path, opts);
        alloc_   = Ort::AllocatorWithDefaultOptions();

        // Cache I/O names
        size_t n_in = session_->GetInputCount();
        for (size_t i = 0; i < n_in; ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc_);
            in_name_bufs_.push_back(std::string(name.get()));
        }
        size_t n_out = session_->GetOutputCount();
        for (size_t i = 0; i < n_out; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc_);
            out_name_bufs_.push_back(std::string(name.get()));
        }
        for (auto& s : in_name_bufs_)  in_names_.push_back(s.c_str());
        for (auto& s : out_name_bufs_) out_names_.push_back(s.c_str());
    }

    // Run with arbitrary float input tensors
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs) {
        return session_->Run(Ort::RunOptions{nullptr},
                             in_names_.data(), inputs.data(), inputs.size(),
                             out_names_.data(), out_names_.size());
    }

    Ort::Session*            get()            { return session_.get(); }
    const std::vector<const char*>& in_names()  const { return in_names_; }
    const std::vector<const char*>& out_names() const { return out_names_; }

private:
    std::unique_ptr<Ort::Session>   session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::vector<std::string>        in_name_bufs_, out_name_bufs_;
    std::vector<const char*>        in_names_, out_names_;
};

// ==============================================================================
// 4. SPECULATIVE DRAFT-TARGET STREAMING DECODER
// ==============================================================================
class SpeculativeStreamingEngine {
public:
    SpeculativeStreamingEngine(
        Ort::Env&           env,
        const ORTCHAR_T*    encoder_path,
        const ORTCHAR_T*    target_decoder_path,
        const ORTCHAR_T*    draft_decoder_path,
        int                 vocab_size,
        int                 gamma             = GAMMA,
        int                 max_tokens        = MAX_DECODE_TOKENS,
        int                 intra_threads     = 4
    )
        : encoder_(env, encoder_path, intra_threads),
          target_dec_(env, target_decoder_path, intra_threads),
          draft_dec_(env, draft_decoder_path, intra_threads),
          vocab_size_(vocab_size),
          gamma_(gamma),
          max_tokens_(max_tokens),
          mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          ring_(),
          rest_detector_(),
          diff_gate_()
    {}

    // ── Push one live frame; return decoded sequence if segment is complete ──
    bool push_frame(const float* frame_data, std::vector<int>& out_tokens) {
        ring_.push(frame_data);

        if (rest_detector_.is_resting(frame_data)) return false;
        if (diff_gate_.should_skip(frame_data))    return false;
        if (ring_.n_filled() < 16)                 return false;  // need at least 16 frames

        // --- Encode -----------------------------------------------------------
        std::vector<float>  feat_flat, mask_flat;
        ring_.get_padded(feat_flat, mask_flat);

        std::vector<int64_t> feat_shape = {1, MAX_LEN, FRAME_SIZE};
        std::vector<int64_t> mask_shape = {1, MAX_LEN};

        std::vector<uint8_t> mask_bool(mask_flat.size());
        for (size_t i = 0; i < mask_flat.size(); ++i) mask_bool[i] = (mask_flat[i] > 0.5f) ? 1 : 0;

        Ort::Value feat_t = Ort::Value::CreateTensor<float>(
            mem_info_, feat_flat.data(), feat_flat.size(), feat_shape.data(), feat_shape.size());
        Ort::Value mask_t = Ort::Value::CreateTensor<bool>(
            mem_info_, reinterpret_cast<bool*>(mask_bool.data()), mask_bool.size(), mask_shape.data(), mask_shape.size());

        std::vector<Ort::Value> enc_inputs;
        enc_inputs.push_back(std::move(feat_t));
        enc_inputs.push_back(std::move(mask_t));

        auto enc_out = encoder_.run(enc_inputs);
        float* mem_data = enc_out[0].GetTensorMutableData<float>();
        auto   mem_info_shape = enc_out[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t mem_seq = mem_info_shape[1]; // T
        std::vector<float> memory(mem_data, mem_data + mem_seq * ENCODER_DIM);

        // --- Speculative Decoding Loop ----------------------------------------
        std::vector<int64_t> prefix = {BOS_TOKEN_ID};
        std::vector<int64_t> mem_shape = {1, mem_seq, ENCODER_DIM};

        while ((int)prefix.size() <= max_tokens_) {
            // 1. Draft phase: autoregressively propose γ tokens
            std::vector<int64_t> draft_tokens;
            std::vector<std::vector<float>> draft_probs_seq; // per step

            std::vector<int64_t> cur_ids = prefix;
            for (int g = 0; g < gamma_; ++g) {
                std::vector<int64_t> tgt_shape = {1, (int64_t)cur_ids.size()};

                Ort::Value tgt_t = Ort::Value::CreateTensor<int64_t>(
                    mem_info_, cur_ids.data(), cur_ids.size(),
                    tgt_shape.data(), tgt_shape.size());
                Ort::Value mem_t = Ort::Value::CreateTensor<float>(
                    mem_info_, memory.data(), memory.size(),
                    mem_shape.data(), mem_shape.size());

                std::vector<Ort::Value> d_in;
                d_in.push_back(std::move(tgt_t));
                d_in.push_back(std::move(mem_t));

                auto d_out   = draft_dec_.run(d_in);
                float* logits = d_out[0].GetTensorMutableData<float>();
                int last_pos  = (int)cur_ids.size() - 1;
                auto dp = softmax(logits + last_pos * vocab_size_, vocab_size_);
                int tok = sample_from_probs(dp);

                draft_tokens.push_back(tok);
                draft_probs_seq.push_back(std::move(dp));
                cur_ids.push_back(tok);

                if (tok == EOS_TOKEN_ID) break;
            }

            int n_proposed = (int)draft_tokens.size();

            // 2. Target phase: verify all γ tokens in ONE parallel forward pass
            //    Input: prefix + all draft tokens (length = |prefix| + γ)
            //    We read logits at positions [|prefix|-1 .. |prefix|+γ-1]
            std::vector<int64_t> verify_ids = prefix;
            for (auto t : draft_tokens) verify_ids.push_back(t);

            std::vector<int64_t> verify_shape = {1, (int64_t)verify_ids.size()};
            Ort::Value vtgt_t = Ort::Value::CreateTensor<int64_t>(
                mem_info_, verify_ids.data(), verify_ids.size(),
                verify_shape.data(), verify_shape.size());
            Ort::Value vmem_t = Ort::Value::CreateTensor<float>(
                mem_info_, memory.data(), memory.size(),
                mem_shape.data(), mem_shape.size());

            std::vector<Ort::Value> v_in;
            v_in.push_back(std::move(vtgt_t));
            v_in.push_back(std::move(vmem_t));

            auto v_out      = target_dec_.run(v_in);
            float* t_logits = v_out[0].GetTensorMutableData<float>();

            // 3. Acceptance / Rejection loop (speculative sampling theorem)
            int accepted = 0;
            int prefix_len = (int)prefix.size();

            for (int g = 0; g < n_proposed; ++g) {
                int pos = prefix_len - 1 + g; // position in verify_ids output
                auto tp = softmax(t_logits + pos * vocab_size_, vocab_size_);
                float accept_ratio = std::min(1.0f,
                    tp[draft_tokens[g]] / (draft_probs_seq[g][draft_tokens[g]] + 1e-9f));

                if (rand_uniform() < accept_ratio) {
                    prefix.push_back(draft_tokens[g]);
                    ++accepted;
                    if (draft_tokens[g] == EOS_TOKEN_ID) {
                        out_tokens = std::vector<int>(prefix.begin() + 1, prefix.end() - 1);
                        return true;
                    }
                } else {
                    // Rejection: sample from adjusted distribution q'(x) = norm(max(0, p-q))
                    std::vector<float> adj(vocab_size_);
                    float adj_sum = 0.0f;
                    for (int v = 0; v < vocab_size_; ++v) {
                        int pos2 = prefix_len - 1 + g;
                        auto tp2 = softmax(t_logits + pos2 * vocab_size_, vocab_size_);
                        adj[v] = std::max(0.0f, tp2[v] - draft_probs_seq[g][v]);
                        adj_sum += adj[v];
                    }
                    if (adj_sum > 1e-9f)
                        for (auto& a : adj) a /= adj_sum;
                    else
                        std::fill(adj.begin(), adj.end(), 1.0f / vocab_size_);

                    int corrected = sample_from_probs(adj);
                    prefix.push_back(corrected);
                    if (corrected == EOS_TOKEN_ID) {
                        out_tokens = std::vector<int>(prefix.begin() + 1, prefix.end() - 1);
                        return true;
                    }
                    break; // stop this round; new round starts with corrected prefix
                }
            }

            // 4. If all γ accepted: also sample one bonus token from target
            if (accepted == n_proposed) {
                int bonus_pos = prefix_len - 1 + n_proposed;
                auto tp = softmax(t_logits + bonus_pos * vocab_size_, vocab_size_);
                int bonus = sample_from_probs(tp);
                prefix.push_back(bonus);
                if (bonus == EOS_TOKEN_ID) {
                    out_tokens = std::vector<int>(prefix.begin() + 1, prefix.end() - 1);
                    return true;
                }
            }

            if ((int)prefix.size() >= max_tokens_) break;
        }

        out_tokens = std::vector<int>(prefix.begin() + 1, prefix.end());
        return !out_tokens.empty();
    }

private:
    OrtSession          encoder_;
    OrtSession          target_dec_;
    OrtSession          draft_dec_;
    int                 vocab_size_;
    int                 gamma_;
    int                 max_tokens_;
    Ort::MemoryInfo     mem_info_;
    RingBuffer          ring_;
    RestStateDetector   rest_detector_;
    FrameDiffGate       diff_gate_;
};

// ==============================================================================
// 4b. SINGLE-PASS NON-AUTOREGRESSIVE CIF STREAMING ENGINE (O(1) Latency)
// ==============================================================================
class CIFStreamingEngine {
public:
    CIFStreamingEngine(
        Ort::Env&           env,
        const ORTCHAR_T*    cif_model_path,
        int                 vocab_size,
        int                 max_cif_tokens = 196,
        int                 intra_threads  = 4
    )
        : cif_session_(env, cif_model_path, intra_threads),
          vocab_size_(vocab_size),
          max_cif_tokens_(max_cif_tokens),
          mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          ring_(),
          rest_detector_(),
          diff_gate_()
    {}

    // Push live frame; returns true when non-AR CIF predicts a complete segment
    bool push_frame(const float* frame_data, std::vector<int>& out_tokens, float& out_qty) {
        ring_.push(frame_data);

        if (rest_detector_.is_resting(frame_data)) return false;
        if (diff_gate_.should_skip(frame_data))    return false;
        if (ring_.n_filled() < 16)                 return false;

        std::vector<float> feat_flat, mask_flat;
        ring_.get_padded(feat_flat, mask_flat);

        std::vector<int64_t> feat_shape = {1, MAX_LEN, FRAME_SIZE};
        std::vector<int64_t> mask_shape = {1, MAX_LEN};

        std::vector<uint8_t> mask_bool(mask_flat.size());
        for (size_t i = 0; i < mask_flat.size(); ++i) mask_bool[i] = (mask_flat[i] > 0.5f) ? 1 : 0;

        Ort::Value feat_t = Ort::Value::CreateTensor<float>(
            mem_info_, feat_flat.data(), feat_flat.size(), feat_shape.data(), feat_shape.size());
        Ort::Value mask_t = Ort::Value::CreateTensor<bool>(
            mem_info_, reinterpret_cast<bool*>(mask_bool.data()), mask_bool.size(), mask_shape.data(), mask_shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(feat_t));
        inputs.push_back(std::move(mask_t));

        // Single pass execution: O(1) latency
        auto outputs = cif_session_.run(inputs);

        // cif_logits: [1, max_cif_tokens, vocab_size]
        // cif_qty_sum: [1]
        float* logits_data = outputs[0].GetTensorMutableData<float>();
        float* qty_data    = outputs[1].GetTensorMutableData<float>();

        out_qty = qty_data[0];
        int num_tokens = std::min(max_cif_tokens_, std::max(1, (int)std::round(out_qty)));

        out_tokens.clear();
        for (int i = 0; i < num_tokens; ++i) {
            const float* tok_logits = logits_data + i * vocab_size_;
            auto probs = softmax(tok_logits, vocab_size_);
            int best_id = argmax(probs);
            if (best_id == EOS_TOKEN_ID) break;
            if (best_id != BOS_TOKEN_ID && best_id != 0) { // skip PAD (0) and BOS (1)
                out_tokens.push_back(best_id);
            }
        }

        return !out_tokens.empty();
    }

private:
    OrtSession          cif_session_;
    int                 vocab_size_;
    int                 max_cif_tokens_;
    Ort::MemoryInfo     mem_info_;
    RingBuffer          ring_;
    RestStateDetector   rest_detector_;
    FrameDiffGate       diff_gate_;
};

// ==============================================================================
// 4c. HYBRID CIF-ENTROPY STREAMING ENGINE
// ==============================================================================
class HybridStreamingEngine {
public:
    HybridStreamingEngine(
        Ort::Env&           env,
        const ORTCHAR_T*    encoder_path,
        const ORTCHAR_T*    decoder_path,
        int                 vocab_size,
        int                 max_tokens     = MAX_DECODE_TOKENS,
        int                 intra_threads  = 4
    )
        : encoder_(env, encoder_path, intra_threads),
          decoder_(env, decoder_path, intra_threads),
          vocab_size_(vocab_size),
          max_tokens_(max_tokens),
          mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          ring_(),
          rest_detector_(),
          diff_gate_(),
          committed_prefix_({BOS_TOKEN_ID})
    {}

    void reset() {
        committed_prefix_ = {BOS_TOKEN_ID};
    }

    // Push live frame; returns true if segment is fully completed (EOS committed)
    bool push_frame(const float* frame_data, std::vector<int>& out_committed, std::vector<int>& out_draft) {
        ring_.push(frame_data);

        if (rest_detector_.is_resting(frame_data)) return false;
        if (diff_gate_.should_skip(frame_data))    return false;
        if (ring_.n_filled() < 16)                 return false;

        std::vector<float> feat_flat, mask_flat;
        ring_.get_padded(feat_flat, mask_flat);

        std::vector<int64_t> feat_shape = {1, MAX_LEN, FRAME_SIZE};
        std::vector<int64_t> mask_shape = {1, MAX_LEN};

        std::vector<uint8_t> mask_bool(mask_flat.size());
        for (size_t i = 0; i < mask_flat.size(); ++i) mask_bool[i] = (mask_flat[i] > 0.5f) ? 1 : 0;

        Ort::Value feat_t = Ort::Value::CreateTensor<float>(
            mem_info_, feat_flat.data(), feat_flat.size(), feat_shape.data(), feat_shape.size());
        Ort::Value mask_t = Ort::Value::CreateTensor<bool>(
            mem_info_, reinterpret_cast<bool*>(mask_bool.data()), mask_bool.size(), mask_shape.data(), mask_shape.size());

        std::vector<Ort::Value> enc_inputs;
        enc_inputs.push_back(std::move(feat_t));
        enc_inputs.push_back(std::move(mask_t));

        auto enc_out = encoder_.run(enc_inputs);
        float* mem_data = enc_out[0].GetTensorMutableData<float>();
        auto   mem_info_shape = enc_out[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t mem_seq = mem_info_shape[1]; // T
        std::vector<float> memory(mem_data, mem_data + mem_seq * ENCODER_DIM);

        // 1. Monotonic Boundary Theorem (CIF)
        float cif_qty_sum = 0.0f;
        if (enc_out.size() > 1) {
            cif_qty_sum = enc_out[1].GetTensorMutableData<float>()[0];
        }
        int cif_target = (int)std::floor(cif_qty_sum);

        std::vector<int64_t> mem_shape = {1, mem_seq, ENCODER_DIM};

        // 3. Prefix-Constrained Decoding
        std::vector<int64_t> cur_ids = committed_prefix_;
        std::vector<int64_t> draft_ids;

        std::vector<int64_t> mem_mask_shape = {1, mem_seq};
        std::vector<uint8_t> mem_mask_bool(mem_seq, 1);

        while ((int)cur_ids.size() <= max_tokens_) {
            std::vector<int64_t> last_id = {cur_ids.back()};
            std::vector<int64_t> tgt_shape = {1, 1};

            Ort::Value tgt_t = Ort::Value::CreateTensor<int64_t>(
                mem_info_, last_id.data(), 1,
                tgt_shape.data(), tgt_shape.size());
            Ort::Value mem_t = Ort::Value::CreateTensor<float>(
                mem_info_, memory.data(), memory.size(),
                mem_shape.data(), mem_shape.size());
            Ort::Value mem_mask_t = Ort::Value::CreateTensor<bool>(
                mem_info_, reinterpret_cast<bool*>(mem_mask_bool.data()), mem_mask_bool.size(),
                mem_mask_shape.data(), mem_mask_shape.size());

            std::vector<Ort::Value> d_in;
            d_in.push_back(std::move(tgt_t));
            d_in.push_back(std::move(mem_t));
            d_in.push_back(std::move(mem_mask_t));

            auto d_out   = decoder_.run(d_in);
            float* logits = d_out[0].GetTensorMutableData<float>();

            auto probs = softmax(logits, vocab_size_);
            int best_id = argmax(probs);

            // 2. Entropy Decay (Information Theory)
            float entropy = 0.0f;
            for (float p : probs) {
                if (p > 1e-9f) {
                    entropy -= p * std::log(p);
                }
            }

            // Commit Logic
            bool should_commit = false;
            int committed_count = (int)committed_prefix_.size() - 1; // subtract BOS

            if (committed_count < cif_target) {
                should_commit = true;
            } else if (entropy < 0.2f) {
                should_commit = true;
            }

            if (should_commit) {
                committed_prefix_.push_back(best_id);
                cur_ids.push_back(best_id);
                if (best_id == EOS_TOKEN_ID) {
                    out_committed = std::vector<int>(committed_prefix_.begin() + 1, committed_prefix_.end() - 1);
                    out_draft.clear();
                    return true;
                }
            } else {
                draft_ids.push_back(best_id);
                cur_ids.push_back(best_id);
                if (best_id == EOS_TOKEN_ID) {
                    break; // Drafted EOS, stop drafting but don't commit it yet
                }
            }

            // In live streaming, we draft ahead a small fixed window (e.g. 3 tokens)
            if (draft_ids.size() >= 3) break;
        }

        out_committed = std::vector<int>(committed_prefix_.begin() + 1, committed_prefix_.end());
        out_draft     = std::vector<int>(draft_ids.begin(), draft_ids.end());
        return false;
    }

private:
    OrtSession          encoder_;
    OrtSession          decoder_;
    int                 vocab_size_;
    int                 max_tokens_;
    Ort::MemoryInfo     mem_info_;
    RingBuffer          ring_;
    RestStateDetector   rest_detector_;
    FrameDiffGate       diff_gate_;
    std::vector<int64_t> committed_prefix_;
};

// ==============================================================================
// 5. BENCHMARK / DEMO  (offline ring-buffer gating test — no ONNX required)
// ==============================================================================
static void run_gating_benchmark() {
    RingBuffer       ring;
    RestStateDetector rest;
    FrameDiffGate    gate;

    std::vector<float> frame(FRAME_SIZE, 0.05f);

    auto t0 = std::chrono::high_resolution_clock::now();
    int processed = 0, skipped = 0;

    for (int f = 0; f < 300; ++f) {
        // Simulate hand motion in ~40% of frames
        if (f % 5 == 0)
            for (int i = 0; i < FRAME_SIZE; ++i)
                frame[i] += (f % 3 == 0) ? 0.02f : -0.01f;

        ring.push(frame.data());
        if (rest.is_resting(frame.data()) || gate.should_skip(frame.data())) { ++skipped; continue; }
        ++processed;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = t1 - t0;

    std::cout << "[+] Gating Benchmark — 300 simulated frames @ 30 FPS:" << std::endl;
    std::cout << "  Processed  : " << processed << std::endl;
    std::cout << "  Skipped    : " << skipped
              << " (" << skipped * 100 / 300 << "% skip rate)" << std::endl;
    std::cout << "  Elapsed    : " << dur.count() << " ms" << std::endl;
    std::cout << "  Pipeline   : " << (300.0 / (dur.count() / 1000.0)) << " FPS" << std::endl;
}

inline std::basic_string<ORTCHAR_T> to_ort_path(const char* s) {
#ifdef _WIN32
    return std::wstring(s, s + strlen(s));
#else
    return std::string(s);
#endif
}

// ==============================================================================
// 6. ENTRY POINT
// ==============================================================================
int main(int argc, char* argv[]) {
    std::cout << "======================================================================"  << std::endl;
    std::cout << "   ASL FOUNDATION MODEL — C++ SPECULATIVE & CIF STREAMING ENGINE"       << std::endl;
    std::cout << "======================================================================"  << std::endl;

    // ── Gating benchmark (always runs; no ONNX files required) ───────────────
    run_gating_benchmark();

    // ── CIF Single-Pass Non-AR Mode: if 2 args provided (cif_encoder.onnx, vocab_size)
    if (argc == 3) {
        try {
            auto cif_path = to_ort_path(argv[1]);
            int vocab_size = std::stoi(argv[2]);

            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ASLCIFEngine");
            std::cout << "\n[*] Loading Single-Pass Non-AR CIF ONNX Model..." << std::endl;
            CIFStreamingEngine cif_engine(env, cif_path.c_str(), vocab_size);
            std::cout << "[+] CIF Session loaded successfully." << std::endl;

            std::vector<float> frame(FRAME_SIZE, 0.05f);
            std::vector<int>   decoded;
            float              qty = 0.0f;

            std::cout << "[*] Feeding 50 simulated motion frames into CIF Engine..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int f = 0; f < 50; ++f) {
                for (int i = 0; i < FRAME_SIZE; ++i)
                    frame[i] = 0.03f + 0.01f * (f % 7) + 0.002f * i;
                bool done = cif_engine.push_frame(frame.data(), decoded, qty);
                if (done) break;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = t1 - t0;

            std::cout << "[+] CIF Single-Pass Decoding Complete!" << std::endl;
            std::cout << "  Predicted Qty : " << qty << std::endl;
            std::cout << "  Decoded Tokens: " << decoded.size() << std::endl;
            std::cout << "  Latency       : " << dur.count() << " ms (O(1) non-AR pass)" << std::endl;
            std::cout << "  Token IDs     : [ ";
            for (int t : decoded) std::cout << t << " ";
            std::cout << "]" << std::endl;

        } catch (const std::exception& ex) {
            std::cerr << "[!] CIF Error: " << ex.what() << std::endl;
            return 1;
        }
    }
    // ── Speculative Draft-Target Mode: if 4 args provided
    else if (argc >= 5) {
        try {
            auto enc_path   = to_ort_path(argv[1]);
            auto tgt_path   = to_ort_path(argv[2]);
            auto draft_path = to_ort_path(argv[3]);
            int vocab_size  = std::stoi(argv[4]);

            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ASLSpeculativeEngine");

            std::cout << "\n[*] Loading Speculative ONNX sessions..." << std::endl;
            SpeculativeStreamingEngine engine(
                env,
                enc_path.c_str(),
                tgt_path.c_str(),
                draft_path.c_str(),
                vocab_size,
                /*gamma=*/GAMMA,
                /*max_tokens=*/MAX_DECODE_TOKENS,
                /*intra_threads=*/4
            );
            std::cout << "[+] All sessions loaded." << std::endl;

            // Simulate one signing segment (50 motion frames)
            std::vector<float> frame(FRAME_SIZE, 0.05f);
            std::vector<int>   decoded;

            std::cout << "[*] Feeding 50 simulated motion frames..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int f = 0; f < 50; ++f) {
                for (int i = 0; i < FRAME_SIZE; ++i)
                    frame[i] = 0.03f + 0.01f * (f % 7) + 0.002f * i;
                bool done = engine.push_frame(frame.data(), decoded);
                if (done) break;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = t1 - t0;

            std::cout << "[+] Speculative Decoding Complete!" << std::endl;
            std::cout << "  Decoded tokens: " << decoded.size() << std::endl;
            std::cout << "  Latency       : " << dur.count() << " ms" << std::endl;
            std::cout << "  Token IDs     : [ ";
            for (int t : decoded) std::cout << t << " ";
            std::cout << "]" << std::endl;

        } catch (const std::exception& ex) {
            std::cerr << "[!] Error: " << ex.what() << std::endl;
            return 1;
        }
    }
    // ── Hybrid CIF-Entropy Mode: if 4 args provided
    else if (argc == 4) {
        try {
            auto enc_path  = to_ort_path(argv[1]);
            auto dec_path  = to_ort_path(argv[2]);
            int vocab_size = std::stoi(argv[3]);

            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ASLHybridEngine");

            std::cout << "\n[*] Loading Hybrid CIF-Entropy ONNX sessions..." << std::endl;
            HybridStreamingEngine engine(
                env, enc_path.c_str(), dec_path.c_str(), vocab_size
            );
            std::cout << "[+] All sessions loaded." << std::endl;

            std::vector<float> frame(FRAME_SIZE, 0.05f);
            std::vector<int>   committed, draft;

            std::cout << "[*] Feeding 50 simulated motion frames into Hybrid Engine..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int f = 0; f < 50; ++f) {
                for (int i = 0; i < FRAME_SIZE; ++i)
                    frame[i] = 0.03f + 0.01f * (f % 7) + 0.002f * i;
                bool done = engine.push_frame(frame.data(), committed, draft);
                
                if (committed.size() > 0 || draft.size() > 0) {
                    std::cout << "  Frame " << f << " | Committed: [";
                    for (int c : committed) std::cout << c << " ";
                    std::cout << "] | Draft: [";
                    for (int d : draft) std::cout << d << " ";
                    std::cout << "]" << std::endl;
                }

                if (done) break;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = t1 - t0;

            std::cout << "[+] Hybrid Decoding Complete!" << std::endl;
            std::cout << "  Latency       : " << dur.count() << " ms" << std::endl;

        } catch (const std::exception& ex) {
            std::cerr << "[!] Error: " << ex.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "\n[*] ONNX demo mode usage:" << std::endl;
        std::cout << "    Single-Pass CIF Mode  : inference_engine <asl_cif_encoder.onnx> <vocab_size>" << std::endl;
        std::cout << "    Hybrid CIF-Ent Mode   : inference_engine <encoder.onnx> <decoder.onnx> <vocab_size>" << std::endl;
        std::cout << "    Speculative Mode      : inference_engine <encoder.onnx> <decoder.onnx> <draft.onnx> <vocab_size>" << std::endl;
    }

    std::cout << "\n[+] Done." << std::endl;
    return 0;
}

