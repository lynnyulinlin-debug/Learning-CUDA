
#include "../tester/utils.h"


// ============================================================================
// 兼容符号：某些测试/SDK 可能会引用 cudaGetDeviceProperties_v2
// 注意：在非 NVIDIA 平台可能会与平台库的实现冲突，所以只在 NVIDIA 下提供。
// ============================================================================
// #if defined(PLATFORM_NVIDIA)
// extern "C" cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop, int device) {
//   return cudaGetDeviceProperties(prop, device);
// }
// #endif

// ============================================================================
// runtime mapping (CUDA / Iluvatar)
// 将所有 runtime 调用统一走这层，便于跨平台替换与定位问题。
// ============================================================================
// #if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
//   #define DEV_MALLOC        cudaMalloc
//   #define DEV_FREE          cudaFree
//   #define DEV_MEMCPY        cudaMemcpy
//   #define DEV_MEMSET        cudaMemset
//   #define DEV_DEVICE_SYNC   cudaDeviceSynchronize
//   #define DEV_GET_LAST_ERR  cudaGetLastError
//   #define DEV_ERR_STR       cudaGetErrorString

//   #define MEMCPY_H2D        cudaMemcpyHostToDevice
//   #define MEMCPY_D2H        cudaMemcpyDeviceToHost
//   #define MEMCPY_D2D        cudaMemcpyDeviceToDevice
// #else
//   #error "This kernels.cu is only for NVIDIA/ILUVATAR in this project layout."
// #endif

// // kernel launch check：先抓 launch error，再 sync（便于区分 launch vs runtime）
// #define KERNEL_LAUNCH_CHECK()                                         \
//   do {                                                                \
//     auto e = DEV_GET_LAST_ERR();                                      \
//     if (e != cudaSuccess) {                                           \
//       std::cerr << "Kernel launch error: " << DEV_ERR_STR(e) << "\n"; \
//       exit(EXIT_FAILURE);                                             \
//     }                                                                 \
//     RUNTIME_CHECK(DEV_DEVICE_SYNC());                                 \
//   } while (0)


// ============================================================================
// device helpers
// 注意：flashAttention 为了数值一致性，统一先转 float 再算，再转回。
// ============================================================================
// template <typename T>
// __device__ __forceinline__ float to_float_dev(T x) {
//   if constexpr (std::is_same_v<T, half>) return __half2float(x);
//   else return (float)x;
// }

// template <typename T>
// __device__ __forceinline__ T from_float_dev(float x) {
//   if constexpr (std::is_same_v<T, half>) return __float2half(x);
//   else return (T)x;
// }

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

// ============================================================================
// 1) trace
// 思路：grid-stride 遍历对角线 → block 内共享内存归约 → atomicAdd 到 out
//
// 注意：本项目只显式实例化 trace<int>, trace<float>，atomicAdd 对应可用。
// ============================================================================
template <typename T>
__device__ __forceinline__ T from_float_dev(float x) {
  if constexpr (std::is_same_v<T, half>) return __float2half(x);
  else return (T)x;
}

 template <typename T>
__global__ void trace_kernel(const T* __restrict__ in, size_t rows, size_t cols, T* __restrict__ out) {
  const size_t n = (rows < cols) ? rows : cols;
  // grid-stride over diagonal
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  T local = (T)0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += stride) {
    local = local + in[i * cols + i];
  }

  // block reduce in shared memory
  // 1) Each thread writes its partial sum to shared memory
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  smem[threadIdx.x] = local;
  __syncthreads();

  // 2) Tree reduction to get one sum per block (simple and portable), s=stride
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + s];
    __syncthreads();
  }

  // 3) One atomicAdd per block to accumulate into the final scalar
  if (threadIdx.x == 0) {
    atomicAdd(out, smem[0]);
  }
}


template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  const size_t n_elem = rows * cols;
  if (n_elem == 0) return (T)0;

  // device malloc
  T *d_in = nullptr, *d_out = nullptr;
  RUNTIME_CHECK(DEV_MALLOC(&d_in, n_elem * sizeof(T)));
  RUNTIME_CHECK(DEV_MALLOC(&d_out, sizeof(T)));

  RUNTIME_CHECK(DEV_MEMCPY(d_in, h_input.data(), n_elem * sizeof(T), MEMCPY_H2D));
  RUNTIME_CHECK(DEV_MEMSET(d_out, 0, sizeof(T)));// clear device scalar; 

  // blocks adaptive
  const int threads = 256; // a common, portable block size 
  const size_t n_diag = (rows < cols) ? rows : cols; //for small diagonals, we don't over-launch.
  int blocks = (int)((n_diag + (size_t)threads - 1) / (size_t)threads);
  blocks = std::max(std::min(blocks, 120), 1);

  // kernel launch ：<<<grid, block, sharedMemBytes>>>
  const size_t shmem = (size_t)threads * sizeof(T);
  trace_kernel<T><<<blocks, threads, shmem>>>(d_in, rows, cols, d_out);
  KERNEL_LAUNCH_CHECK();

  T h_out;
  RUNTIME_CHECK(DEV_MEMCPY(&h_out, d_out, sizeof(T), MEMCPY_D2H));

  RUNTIME_CHECK(DEV_FREE(d_in));
  RUNTIME_CHECK(DEV_FREE(d_out));
  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// ============================================================================
// 2) flashAttention (baseline, correctness-first)
// 目标：稳定通过测试用例（max diff/tolerance），所以采取“确定性累加”的做法。
// 设计：每个 block 负责一个 (b, t, qh)，threadIdx.x 覆盖 head_dim 的 d。
// - dot(q,k) 由 d==0 的线程串行累加（避免并行归约导致浮点顺序变化）
// - softmax 用 max-trick，防溢出
// - 输出对每个 d 独立累加 p*v
//
// 性能：不是最快，但足够通过并有一定并行度（在 d 维并行）。
// ============================================================================
template <typename T>
__global__ void flashattn_kernel(const T* __restrict__ q,
                                 const T* __restrict__ k,
                                 const T* __restrict__ v,
                                 T* __restrict__ o,
                                 int B, int Tt, int Ss, int QH, int KVH, int D, 
                                 bool causal) {
  const int b  = (int)blockIdx.x;
  const int t  = (int)blockIdx.y;
  const int qh = (int)blockIdx.z;
  const int d  = (int)threadIdx.x;

  if (b >= B || t >= Tt || qh >= QH || d >= D) return;

  // GQA map: qh -> kvh
  int kvh = 0;
  if (KVH > 0 && QH == KVH) kvh = qh;
  else if (KVH > 0 && (QH % KVH == 0)) kvh = qh / (QH / KVH);
  else if (KVH > 0) kvh = qh % KVH;

  // NOTE: scale uses 1/sqrtf rather than rsqrtf.
  // Some testcases are sensitive to 1-ulp differences in scale.
  const float scale = 1.0f / sqrtf((float)D);

  const size_t q_base = ((size_t)b * (size_t)Tt * (size_t)QH * (size_t)D)
                      + ((size_t)t * (size_t)QH * (size_t)D)
                      + ((size_t)qh * (size_t)D);
  const size_t o_base = q_base;

  __shared__ float sh_max;
  __shared__ float sh_p;    // broadcast exp(score-max)
  __shared__ float sh_sum;  // broadcast sum for normalization

  // ---------------- SoftMax Standard Pass 1: compute max score ----------------
  if (d == 0) {
    float m = -INFINITY;
    for (int s = 0; s < Ss; ++s) {
      if (causal && s > t) continue;

      //const size_t k_base = ((size_t)b * Ss * KVH * D) + ((size_t)s * KVH * D) + ((size_t)kvh * D);
      const size_t k_base = ((size_t)b * (size_t)Ss * (size_t)KVH * (size_t)D)
                          + ((size_t)s * (size_t)KVH * (size_t)D)
                          + ((size_t)kvh * (size_t)D);

      float dot = 0.0f;
      // Keep fmaf accumulation to match GPU reference behavior
      for (int dd = 0; dd < D; ++dd) {
        dot = fmaf(to_float_dev(q[q_base + (size_t)dd]),
                   to_float_dev(k[k_base + (size_t)dd]),
                   dot);
      }
      const float score = dot * scale;
      m = fmaxf(m, score);
    }
    sh_max = m;
  }
  __syncthreads();

  // ---------------- SoftMax Standard Pass 2: compute sum and output ----------------
  const float m = sh_max;

  float out = 0.0f;
  float sum = 0.0f;

  for (int s = 0; s < Ss; ++s) {
    if (causal && s > t) continue;

    const size_t k_base = ((size_t)b * (size_t)Ss * (size_t)KVH * (size_t)D)
                        + ((size_t)s * (size_t)KVH * (size_t)D)
                        + ((size_t)kvh * (size_t)D);
    const size_t v_base = k_base; // same layout for v

    if (d == 0) {
      float dot = 0.0f;
      for (int dd = 0; dd < D; ++dd) {
        dot = fmaf(to_float_dev(q[q_base + (size_t)dd]),
                   to_float_dev(k[k_base + (size_t)dd]),
                   dot);
      }
      const float score = dot * scale;
      sh_p = expf(score - m);
    }
    __syncthreads();

    const float p = sh_p;
    sum += p;
    out += p * to_float_dev(v[v_base + (size_t)d]);

    __syncthreads();
  }

  // Normalize: all threads should use the same denom
  if (d == 0) sh_sum = sum;
  __syncthreads();

  const float denom = sh_sum;
  if (denom > 0.0f) out /= denom;

  o[o_base + (size_t)d] = from_float_dev<T>(out);
}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const int B = batch_size, Tt = target_seq_len, Ss = src_seq_len;
  const int QH = query_heads, KVH = kv_heads, D = head_dim;

  const size_t q_sz = (size_t)B * (size_t)Tt * (size_t)QH * (size_t)D;
  const size_t k_sz = (size_t)B * (size_t)Ss * (size_t)KVH * (size_t)D;
  const size_t v_sz = (size_t)B * (size_t)Ss * (size_t)KVH * (size_t)D;
  const size_t o_sz = (size_t)B * (size_t)Tt * (size_t)QH * (size_t)D;

  h_o.resize(o_sz);

  T *d_q=nullptr, *d_k=nullptr, *d_v=nullptr, *d_o=nullptr;
  RUNTIME_CHECK(DEV_MALLOC(&d_q, q_sz * sizeof(T)));
  RUNTIME_CHECK(DEV_MALLOC(&d_k, k_sz * sizeof(T)));
  RUNTIME_CHECK(DEV_MALLOC(&d_v, v_sz * sizeof(T)));
  RUNTIME_CHECK(DEV_MALLOC(&d_o, o_sz * sizeof(T)));

  RUNTIME_CHECK(DEV_MEMCPY(d_q, h_q.data(), q_sz * sizeof(T), MEMCPY_H2D));
  RUNTIME_CHECK(DEV_MEMCPY(d_k, h_k.data(), k_sz * sizeof(T), MEMCPY_H2D));
  RUNTIME_CHECK(DEV_MEMCPY(d_v, h_v.data(), v_sz * sizeof(T), MEMCPY_H2D));

  // grid: (B, Tt, QH), block: (D)
  dim3 grid((unsigned)B, (unsigned)Tt, (unsigned)QH);
  dim3 block((unsigned)D, 1, 1);   // baseline：一维线程覆盖 head_dim

  // 若 D > 1024 需要拆分，这里假设测试用例 head_dim <= 256/512/1024
  flashattn_kernel<T><<<grid, block>>>(d_q, d_k, d_v, d_o, B, Tt, Ss, QH, KVH, D, is_causal);
  RUNTIME_CHECK(DEV_DEVICE_SYNC());

  RUNTIME_CHECK(DEV_MEMCPY(h_o.data(), d_o, o_sz * sizeof(T), MEMCPY_D2H));

  RUNTIME_CHECK(DEV_FREE(d_q));
  RUNTIME_CHECK(DEV_FREE(d_k));
  RUNTIME_CHECK(DEV_FREE(d_v));
  RUNTIME_CHECK(DEV_FREE(d_o));
     
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
