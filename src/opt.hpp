#ifndef OPT_HPP_
#define OPT_HPP_

#include <cassert>
#include "matrix.hpp"

namespace opt {

namespace {

using value_type = typename Matrix::value_type;

static constexpr size_t ALIGNMENT = 32;
static constexpr size_t N_R =  16;
static constexpr size_t M_R =   6;
static constexpr size_t M_C = 192;
static constexpr size_t K_C = 256;

template <typename T>
inline T *aligned_malloc(size_t n){
	void *ptr = nullptr;
	if(posix_memalign(&ptr, ALIGNMENT, n * sizeof(T)) != 0){ return nullptr; }
	return static_cast<T *>(ptr);
}

inline __m256i mm256_make_mask(size_t i){
	static const uint32_t table[] __attribute__((aligned(ALIGNMENT))) = {
		~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
		 0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u
	};
	if(i >= 8){
		return _mm256_load_si256(reinterpret_cast<const __m256i *>(table));
	}
	return _mm256_loadu_si256(
		reinterpret_cast<const __m256i *>(table + 8 - i));
}

inline void pack_Bp(value_type *Bp, const Matrix& B, size_t k0){
	const size_t M = B.cols(), K = B.rows();
	const size_t k_lo = k0, k_hi = std::min(K, k0 + K_C);
#pragma omp for
	for(size_t j0 = 0; j0 < M; j0 += N_R){
		const value_type *src = &B(k0, j0);
		value_type *dst = Bp + j0 * K_C;
		const __m256i mask0 = mm256_make_mask(M - j0);
		const __m256i mask1 = mm256_make_mask(j0 + 8 < M ? M - j0 - 8 : 0);
		for(size_t k = k_lo; k < k_hi; ++k){
			const __m256 b0 = _mm256_maskload_ps(src + 0, mask0);
			const __m256 b1 = _mm256_maskload_ps(src + 8, mask1);
			_mm256_store_ps(dst + 0, b0);
			_mm256_store_ps(dst + 8, b1);
			src += M;
			dst += N_R;
		}
	}
}

inline void pack_Ap(value_type *Ap, const Matrix& A, size_t k0, size_t i0){
	const size_t N = A.rows(), K = A.cols();
	const size_t k_lo = k0, k_hi = std::min(K, k0 + K_C);
	const size_t i_lo = i0, i_hi = std::min(N, i0 + M_C);
	for(size_t i = i_lo; i < i_hi; i += M_R){
		value_type *dst = Ap + (i - i_lo) * K_C;
		for(size_t k = k_lo; k < k_hi; ++k){
			*(dst++) = (i + 0 < N) ? A(i + 0, k) : 0;
			*(dst++) = (i + 1 < N) ? A(i + 1, k) : 0;
			*(dst++) = (i + 2 < N) ? A(i + 2, k) : 0;
			*(dst++) = (i + 3 < N) ? A(i + 3, k) : 0;
			*(dst++) = (i + 4 < N) ? A(i + 4, k) : 0;
			*(dst++) = (i + 5 < N) ? A(i + 5, k) : 0;
		}
	}
}

inline __m256 mm256_load_from_matrix(const Matrix& C, size_t i0, size_t j0){
	const size_t N = C.rows(), M = C.cols();
	if(i0 >= N || j0 >= M){ return _mm256_setzero_ps(); }
	const float *ptr = &C(i0, j0);
	if(j0 + 8 <= M){
		return _mm256_loadu_ps(ptr);
	}else{
		return _mm256_maskload_ps(ptr, mm256_make_mask(M - j0));
	}
}

inline void mm256_store_to_matrix(Matrix& C, size_t i0, size_t j0, __m256 x){
	const size_t N = C.rows(), M = C.cols();
	if(i0 >= N || j0 >= M){ return; }
	float *ptr = &C(i0, j0);
	if(j0 + 8 <= M){
		_mm256_storeu_ps(ptr, x);
	}else{
		_mm256_maskstore_ps(ptr, mm256_make_mask(M - j0), x);
	}
}

inline void compute_patch(
	Matrix& C,
	const value_type *Ap,
	const value_type *Bp,
	size_t i0,
	size_t j0,
	size_t k0,
	size_t K)
{
	const size_t N = C.rows(), M = C.cols();
	const size_t k_lo = k0, k_hi = std::min(K, k0 + K_C);

	float *cptr0 = &C(std::min(i0 + 0, N - 1), j0);

	const __m256 zero = _mm256_setzero_ps();
	__m256 c00 = zero, c01 = zero;
	__m256 c10 = zero, c11 = zero;
	__m256 c20 = zero, c21 = zero;
	__m256 c30 = zero, c31 = zero;
	__m256 c40 = zero, c41 = zero;
	__m256 c50 = zero, c51 = zero;

	const value_type *aptr = Ap;
	const value_type *bptr = Bp;

#define UNROLL_BLOCK() \
	do { \
		const __m256 b0 = _mm256_load_ps(bptr +  0); \
		const __m256 b1 = _mm256_load_ps(bptr +  8); \
		bptr += N_R; \
		const __m256 a0 = _mm256_broadcast_ss(aptr++); \
		c00 = _mm256_fmadd_ps(a0, b0, c00); \
		c01 = _mm256_fmadd_ps(a0, b1, c01); \
		const __m256 a1 = _mm256_broadcast_ss(aptr++); \
		c10 = _mm256_fmadd_ps(a1, b0, c10); \
		c11 = _mm256_fmadd_ps(a1, b1, c11); \
		const __m256 a2 = _mm256_broadcast_ss(aptr++); \
		c20 = _mm256_fmadd_ps(a2, b0, c20); \
		c21 = _mm256_fmadd_ps(a2, b1, c21); \
		const __m256 a3 = _mm256_broadcast_ss(aptr++); \
		c30 = _mm256_fmadd_ps(a3, b0, c30); \
		c31 = _mm256_fmadd_ps(a3, b1, c31); \
		const __m256 a4 = _mm256_broadcast_ss(aptr++); \
		c40 = _mm256_fmadd_ps(a4, b0, c40); \
		c41 = _mm256_fmadd_ps(a4, b1, c41); \
		const __m256 a5 = _mm256_broadcast_ss(aptr++); \
		c50 = _mm256_fmadd_ps(a5, b0, c50); \
		c51 = _mm256_fmadd_ps(a5, b1, c51); \
	} while(false)

	constexpr size_t UNROLL_COUNT = 16;
#define UNROLLED() \
	do { \
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); \
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); \
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); \
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); \
	} while(false)

	constexpr size_t PREFETCH_BREAK = 32;
	size_t k = k_lo;
	for(; k + UNROLL_COUNT + PREFETCH_BREAK <= k_hi; k += UNROLL_COUNT){
		UNROLLED();
	}
	_mm_prefetch(reinterpret_cast<const char *>(cptr0 + M * 5), _MM_HINT_T0);
	_mm_prefetch(reinterpret_cast<const char *>(cptr0 + M * 4), _MM_HINT_T0);
	_mm_prefetch(reinterpret_cast<const char *>(cptr0 + M * 3), _MM_HINT_T0);
	_mm_prefetch(reinterpret_cast<const char *>(cptr0 + M * 2), _MM_HINT_T0);
	_mm_prefetch(reinterpret_cast<const char *>(cptr0 + M), _MM_HINT_T0);
	_mm_prefetch(reinterpret_cast<const char *>(cptr0), _MM_HINT_T0);
	for(; k + UNROLL_COUNT <= k_hi; k += UNROLL_COUNT){ UNROLLED(); }
	for(; k < k_hi; ++k){ UNROLL_BLOCK(); }

#undef UNROLLED
#undef UNROLL_BLOCK

	if(j0 + N_R <= M){
		__m256 t0, t1;
#define WRITE_ROW(r) \
		do { \
			t0 = _mm256_loadu_ps(cptr0 + M * (r) + 0); \
			t1 = _mm256_loadu_ps(cptr0 + M * (r) + 8); \
			t0 = _mm256_add_ps(t0, c ## r ## 0); \
			t1 = _mm256_add_ps(t1, c ## r ## 1); \
			_mm256_storeu_ps(cptr0 + M * (r) + 0, t0); \
			_mm256_storeu_ps(cptr0 + M * (r) + 8, t1); \
		} while(false)
		switch(N - i0){
		default: WRITE_ROW(5);
		case 5:  WRITE_ROW(4);
		case 4:  WRITE_ROW(3);
		case 3:  WRITE_ROW(2);
		case 2:  WRITE_ROW(1);
		case 1:  WRITE_ROW(0);
		case 0:  break;
		}
#undef WRITE_ROW
	}else{
		const __m256i mask0 = mm256_make_mask(M - j0);
		const __m256i mask1 = mm256_make_mask(j0 + 8 < M ? M - j0 - 8 : 0);
		__m256 t0, t1;
#define WRITE_ROW(r) \
		do { \
			t0 = _mm256_maskload_ps(cptr0 + M * (r) + 0, mask0); \
			t1 = _mm256_maskload_ps(cptr0 + M * (r) + 8, mask1); \
			t0 = _mm256_add_ps(t0, c ## r ## 0); \
			t1 = _mm256_add_ps(t1, c ## r ## 1); \
			_mm256_maskstore_ps(cptr0 + M * (r) + 0, mask0, t0); \
			_mm256_maskstore_ps(cptr0 + M * (r) + 8, mask1, t1); \
		} while(false)
		switch(N - i0){
		default: WRITE_ROW(5);
		case 5:  WRITE_ROW(4);
		case 4:  WRITE_ROW(3);
		case 3:  WRITE_ROW(2);
		case 2:  WRITE_ROW(1);
		case 1:  WRITE_ROW(0);
		case 0:  break;
		}
#undef WRITE_ROW
	}
}

}

void matmul(Matrix& C, const Matrix& A, const Matrix& B){
	using allocator_type = AlignedAllocator<value_type, ALIGNMENT>;
	const size_t N = A.rows(), M = B.cols(), K = A.cols();
	assert(B.rows() == K);
	assert(C.rows() == N);
	assert(C.cols() == M);

	const size_t kBp = (M + N_R - 1) / N_R;
	std::unique_ptr<value_type, decltype(&free)> Bp(
		aligned_malloc<value_type>(kBp * N_R * K_C), free);

	static thread_local value_type Ap[M_C * K_C]
		__attribute__((aligned(ALIGNMENT)));

#pragma omp parallel
	for(size_t k0 = 0; k0 < K; k0 += K_C){
		pack_Bp(Bp.get(), B, k0);
#pragma omp barrier
#pragma omp for
		for(size_t i0 = 0; i0 < N; i0 += M_C){
			const size_t i_lo = i0, i_hi = std::min(N, i0 + M_C);
			pack_Ap(Ap, A, k0, i0);
			for(size_t j = 0; j < M; j += N_R){
				for(size_t i = i_lo; i < i_hi; i += M_R){
					compute_patch(
						C, Ap + (i - i_lo) * K_C, Bp.get() + j * K_C,
						i, j, k0, K);
				}
			}
		}
#pragma omp barrier
	}

}

}

#endif
