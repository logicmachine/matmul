#ifndef OPT_HPP_
#define OPT_HPP_

#include <cassert>
#include "matrix.hpp"

namespace opt {

namespace {

using value_type = typename Matrix::value_type;

static constexpr size_t ALIGNMENT = 32;
static constexpr size_t N_R =  24;
static constexpr size_t M_R =   4;
static constexpr size_t M_C = 256;
static constexpr size_t K_C = 256;

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
		const __m256i mask1 = mm256_make_mask(j0 +  8 < M ? M - j0 -  8 : 0);
		const __m256i mask2 = mm256_make_mask(j0 + 16 < M ? M - j0 - 16 : 0);
		for(size_t k = k_lo; k < k_hi; ++k){
			const __m256 b0 = _mm256_maskload_ps(src +  0, mask0);
			const __m256 b1 = _mm256_maskload_ps(src +  8, mask1);
			const __m256 b2 = _mm256_maskload_ps(src + 16, mask2);
			_mm256_store_ps(dst +  0, b0);
			_mm256_store_ps(dst +  8, b1);
			_mm256_store_ps(dst + 16, b2);
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
			*(dst++) = A(i + 0, k);
			*(dst++) = A(i + 1, k);
			*(dst++) = A(i + 2, k);
			*(dst++) = A(i + 3, k);
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
	const size_t k_lo = k0, k_hi = std::min(K, k0 + K_C);
	__m256 c00 = mm256_load_from_matrix(C, i0 + 0, j0 +  0);
	__m256 c01 = mm256_load_from_matrix(C, i0 + 0, j0 +  8);
	__m256 c02 = mm256_load_from_matrix(C, i0 + 0, j0 + 16);
	__m256 c10 = mm256_load_from_matrix(C, i0 + 1, j0 +  0);
	__m256 c11 = mm256_load_from_matrix(C, i0 + 1, j0 +  8);
	__m256 c12 = mm256_load_from_matrix(C, i0 + 1, j0 + 16);
	__m256 c20 = mm256_load_from_matrix(C, i0 + 2, j0 +  0);
	__m256 c21 = mm256_load_from_matrix(C, i0 + 2, j0 +  8);
	__m256 c22 = mm256_load_from_matrix(C, i0 + 2, j0 + 16);
	__m256 c30 = mm256_load_from_matrix(C, i0 + 3, j0 +  0);
	__m256 c31 = mm256_load_from_matrix(C, i0 + 3, j0 +  8);
	__m256 c32 = mm256_load_from_matrix(C, i0 + 3, j0 + 16);

	const value_type *aptr = Ap;
	const value_type *bptr = Bp;

#define UNROLL_BLOCK() \
	do { \
		const __m256 b0 = _mm256_load_ps(bptr +  0); \
		const __m256 b1 = _mm256_load_ps(bptr +  8); \
		const __m256 b2 = _mm256_load_ps(bptr + 16); \
		bptr += N_R; \
		const __m256 a0 = _mm256_broadcast_ss(aptr++); \
		c00 = _mm256_fmadd_ps(a0, b0, c00); \
		c01 = _mm256_fmadd_ps(a0, b1, c01); \
		c02 = _mm256_fmadd_ps(a0, b2, c02); \
		const __m256 a1 = _mm256_broadcast_ss(aptr++); \
		c10 = _mm256_fmadd_ps(a1, b0, c10); \
		c11 = _mm256_fmadd_ps(a1, b1, c11); \
		c12 = _mm256_fmadd_ps(a1, b2, c12); \
		const __m256 a2 = _mm256_broadcast_ss(aptr++); \
		c20 = _mm256_fmadd_ps(a2, b0, c20); \
		c21 = _mm256_fmadd_ps(a2, b1, c21); \
		c22 = _mm256_fmadd_ps(a2, b2, c22); \
		const __m256 a3 = _mm256_broadcast_ss(aptr++); \
		c30 = _mm256_fmadd_ps(a3, b0, c30); \
		c31 = _mm256_fmadd_ps(a3, b1, c31); \
		c32 = _mm256_fmadd_ps(a3, b2, c32); \
	} while(false)

	size_t k = k_lo;
	for(; k + 16 <= k_hi; k += 16){
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK();
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK();
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK();
		UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK(); UNROLL_BLOCK();
	}
	for(; k < k_hi; ++k){
		UNROLL_BLOCK();
	}
	mm256_store_to_matrix(C, i0 + 0, j0 +  0, c00);
	mm256_store_to_matrix(C, i0 + 0, j0 +  8, c01);
	mm256_store_to_matrix(C, i0 + 0, j0 + 16, c02);
	mm256_store_to_matrix(C, i0 + 1, j0 +  0, c10);
	mm256_store_to_matrix(C, i0 + 1, j0 +  8, c11);
	mm256_store_to_matrix(C, i0 + 1, j0 + 16, c12);
	mm256_store_to_matrix(C, i0 + 2, j0 +  0, c20);
	mm256_store_to_matrix(C, i0 + 2, j0 +  8, c21);
	mm256_store_to_matrix(C, i0 + 2, j0 + 16, c22);
	mm256_store_to_matrix(C, i0 + 3, j0 +  0, c30);
	mm256_store_to_matrix(C, i0 + 3, j0 +  8, c31);
	mm256_store_to_matrix(C, i0 + 3, j0 + 16, c32);
}

}

void matmul(Matrix& C, const Matrix& A, const Matrix& B){
	using allocator_type = AlignedAllocator<value_type, ALIGNMENT>;
	const size_t N = A.rows(), M = B.cols(), K = A.cols();
	assert(B.rows() == K);
	assert(C.rows() == N);
	assert(C.cols() == M);

	const size_t kBp = (M + N_R - 1) / N_R * N_R;
	std::unique_ptr<value_type, decltype(&free)> Bp(
		static_cast<value_type *>(aligned_alloc(
			sizeof(value_type) * kBp * N_R * K_C, ALIGNMENT)),
		free);

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
