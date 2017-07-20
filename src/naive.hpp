#ifndef NAIVE_HPP_
#define NAIVE_HPP_

#include <cassert>
#include "matrix.hpp"

namespace naive {

void matmul(Matrix& C, const Matrix& A, const Matrix& B){
	using value_type = Matrix::value_type;
	const size_t N = A.rows(), M = B.cols(), K = A.cols();
	assert(B.rows() == K);
	assert(C.rows() == N);
	assert(C.cols() == M);
#pragma omp parallel for
	for(size_t i = 0; i < N; ++i){
		for(size_t j = 0; j < M; ++j){
			value_type sum = 0;
			for(size_t k = 0; k < K; ++k){
				sum += A(i, k) * B(k, j);
			}
			C(i, j) = sum;
		}
	}
}

}

#endif
