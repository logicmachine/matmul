#include <iostream>
#include <random>
#include <chrono>
#include "matrix.hpp"
#include "naive.hpp"
#include "opt.hpp"

template <class Random>
inline void random_fill(Matrix& matrix, Random& engine){
	using value_type = typename Matrix::value_type;
	std::uniform_real_distribution<value_type> dist(-1.0, 1.0);
	for(size_t i = 0; i < matrix.rows(); ++i){
		for(size_t j = 0; j < matrix.cols(); ++j){
			matrix(i, j) = dist(engine);
		}
	}
}

template <class Func>
inline void benchmark_run(
	Matrix& C, const Matrix& A, const Matrix& B, Func&& func)
{
	namespace chrono = std::chrono;
	using duration_type = chrono::duration<double>;
	const auto begin = chrono::steady_clock::now();
	func(C, A, B);
	const auto end = chrono::steady_clock::now();
	const auto duration = chrono::duration_cast<duration_type>(end - begin);
	const size_t N = A.rows(), M = B.cols(), K = A.rows();
	const size_t flops = 2 * N * M * K;
	std::cout << duration.count() << " [sec]" << std::endl;
	std::cout << (flops / duration.count() * 1e-9) << " [GFLOPS]" << std::endl;
}

int main(){
	size_t N, M, K;
	std::cin >> N >> M >> K;
	Matrix A(N, K), B(K, M), C_naive(N, M), C_opt(N, M);

	std::default_random_engine engine;
	random_fill(A, engine);
	random_fill(B, engine);

	//std::cout << "Naive:" << std::endl;
	//benchmark_run(C_naive, A, B, naive::matmul);

	std::cout << "Opt:" << std::endl;
	benchmark_run(C_opt, A, B, opt::matmul);

	float max_error = 0.0f;
	for(size_t i = 0; i < N; ++i){
		for(size_t j = 0; j < M; ++j){
			max_error = std::max(
				max_error, fabsf(C_naive(i, j) - C_opt(i, j)));
		}
	}
	std::cout << "Max error: " << max_error << std::endl;

	return 0;
}

