#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <vector>
#include "aligned_allocator.hpp"

class Matrix {

public:
	using value_type = float;
	using allocator_type = AlignedAllocator<value_type, 32>;

private:
	size_t m_rows;
	size_t m_cols;
	std::vector<value_type, allocator_type> m_data;

public:
	Matrix()
		: m_rows(0)
		, m_cols(0)
		, m_data()
	{ }

	Matrix(size_t rows, size_t cols)
		: m_rows(rows)
		, m_cols(cols)
		, m_data(rows * cols)
	{ }


	size_t rows() const {
		return m_rows;
	}

	size_t cols() const {
		return m_cols;
	}


	const value_type& operator()(size_t i, size_t j) const {
		return m_data[i * m_cols + j];
	}

	value_type& operator()(size_t i, size_t j){
		return m_data[i * m_cols + j];
	}

};

#endif
