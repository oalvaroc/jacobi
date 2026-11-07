#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

template <typename T>
class mat2d {
public:
	mat2d(size_t order) : order_(order), data_(order * order) {}

	template <typename Generator>
	static mat2d make_random(size_t order, Generator gen) {
		mat2d mat(order);
		std::generate(mat.data_.begin(), mat.data_.end(), gen);
		return mat;
	}

	const T& operator[](size_t row, size_t col) const {
		assert(row < order_ && col < order_);
		return data_[row * order_ + col];
	}

	T& operator[](size_t row, size_t col) {
		assert(row < order_ && col < order_);
		return data_[row * order_ + col];
	}

	const T *data() const {
		return data_.data();
	}

	T *data() {
		return data_.data();
	}

	size_t order() const {
		return order_;
	}

	void swap(mat2d<T>& other) noexcept {
		std::swap(order_, other.order_);
		std::swap(data_, other.data_);
	}

private:
	size_t order_;
	std::vector<T> data_;
};
