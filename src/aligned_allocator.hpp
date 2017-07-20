#ifndef ALIGNED_ALLOCATOR_HPP_
#define ALIGNED_ALLOCATOR_HPP_

#include <memory>
#include <cstdlib>

template <typename T, size_t ALIGNMENT>
class AlignedAllocator {

public:
	using value_type = T;

	template <typename U>
	struct rebind {
		using other = AlignedAllocator<U, ALIGNMENT>;
	};

	AlignedAllocator() noexcept { }

	template <typename U>
	AlignedAllocator(const AlignedAllocator<U, ALIGNMENT>&) noexcept { }


	value_type *allocate(size_t n){
		void *ptr = nullptr;
		if(posix_memalign(&ptr, ALIGNMENT, n * sizeof(float)) != 0){
			return nullptr;
		}else{
			return reinterpret_cast<value_type *>(ptr);
		}
	}

	void deallocate(value_type *p, size_t){
		free(p);
	}

};


template <typename T, typename U, size_t ALIGNMENT>
inline bool operator==(
	const AlignedAllocator<T, ALIGNMENT>&,
	const AlignedAllocator<U, ALIGNMENT>&)
{
	return true;
}

template <typename T, typename U, size_t ALIGNMENT>
inline bool operator!=(
	const AlignedAllocator<T, ALIGNMENT>&,
	const AlignedAllocator<U, ALIGNMENT>&)
{
	return false;
}

#endif
