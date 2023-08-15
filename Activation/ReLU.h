#pragma once
#include <Eigen/Core>
#include "../Config.h"

class ReLU
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
	static inline void activate(const Matrix& Z, Matrix& A)
	{
		A.noalias() = Z;
	}

	static inline void apply_jacobian(const Matrix& Z, Matrix& A, const Matrix& F, Matrix& G)
	{
		G.noalias() = F;
	}


};