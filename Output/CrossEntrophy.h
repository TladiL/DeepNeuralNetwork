#pragma once

#include <Eigen/Core>
#include "../Config.h"

class CrossEntrophy : public Output
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_din;

public:
	void evaluate(const Matrix& prev_layer_data, const Matrix& target)
	{
		const int nobs = prev_layer_data.cols();
		const int nvar = prev_layer_data.rows();

		// Complete the derivative of the input layer
		m_din.resize(nvar, nobs);
		m_din.noalias() = -target.cwiseQuotient(prev_layer_data);
	}

	void evaluate(const Matrix& prev_layer_data, const Intergervector& target) 
	{
		const int nobs = prev_layer_data.cols();
		const int nvar = prev_layer_data.rows();

		// Complete the derivative of the input layer
		m_din.resize(nvar, nobs);
		m_din.setZero();

		for (int i = 0; i < nobs; i++)
		{
			m_din(target[i], i) = -Scalar(1) / prev_layer_data(target[1], i);
		}
	}

	const Matrix& backprop_data() const { return m_din; }

	Scalar loss() const
	{
		Scalar result = Scalar(0);
		const int nelem = m_din.size();
		const Scalar* din_data = m_din.data();

		for (int i = 0; i < nelem; i++)
		{
			if (din_data[i] < 0)
			{
				result += std::log(-din_data[i]);
			}
		}
		return result / m_din.cols();
	}
};