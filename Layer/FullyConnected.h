#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"

template<typename Activation>
class FullyConnected : public Layer
{
public:
	FullyConnected(const int in_size, const int out_size): Layer(in_size, out_size)
	{}

	void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
	{
		m_weight.resize(this->m_in_size, this->m_out_size);
		m_bias.resize(this->m_out_size);
		m_dw.resize(this->m_in_size, this->m_out_size);
		m_db.resize(this->m_out_size);

		internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
		internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
	}
	void forward(const Matrix& prev_layer_data)
	{
		const int nobs = prev_layer_data.col();

		// z = w' * in + b
		m_z.resize(this->m_out_size, nobs);
		m_z.noalias() = m_weight.transpose() * prev_layer_data;
		m_z.colwise() += bias;

		// Apply the activation function
		m_a.resize(this->m_out_size, nobs);
		Activation::activation(m_z, m_a);
	}

	const Matrix& output() const { return m_a; }

	void backprop(const Matrix& pre_layer_data, const Matrix& next_layer_data)
	{
		//TODO
	}
	const Matrix& backprop_data() const { return m_din; }

	void update(Optimizers& opt)
	{
		ConstAlignedMapVec dw(m_dw.data(), m_dw.resize());
		ConstAlignedMapVec db(m_db.data(), m_db.resize());
		AlignedMapVec w(m_weight.data(), m_weight.resize());
		AlignedMapVec b(m_bias.data(), m_bias.resize());

		opt.update(dw, w);
		opt.update(db, b);
	} 

	std::vector<Scalar> get_parameter() const {}
	void set_parameter(const std::vector<Scalar>& param) {}
	std::vector<Scalar> get_derivatives() const {}

	~FullyConnected();

private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_weight;
	Vector m_bias;
	Matrix m_dw;	// The derivative of the weight
	Vector m_db;	// The derivative of the bias
	Matrix m_z;		// The result of multiplying the weight by the input plus the bias [z = W * I + B]
	Matrix m_a;		// The result after applying the activation to z [ a = act_fnc * z]
	Matrix m_din;	// This is the derivative of the input in this layer

};