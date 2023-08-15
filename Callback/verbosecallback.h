#pragma once
#include <Eigen/Core>
#include <iostream>
#include "../Config.h"
#include "../Callback.h"
#include "../NeuralNet.h"

class VerboseCallback : public Callback
{
	void post_trainign_batch(const NeuralNet* net, const Matrix& x, const Matrix& y)
	{
		const Scalar loss = net->get_output()->loss();
		std::cout << "[Epoch " << m_epoch_id << ", batch " << m_bath_id << "] Loss = " << loss << std::endl;
		m_loss.push_back(loss);
	}

	void post_trainign_batch(const NeuralNet* net, const Matrix& x, const IntergerVector& y)
	{
		const Scalar loss = net->get_output()->loss();
		std::cout << "[Epoch " << m_epoch_id << ", batch " << m_bath_id << "] Loss = " << loss << std::endl;
		m_loss.push_back(loss);
	}
};