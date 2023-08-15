#pragma once

class Optimizer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
	typedef Vector::ConstAlignedMapType AlignedMapVec;

public:
	virtual ~Optimizer();

	virtual void reset() {};
	virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};