#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <random>
#include <chrono>
#include <stdexcept>
#include <sstream>

using namespace std;
vector<double> random_parallel(size_t size);

class Matrix {
private:
	size_t m_rows;
	size_t m_cols;
	vector<double> m_data;

	Matrix(size_t rows, size_t cols,const vector<double> &data): m_rows(rows), m_cols(cols), m_data(data){}

public:

	Matrix(size_t rows, size_t cols): m_rows(rows), m_cols(cols), m_data(rows*cols) {}

	size_t rows() const
	{
		return m_rows;
	}

	size_t cols() const
	{
		return m_cols;
	}

	double &operator() (size_t row, size_t col) 
	{
		return m_data[row*m_cols+col]; 
	}

	const double &operator() (size_t row, size_t col) const
	{
		return m_data[row*m_cols+col]; 
	}

	static Matrix random(size_t rows, size_t cols)
	{
		return Matrix(rows, cols, random_parallel(rows*cols));
	}

	Matrix operator * (const Matrix &matrix);

};

Matrix Mult(const Matrix &X, const Matrix &Y){

	Matrix Z(X.rows(), Y.cols());

	if (X.cols() == Y.rows()) {

		for (size_t i = 0; i < Z.cols(); ++i)
		{
			for (size_t j = 0; j < Z.cols(); ++j)
			{
				for (size_t k = 0; k < Z.rows(); ++k)
				{
					Z(i,j)+=X(i,k)*Y(k,j);
				}
			}
		}
	}
	else
		throw invalid_argument("Ошибка размерностей!");

	return Z;
}

Matrix ParallelMult(const Matrix &X, const Matrix &Y){

	Matrix Z(X.rows(), Y.cols());

	if (X.cols() == Y.rows()) {

		#pragma omp parallel for shared(X,Y,Z)

		for (size_t j = 0; j < Z.cols(); ++j)
		{
			for (size_t k = 0; k < Z.cols(); ++k)
			{
				for (size_t i = 0; i < Z.rows(); ++i)
				{
					Z(i,j)+=X(i,k)*Y(k,j);
				}
			}
		}
	}
	else
		throw invalid_argument("Ошибка размерностей!");

	return Z;
}

string ToString(const Matrix &matrix){
	
	stringstream outStream;

	for(size_t i = 0; i < matrix.rows(); ++i) {
		for(size_t j = 0; j < matrix.cols(); ++j) {
			outStream << matrix(i, j) << " ";
		}
		outStream << endl;
	}
	return outStream.str();
}

vector<double> random_parallel(size_t size){

	vector<double>result(size);

	#pragma omp parallel shared(result)
	{
		random_device rd;
		mt19937_64 gen(rd());
		chi_squared_distribution<double>dis(2);

		#pragma omp for schedule(static)
		for (size_t i = 0; i < size; i++) {
			result[i] = dis(gen);
		}
	}

	return result;
}


Matrix Matrix::operator*(const Matrix &matrix) {
	return ParallelMult((*this), matrix);
}

void Test(size_t t, size_t d, double time) {
	cout << t << "," << d << "," << time << endl;
}



int main(int argc, char* argv[]){

	size_t rows = 4;
	size_t cols = 4;	

	if (argc > 1){
    istringstream ss(argv[1]);
    int dim;
    if (!(ss >> dim)){
      throw invalid_argument("Ошибка");
    } else {
      rows = dim;
      cols = dim;
	    }
	} 

	auto firstTime = chrono::steady_clock::now();

	Matrix X = Matrix::random(rows, cols);
	Matrix Y = Matrix::random(rows, cols);
	
	Matrix Z = X*Y;

	auto secondTime = chrono::steady_clock::now();

	auto Time = chrono::duration_cast<chrono::duration<double>>(secondTime - firstTime);
	Test(omp_get_max_threads(), rows, Time.count());


	return 0;
}