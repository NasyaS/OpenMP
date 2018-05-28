#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <random>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cmath>

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

string test2(Matrix &matrix,double time,double iters){ 
	stringstream outStream;

  for (size_t i = 0; i < matrix.rows(); ++i) {

    for (size_t j = 0; j < matrix.cols(); ++j) {

      if(j==matrix.cols()-1){

        outStream << matrix(i, j) << "";
      }
      else{

        outStream << matrix(i, j) << ",";
      }
      
    }
    outStream<<endl;
  }
  outStream<<time<<","<<iters<<endl;
  return outStream.str();

}

double F(double x,double y){

	return 0;
}
double G(double x,double y){

	if(x==0)return 1;
	if(x==1)return exp(y);
	if(y==0)return 1;
	if(y==1)return exp(x);
	else return 0;
	return 0;
	
}


size_t dirichletProblem(Matrix &u,Matrix &f,size_t N,double epsilon){

	double h=1.0/(N+1);
	size_t iters=0;
	double max=0;
	double delta=0;
	double t1=0;
	for(size_t i=0;i<N;++i){

		for(size_t j=0;j<N;++j)

			f(i,j)=F((i+1)*h,(j+1)*h);
			
		
	}

	for(size_t i=1;i<N+1;++i){

		u(i,0)=G(h*i,0);
		u(i,N+1)=G(i*h,(N+1)*h);
		
	}	

	for(size_t j=0;j<N+2;++j){
		
		u(0,j)=G(0,j*h);
		u(N+1,j)=G((N+1)*h,j*h);
		
	}		

	do {
		iters++;
		max=0;
		for(size_t i=1;i<N+1;++i)
			for(size_t j=1;j<N+1;++j){

				t1=u(i,j);
				u(i,j)=0.25*(u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - h*h*f(i - 1, j - 1));
				delta=fabs(u(i,j)-t1);
				if (delta>max) {

					max=delta;
				}
						
	
					
		}	

	}while(max>epsilon);

	return iters;
}


size_t dirichletProblemParallel(Matrix &u,Matrix &f,size_t N,double epsilon){

	double h=1.0/(N+1);
	size_t iters=0;
	double max=0;
	double delta=0;
	double t1=0;
	for(size_t i=0;i<N;++i){

		for(size_t j=0;j<N;++j)

			f(i,j)=F((i+1)*h,(j+1)*h);
			
		
	}

	for(size_t i=1;i<N+1;++i){

		u(i,0)=G(h*i,0);
		u(i,N+1)=G(i*h,(N+1)*h);
		
	}	

	for(size_t j=0;j<N+2;++j){
		
		u(0,j)=G(0,j*h);
		u(N+1,j)=G((N+1)*h,j*h);
		
	}		
	size_t j=0;
	do {
		iters++;
		max=0;
		#pragma omp parallel for private(j,t1) reduction (max:delta) 
		for(size_t i=1;i<N+1;++i)
			for(j=1;j<N+1;++j){

				t1=u(i,j);
				u(i,j)=0.25*(u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - h*h*f(i - 1, j - 1));
				delta=fabs(u(i,j)-t1);
				if (delta>max) {

					max=delta;
				}
						
	
					
		}	

	}while(max>epsilon);

	return iters;
}

int main(int argc, char* argv[]){

	size_t N = 99;
	if (argc > 1){
    istringstream ss(argv[1]);
    int dim;
    if (!(ss >> dim)){
      throw invalid_argument("Ошибка");
    } else {
      N = dim;
	    }
	} 
	double epsilon=0.0001;
	Matrix u(N+2,N+2);
	Matrix f(N,N);
	auto firstTime = chrono::steady_clock::now();

	// size_t iters=dirichletProblem(u,f,N,epsilon);
	size_t iters=dirichletProblemParallel(u,f,N,epsilon);
	
	auto secondTime = chrono::steady_clock::now();

	auto Time = chrono::duration_cast<chrono::duration<double>>(secondTime - firstTime);
	cout<<test2(u,Time.count(),iters)<<endl;
	return 0;
}