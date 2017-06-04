/*这是一段用LevenbergMarquardt最小化y^2x-5*x的代码*/
/*非线性最小化问题（局部最小）*/
/*求Jacobian分别用解析法和数值法*/
//http://www.cnblogs.com/is-smiling/archive/2013/05/12/3074259.html

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

struct MyObjectFunctor
{
  int inputs() const { return 1; } // dimention of X
  int values() const { return 1; } // number of constraints

  //optional, for init value
  MyObjectFunctor(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) { }

  int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
  {
    // Implement y = x^2-5*x
    fvec(0) = x(0)*x(0) - 5.0f * x(0);
    return 0;
  }

//#define Analytical
#ifdef Analytical
  //Analytical
  int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
  {
    // Implement dy/dx = 2*x
    for (int i = 0; i < inputs(); i++)
    {
      fjac(i, 0) = 2.0f * x(0) - 5.0f;
    }
    return 0;
  }
#else
  //numerical
  int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const //fjac Jacobian is a Matrix
  {
    float epsilon = 1e-5f;
    Eigen::VectorXf xp = x;
    for (int i = 0; i<inputs(); i++)
    {
      xp = x;
      xp(i) = x(i) + epsilon;
      Eigen::VectorXf fvec1(1);
      (*this)(xp, fvec1);

      xp(i) = x(i) - epsilon;
      Eigen::VectorXf fvec2(1);
      (*this)(xp, fvec2);

       fjac(i) = (fvec1(i) - fvec2(i)) / (2.0f*epsilon);
    }
    std::cout << "ok df()\n";
    return 0;
  }
#endif
};

int main(int argc, char *argv[])
{
  Eigen::VectorXf x(1);  
  Eigen::VectorXf y(1);

  x(0) = 200;
  std::cout << "Init x: " << x << std::endl;
 
  MyObjectFunctor functor(x,y);
  Eigen::LevenbergMarquardt<MyObjectFunctor,float> lm(functor);
  lm.parameters.maxfev=8;//max iter number
   //lm.parameters.xtol = 0.0001;
  int ret = lm.minimize(x);
 
  std::cout << "x that minimizes the function: " << x << std::endl;
  std::cout <<"Iter times: "<< lm.iter <<std::endl;

  //evaluation  
  functor(x,y); 
  std::cout <<"Value y = "<<y<<std::endl;
  return 0;
}
 