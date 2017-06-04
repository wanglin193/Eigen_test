//created and tested by Wang Lin
//wanglin193 at gmail.com

#include <iostream>
#include <Eigen/Dense> 
#include <Eigen/Geometry> 
#include <unsupported/Eigen/NonLinearOptimization> 

#include "FisheyeModel.h"

//Optimization Fisheye Camera Intrinsic and Extrinsic by Eigen::LevenbergMarquardt<>
//Single view(image) only in this example, 9+6 parameters
 
//--------------------------------------------------------------------------------------------
// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  // you should define that in the subclass :
  // void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

//也可以直接定义在Functor里
struct projective_functor : Functor<double>
{
  typedef InputType::Scalar Scalar;

  //set dimention of parameters
  projective_functor(int num_in, int num_val) : Functor<double>(num_in, num_val) {}

  //constant variables
  Eigen::Matrix<double, 3, Eigen::Dynamic>  ObjectP3d;
  Eigen::Matrix<double, 2, Eigen::Dynamic>  ImageP2d;
  
  //params:优化变量
  //fvec:残差矢量，最小二乘的优化函数目标是fvec.norm()最小
  int operator() (const InputType& params, ValueType& fvec ) const
  {   
    //for each 2d point     
    for (int i = 0; i < values() / 2; i++)
    {
      Eigen::Vector2d  v, P2d = ImageP2d.col(i);
      P3d_to_P2d<Scalar>(ObjectP3d.col(i), v, params);
      fvec[2 * i] = P2d(0) - v(0);
      fvec[2 * i + 1] = P2d(1) - v(1);
    }
    //std::cout << fvec.norm()  ;
    return 0;
  }

  //手工计算Jacobian的版本，这里用Numerical Method
  int df(const InputType& x, JacobianType& fjac) const
  {
    ValueType fvec1(values());
    ValueType fvec2(values());

    Scalar epsilon = 1e-8f;
    InputType xp = x;
    auto inv = 1.0 / (2.0f*epsilon); 
    
    //add disturb to each parameters
    for (int i = 0; i < inputs(); i++)
    {
      xp(i) = x(i) + epsilon;   
      (*this)(xp, fvec1); 
   
      xp(i) = x(i) - epsilon;    
      (*this)(xp, fvec2);

      fjac.col(i) = (fvec1 - fvec2)*inv;
    }
    //LM不是每次调用operator都同时计算df的
    //std::cout << "ok df()\n";
    return 0;
  }
};

#include<chrono>
//单张图片 LM法估计鱼眼相机内参外参
void test_fisheye_singleview_calibrate_LM()
{
  using namespace Eigen;
  using namespace std;

  typedef Eigen::Matrix<double, 9, 1> TypeCamIntrinsic;
  typedef Eigen::Matrix<double, 6, 1> TypeCamExtrinsic;

  //-------------------------------generate ground truth
  float tagsize = 26.7;//mm
  Eigen::Matrix<double, 3, Dynamic> ObjectP3d;
  generatePtsInPlane(tagsize, tagsize / 4, tagsize / 4, 8, 6, ObjectP3d);

  //Intrinsic: 9 camera parameters
  TypeCamIntrinsic RealCameraParams;
  RealCameraParams << 278.421, 277.217, 322.9450, 238.80910, 0.0,
    1.9879565994521477e-02, -4.8905763164846051e-02, 4.1867941284700234e-02, -1.2773104992844659e-02;

  //Extrinsic: 6-dof rotation and motion
  TypeCamExtrinsic RealtCameraPoses;
  RealtCameraPoses << -9.1757138480175165e-02, 3.1935484604661840e-01, 4.3150647234269382e-02, -1.6446818, -46.849107647, 153.1005;

  //put all params to a vector
  Eigen::Matrix<double, 15, 1> params;
  params << RealCameraParams, RealtCameraPoses;

  //gernerate image points  
  Eigen::Matrix<double, Dynamic, Dynamic> ImageP2d;
  ImageP2d.resize(2, ObjectP3d.cols());
  for (int i = 0; i < ObjectP3d.cols(); i++)
  {
    Vector2d P2d;
    Vector3d P3d = ObjectP3d.col(i);
    P3d_to_P2d<double>(P3d, P2d, params);
    ImageP2d.col(i) = P2d;
  }
  //cout << ImageP2d << endl;
  //--------------------------------init camera params and poses
  TypeCamIntrinsic InitCameraParams;//intrinsic
  TypeCamExtrinsic InitCameraPoses; //extrinsic

  InitCameraParams << 280, 280, 320, 240, 0, 0, 0, 0, 0;
  InitCameraPoses << 0, 0, 0.0001, 0, 0, 100;

  //------------------------------------------------------------------
  //all parameters need to campact to a vector 
  Eigen::VectorXd p(15);
  p << InitCameraParams, InitCameraPoses;

  //输入变量15，输出是图像投影点的2倍(x,y两坐标)
  projective_functor foo(15, 2 * ImageP2d.cols());

  //设置3D-2D对应数据
  foo.ObjectP3d = ObjectP3d;
  foo.ImageP2d = ImageP2d;

  //test functor
  if (1)
  {
    VectorXd residual(2 * ImageP2d.cols());
    foo(p, residual);
    cout << "Init Energy: " << residual.norm() << endl;
  }

  auto eigen_begin = std::chrono::steady_clock::now(); 

#if 1
  //auto cal numerical diff 不调用自定义df
  Eigen::NumericalDiff<projective_functor> numdiff(foo);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<projective_functor>, double> lm(numdiff);
#else
  //calculate df by hand,调用自定义df
  Eigen::LevenbergMarquardt<projective_functor> lm(foo);
#endif

  lm.parameters.maxfev = 100;//max iter number
 //lm.parameters.gtol = 0.01;
 //lm.parameters.xtol = 0.1;

  //do computation
  int ret = lm.minimize(p);

  auto eigen_end = std::chrono::steady_clock::now();
  auto eigen_time = std::chrono::duration_cast<std::chrono::duration<double> >(eigen_end - eigen_begin).count();
  
  std::cout << "time (ms): " << 1000*eigen_time << std::endl;
 
  //http://eigen.tuxfamily.org/dox/unsupported/group__NonLinearOptimization__Module.html

  //cout << "blueNorm of residual: " << lm.fvec.blueNorm() << endl; 
  cout << "number of iterations performed: " << lm.iter << endl;
  cout << "number of functions evaluation: " << lm.nfev << ", number of jacobian evaluation: " << lm.njev << endl;
  cout << "Result parameters: " << p.transpose() << endl;

  //test functor
  if (1)
  {
    VectorXd residual(2 * ImageP2d.cols());
    foo(p, residual);
    cout << "Final Energy: " << residual.norm() << endl;
  }
}

void main()
{
  test_fisheye_singleview_calibrate_LM();
}