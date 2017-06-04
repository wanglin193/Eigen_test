//created and tested by Wang Lin
//wanglin193 at gmail.com

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "FisheyeModel.h"

//define AD will make link step very slow

//#define AD

template<typename Scalar, int NX = Dynamic, int NY = Dynamic>
struct FuctorPointProjection
{
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };

  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  FuctorPointProjection() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  FuctorPointProjection(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  //eigen 3.2.8不支持在变量之后输入常数参数
  template<typename T>
  void operator() (const Eigen::Matrix<T, InputsAtCompileTime, 1>& x,
    Eigen::Matrix<T, ValuesAtCompileTime, 1>* _v,
    const Eigen::Matrix<Scalar, 3, 1> &P3d) const
  {
    Eigen::Matrix<T, ValuesAtCompileTime, 1>& P2d = *_v;
    //std::cout << x << std::endl<<std::endl;
    //if (inputs() != 15)  return;  
    P3d_to_P2d<T>(P3d, P2d, x);
  }

  //手工计算Jacobian的版本，这里用Numerical Method
  void operator() (const InputType& x, ValueType* v, JacobianType* _j, const Eigen::Matrix<Scalar, 3, 1> &P3d) const
  {
    (*this)(x, v, P3d);
 
    if (_j)
    {
      ValueType P2d_plus, P2d_minus;
      JacobianType& j = *_j;
      Scalar epsilon = 1e-6;
      auto x_buffer = x;
      for (int i = 0; i < inputs(); i++)
      {
        auto temp = x(i);
        x_buffer[i] = temp + epsilon;
        P3d_to_P2d<Scalar>(P3d, P2d_plus, x_buffer);

        x_buffer[i] = temp - epsilon;
        P3d_to_P2d<Scalar>(P3d, P2d_minus, x_buffer);

        j.col(i) = (P2d_plus - P2d_minus) / (2 * epsilon);
      }
    }
  }
};

void test_PointProjection_jacobian()
{
  using namespace Eigen;
  using namespace std;

  float tagsize = 26.7;//mm
  Eigen::Matrix<double, 3, Dynamic> ObjectP3d;
  generatePtsInPlane(tagsize, tagsize / 4, tagsize / 4, 8, 6, ObjectP3d);
  //std::cout << ObjectP3d << std::endl;

  //6dof rotation motion
  Vector3d vr; vr << -9.1757138480175165e-02, 3.1935484604661840e-01, 4.3150647234269382e-02;
  Vector3d vt; vt << -1.6446818, -46.849107647, 153.1005;

  //camera parameters
  Vector2d fol; fol << 278.421, 277.217;
  Vector2d uv;  uv << 322.9450, 238.80910;
  Vector4d D; D << 1.9879565994521477e-02, -4.8905763164846051e-02, 4.1867941284700234e-02, -1.2773104992844659e-02;
  Vector4d::Scalar alpha = 0.0;

  for (int i = 0; i < ObjectP3d.cols(); i++)
  {
    Vector2d P2d;
    Vector3d P3d = ObjectP3d.col(i);
    P3d_to_P2d<double>(P3d, P2d, vr, vt, fol, uv, D, alpha);
    //  std::cout << P2d.transpose() << std::endl;
  }

  //定义函数，指定输入输出数据类型和维度
  //15个camera内参外参,输出2个坐标值
  typedef FuctorPointProjection<double, 15, 2> MyFuc;
  MyFuc foo;
  MyFuc::ValueType P2d;
  MyFuc::InputType Pinput;
  MyFuc::JacobianType j_df;

  // MyFuc::JacobianType jref;
  Pinput << fol, uv, alpha, D, vr, vt;
  size_t num_point = ObjectP3d.cols();

  //jacobian of all point over 15 camera parameters
  Eigen::MatrixXd J_dxy_dp;
  J_dxy_dp.resize(2 * num_point, 15);

  for (int i = 0; i < num_point; i++)
  {
    Vector3d P3d = ObjectP3d.col(i);
    //ver1
    foo(Pinput, &P2d, P3d);
    //std::cout << P2d.transpose() << std::endl;

    //ver2: with Numerical Jacobian
    foo(Pinput, &P2d, &j_df, P3d);
    //  cout <<"P2d: "<< P2d.transpose() << endl;
    //  cout << "J_df\n" << j_df.transpose() << endl;

    J_dxy_dp.middleRows(2 * i, 2) = j_df;
  }
  cout << " Jacobian by numeraical:\n" << J_dxy_dp << endl;

#ifdef AD
  //very slow using AutoDiffJacobian
  //定义输出矢量 
  MyFuc::JacobianType j;
  //自动微分jacobian调用
  Eigen::AutoDiffJacobian<MyFuc> autoj(foo);

  //encode all parameters to a vector
  Pinput << vr, vt, fol, uv, D, alpha;
  for (int i = 0; i < num_point; i++)
  {
    Vector3d P3d = ObjectP3d.col(i);
    autoj(Pinput, &P2d, &j, P3d);
    // cout << "P2d: " << P2d.transpose() << endl;
    // cout << "J:\n" << j.transpose() << endl;
    J_dxy_dp.middleRows(2 * i, 2) = j;
  }
  cout << " Jacobian by AutoJacobian:\n" << J_dxy_dp << endl;
#endif //AD
}

// ----------------------------------------------------------------
//fisheye camera calibaration
template<typename T>
void fisheye_calibrate(const Eigen::Matrix<T, 3, Eigen::Dynamic>& ObjectP3d,
  const Eigen::Matrix<T, 2, Eigen::Dynamic>& ImageP2d,
  Eigen::Matrix<T, 9, 1>& IntrinsicParams, /*init/output camera params value,9*1*/
  Eigen::Matrix<T, 6, 1>& ExtrinsicParams  /*init/output pose 6*1*/)
{
  using namespace Eigen;
  using namespace std;

  typedef FuctorPointProjection<T, 15, 2> MyFuc; 

  MyFuc foo;
  MyFuc::ValueType P2d;
  MyFuc::JacobianType j_AD, j_df;

  MyFuc::InputType finalParam, currentParam;
  MyFuc::InputType increParam;

  //自动微分jacobian调用
  Eigen::AutoDiffJacobian<MyFuc> autoj(foo);

  //-------------------------------Initialization
  T alpha_smooth = 0.4, err2d = 0.0001;
  int maxCount = 20;

  finalParam << IntrinsicParams, ExtrinsicParams;

  //-------------------------------Optimization
  int num_point = ObjectP3d.cols();
  //jacobian of all positions over 15 camera parameters
  Eigen::Matrix<T, Dynamic, Dynamic> J_dxy_dp;
  Eigen::Matrix<T, Dynamic, 1> residual;
  J_dxy_dp.resize(2 * num_point, 15);
  residual.resize(2 * num_point, 1);

  for (int iter = 0; iter < maxCount; ++iter)
  {
    std::cout << "calibrate() iter : " << iter << std::endl;
    auto alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, iter + 1.0);
    for (int i = 0; i < num_point; i++)
    {
      Vector3d P3d = ObjectP3d.col(i);
     
#ifdef AD   
      //AutoDiffJacobian
      autoj(finalParam, &P2d, &j_AD, P3d);
      J_dxy_dp.middleRows(2 * i, 2) = j_AD;
#else  
      //Numeraical Jacobian
    
      //不必每次计算Jacobian
      if ( iter % 5 == 0 )
      {
        foo(finalParam, &P2d, &j_df, P3d);
        J_dxy_dp.middleRows(2 * i, 2) = j_df;
      }
      else
        foo(finalParam, &P2d,P3d);
#endif

      //collect residual in all points
      residual.segment(2 * i, 2) = -(P2d - ImageP2d.col(i));
    }
    cout << "residual: " << residual.norm() << endl;
    if (residual.norm() < err2d)
      break;

    // J_dxy_dp * detP = residual
    Eigen::Matrix<T, Dynamic, Dynamic> JJ2, ex3;
    Eigen::Matrix<T, 15, 15> lambda = Eigen::Matrix<T, 15, 15>::Identity()*0.001;
    JJ2 = J_dxy_dp.transpose()*J_dxy_dp + lambda;
    ex3 = J_dxy_dp.transpose()*residual;

    increParam = JJ2.inverse()*ex3;
    //cout << "increParam\n" << increParam.transpose() << endl;

    currentParam = finalParam + alpha_smooth2*increParam;
    //cout << "finalParam\n" << finalParam.transpose()<<endl; 
    //cout << "currentParam\n" << currentParam.transpose() << endl;

    finalParam = currentParam;
  }
  IntrinsicParams = finalParam.head(9);
  ExtrinsicParams = finalParam.tail(6);
}

//单张图片 高斯牛顿法估计鱼眼相机内参外参
void test_fisheye_singleview_calibrate()
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
  TypeCamIntrinsic InitCameraParams, CameraParams;//intrinsic
  TypeCamExtrinsic InitCameraPoses, CameraPoses; //extrinsic

  InitCameraParams << 280,280,320,240,0, 0,0,0,0;
  InitCameraPoses << 0, 0, 0.0001, 0, 0, 100;

  CameraParams = InitCameraParams;
  CameraPoses = InitCameraPoses;

  //-----------------------------calibration: camera parameters and pose estimation
  fisheye_calibrate<double>(ObjectP3d, ImageP2d, CameraParams, CameraPoses);

  cout << ">> ------ init parameters ----------" << endl;
  cout << "Init Camera Params:\n" << InitCameraParams.transpose() << endl;
  cout << "Init Camera Poses:\n" << InitCameraPoses.transpose() << endl;

  cout << ">> ------ optimization result ----------" << endl;
  cout << "Camera Params:\n" << CameraParams.transpose() << endl;
  cout << "Camera Poses:\n" << CameraPoses.transpose() << endl;

  cout << ">> ------ ground truth ----------" << endl; 
  cout << "Real Camera Params:\n" << RealCameraParams.transpose() << endl;
  cout << "Real Camera Poses:\n" << RealtCameraPoses.transpose() << endl;
}

void main()
{  
  // test_PointProjection_jacobian();

  //高斯牛顿法,计算鱼眼相机内参外参，只有单张图片
  test_fisheye_singleview_calibrate();
}