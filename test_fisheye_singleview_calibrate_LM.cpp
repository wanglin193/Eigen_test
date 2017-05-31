#include <iostream>
#include <Eigen/Dense> 
#include <Eigen/Geometry> 

#include <unsupported/Eigen/NonLinearOptimization>
//#include <unsupported/Eigen/LevenbergMarquardt>

//Optimization Fisheye Camera Intrinsic and Extrinsic by Eigen::LevenbergMarquardt<>

//fish eye camera model for imaging 
//调用Autodiff时，T的类型变成AutoDiffScalar<>或 AutoDiffJacobian<>
template<typename T>
void P3d_to_P2d(const Eigen::Matrix<T, 3, 1>& P3d, Eigen::Matrix<T, 2, 1>& P2d,
  const Eigen::Matrix<T, 3, 1>& vr,//rotation
  const Eigen::Matrix<T, 3, 1>& vt,//motion
  const Eigen::Matrix<T, 2, 1>& fol,//focal length
  const Eigen::Matrix<T, 2, 1>& uv,//center 
  const Eigen::Matrix<T, 4, 1>& distor,//distortion
  const T alpha = 0.0)
{
  //se3->SE3 
  Eigen::Transform<T, 3, Eigen::Isometry> transformation;
  T angle = vr.norm();
  //  if (angle < 1e-8) angle = 1e-8;

  transformation = Eigen::AngleAxis<T>(angle, vr / angle);
  transformation.translation() = vt;
  auto RT = transformation.matrix();
  //std::cout << "RT:\n" <<RT<<std::endl; 
  //std::cout << "so3" << vr << vt << std::endl;
  //std::cout << "SE3:\n" << RT << std::endl;

  typedef Eigen::Matrix<T, 2, 1>  VEC2;
  typedef Eigen::Matrix<T, 3, 1>  VEC3;
  typedef Eigen::Matrix<T, 4, 1>  VEC4;

  VEC4 P3dt = RT*P3d.homogeneous();
  VEC2 p2 = VEC2(P3dt[0] / P3dt[2], P3dt[1] / P3dt[2]);
  T r = sqrt(p2.dot(p2));

  if (r < 1e-8)
    r = 1e-8;

  //不是所有数学函数都支持AD，atan(AD)就不支持. see definition in AutoDiffScalar.h
  T theta = atan2(r, T(1.0));
  T theta2 = theta*theta, theta3 = theta2*theta, theta5 = theta3*theta2, theta7 = theta5*theta2, theta9 = theta7*theta2;
  T theta_d = theta + distor[0] * theta3 + distor[1] * theta5 + distor[2] * theta7 + distor[3] * theta9;
  VEC2 xd1 = p2 * theta_d / r;

  VEC2 xd3(xd1[0] + alpha*xd1[1], xd1[1]);
  P2d = VEC2(xd3[0] * fol(0) + uv(0), xd3[1] * fol(1) + uv(1));
}

//version use compact input vector
template<typename T>
void P3d_to_P2d(const Eigen::Matrix<T, 3, 1>& P3d, Eigen::Matrix<T, 2, 1>& P2d, const Eigen::Matrix<T, 15, 1>& x)
{
  typedef Eigen::Matrix<T, 2, 1>  VEC2;
  typedef Eigen::Matrix<T, 3, 1>  VEC3;
  typedef Eigen::Matrix<T, 4, 1>  VEC4;

  VEC2 fol = VEC2(x.segment(0, 2)), uv = VEC2(x.segment(2, 2));
  T alpha = T(x(4));
  VEC4 D = VEC4(x.segment(5, 4));
  VEC3 vr = VEC3(x.segment(9, 3)), vt = VEC3(x.segment(12, 3));

  P3d_to_P2d<T>(P3d, P2d, vr, vt, fol, uv, D, alpha);
}

void generatePtsInPlane(float tagSize, float tagMarginX, float tagMarginY,
  int numx, int numy, Eigen::Matrix<double, 3, Eigen::Dynamic> & objPtsInPlane)
{
  //side length
  float xsize = (float)numx*tagSize + tagMarginX*((float)numx - 1);
  float ysize = (float)numy*tagSize + tagMarginY*((float)numy - 1);
  float offsetx = tagSize + tagMarginX;
  float offsety = tagSize + tagMarginY;
  objPtsInPlane.resize(3, numx*numy * 4);
  int idx = 0;
  for (int y = 0; y < numy; y++)
  {
    for (int x = 0; x < numx; x++)
    {
      float l = 0 + offsetx*x - xsize / 2, u = 0 + offsety*y - ysize / 2;
      float r = l + tagSize, b = u + tagSize;
      objPtsInPlane.col(idx) = Eigen::Vector3d(l, b, 0);  //p0 (-1,1)
      objPtsInPlane.col(idx + 1) = Eigen::Vector3d(r, b, 0);//p1 (1,1)
      objPtsInPlane.col(idx + 2) = Eigen::Vector3d(r, u, 0);//p2 (1,-1)
      objPtsInPlane.col(idx + 3) = Eigen::Vector3d(l, u, 0);//p3 (-1,-1)
      idx += 4;
    }
  }
}

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

  //set dimention of parameters from here
  projective_functor(int num_in, int num_val) : Functor<double>(num_in, num_val) {}

  Eigen::Matrix<double, 3, Eigen::Dynamic>  ObjectP3d;
  Eigen::Matrix<double, 2, Eigen::Dynamic>  ImageP2d;

  int operator() (const InputType& param, ValueType& fvec) const
  {
    //std::cout << "inputs: " << inputs() << std::endl;
   // std::cout << "values: " << values() << std::endl;
    std::cout << "fvec " << fvec.rows() << " * " << fvec.cols() << std::endl;

    Eigen::Matrix<Scalar, 2, 1>  v_;

    //for each 2d point     
    for (int i = 0; i < values() / 2; i++)
    {
      P3d_to_P2d<Scalar>(ObjectP3d.col(i), v_, param);
      fvec[2 * i] = ImageP2d(0, i) - v_(0);
      fvec[2 * i + 1] = ImageP2d(1, i) - v_(1);
    }
    // std::cout << fvec.transpose()<< std::endl;
    std::cout << "done" << std::endl;
    return 0;
  }

  //手工计算Jacobian的版本，这里用Numerical Method
  int df(const InputType& x, JacobianType& fjac) const
  {
    double epsilon = 1e-6;

    Eigen::Matrix<Scalar, 2, -1> J_pt;
    J_pt.resize(2, inputs());
    std::cout << "fjac: " << fjac.rows() << " * " << fjac.cols() << std::endl;

    for (int j = 0; j < fjac.rows() / 2; j++)
    {
      Eigen::Matrix<Scalar, 2, 1> P2d_plus, P2d_minus;
      auto x_buffer = x;
      for (int i = 0; i < inputs(); i++)
      {
        auto temp = x(i);
        x_buffer[i] = temp + epsilon;
        P3d_to_P2d<Scalar>(ObjectP3d.col(j), P2d_plus, x_buffer);

        x_buffer[i] = temp - epsilon;
        P3d_to_P2d<Scalar>(ObjectP3d.col(j), P2d_minus, x_buffer);

        J_pt.col(i) = (P2d_plus - P2d_minus) / (2 * epsilon);
      }
      fjac.row(2 * j) = J_pt.row(0);
      fjac.row(2 * j + 1) = J_pt.row(1);
    }
    //std::cout<< fjac<< std::endl;
    std::cout << "done in J." << std::endl;
    return 0;
  }
};

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
  TypeCamIntrinsic InitCameraParams, CameraParams;//intrinsic
  TypeCamExtrinsic InitCameraPoses, CameraPoses; //extrinsic

  InitCameraParams << 280, 280, 320, 240, 0, 0, 0, 0, 0;
  InitCameraPoses << 0, 0, 0.0001, 0, 0, 100;

  //------------------------------------------------------------------
  Eigen::VectorXd p; p.resize(15, 1);
  p << InitCameraParams, InitCameraPoses;

  projective_functor foo(15, 2 * ImageP2d.cols());
  foo.ObjectP3d = ObjectP3d;
  foo.ImageP2d = ImageP2d;
  Eigen::LevenbergMarquardt<projective_functor> lm(foo);

  //VectorXd residual;  residual.resize(2 * ImageP2d.cols(), 1);
  //foo(p, residual);
 // cout << residual << endl;
  lm.parameters.maxfev = 5;//max iter number

 // lm.parameters.xtol = 0.0001;

  int ret = lm.lmder1(p);
  cout << lm.fvec.blueNorm() << endl;
  cout << p.transpose() << endl;
}

void main()
{

  test_fisheye_singleview_calibrate_LM();
}
