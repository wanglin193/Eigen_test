//created and tested by Wang Lin
//wanglin193 at gmail.com

#include <iostream>
#include <Eigen/Dense> 
#include <Eigen/Geometry> 
#include <unsupported/Eigen/NonLinearOptimization> 
#include "FisheyeModel.h"

//Optimization Fisheye Camera Intrinsic and Extrinsic by Eigen::LevenbergMarquardt<>
//Multi-view(images)
//9 + 6 * num_view parameters to be optimized
//each image has different projective 2d points.
//每个图像上的点隔点采集。模拟Sparse Bundle Adjustment

typedef Eigen::Matrix<double, 9, 1> TypeCamIntrinsic;
typedef Eigen::Matrix<double, 6, 1> TypeCamExtrinsic;
 
//Generic functor
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
  std::vector<std::vector<int> > v_IdxP2d;
  std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic> >  v_ImageP2d;

  //params:优化变量
  //fvec:残差矢量，最小二乘的优化函数目标是fvec.norm()最小

  int operator() (const InputType& params, ValueType& fvec) const
  {
    //parsing all params into 9+6*nview
    TypeCamIntrinsic CameraIntrincs;
    std::vector<TypeCamExtrinsic> CameraPoses;

    CameraIntrincs = params.head(9);
    for (int n = 0; n < v_IdxP2d.size(); n++)
    {
      TypeCamExtrinsic pose = params.segment(9 + n * 6, 6);
      CameraPoses.push_back(pose);
    }

    //for each image
    int j = 0;
    for (int n = 0; n < v_IdxP2d.size(); n++)
    {
      Eigen::Matrix<double, 15, 1> params_this_view;
      params_this_view << CameraIntrincs, CameraPoses[n];
      //std::cout << params_this_view.transpose() << std::endl;

      //for 2d points in each image     
      for (int i = 0; i < v_IdxP2d[n].size(); i++)
      {
        int id_p3d = v_IdxP2d[n][i];
        Eigen::Vector2d P2d = v_ImageP2d[n].col(i);
        Eigen::Vector3d P3d = ObjectP3d.col(id_p3d);
        Eigen::Vector2d v;

        P3d_to_P2d<Scalar>(P3d, v, params_this_view);
        fvec[j] = P2d(0) - v(0);
        fvec[j + 1] = P2d(1) - v(1);
        j += 2;
      }
    }
    //if ( j != values()) std::cout << " err" << std::endl;  
    return 0;
  }

  //名字必须df,手工计算Jacobian的版本，这里用Numerical Method
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

  void print_info()
  {
    for (int n = 0; n < v_IdxP2d.size(); n++)
    {
      std::cout << "num in this image: " << v_IdxP2d[n].size() << std::endl;;

      for (int i = 0; i < v_IdxP2d[n].size(); i++)
        std::cout << v_IdxP2d[n][i] << "|";
      std::cout << std::endl;

      std::cout << v_ImageP2d[n] << std::endl;
    }
  }
};

//fake ground truth
void generate_image_projective_points(Eigen::Matrix<double, 3, Eigen::Dynamic>& ObjectP3d,
  std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic> >& v_ImageP2d,
  std::vector<std::vector<int> >& v_IdxPt)
{
  using namespace std;
  //-------------------------------generate ground truth
  float tagsize = 26.7;//mm
 // Eigen::Matrix<double, 3, Eigen::Dynamic> ObjectP3d;
  generatePtsInPlane(tagsize, tagsize / 4, tagsize / 4, 8, 6, ObjectP3d);

  //Intrinsic: 9 camera parameters
  TypeCamIntrinsic RealCameraParams;
  RealCameraParams << 278.421, 277.217, 322.9450, 238.80910, 0.001,
    1.9879565994521477e-02, -4.8905763164846051e-02, 4.1867941284700234e-02, -1.2773104992844659e-02;

  //Extrinsic: 6-dof rotation and motion
  const int num_view = 8;
  TypeCamExtrinsic RealCameraPoses[num_view];
  RealCameraPoses[0] << -9.1757138480175165e-02, 3.1935484604661840e-01, 4.3150647234269382e-02, -1.6446818, -46.849107647, 153.1005;
  RealCameraPoses[1] << -8.9357572145657482e-02, 3.1514268943015450e-01, 3.1397392813540391e-02, -2.2794429082176455e+00, -3.9690784348848602e+01, 1.3594802686048018e+02;
  RealCameraPoses[2] << -8.4582723905902088e-02, 3.1958160871304908e-01, 3.6379528242504018e-02, -5.6343640863438580e+00, -3.6482256867944692e+01, 1.3820080783279479e+02;
  RealCameraPoses[3] << -9.1723576790968445e-02, 3.2333768784695655e-01, 3.5596461594359127e-02, -5.9738048287144121e+00, -3.9315940139563047e+01, 1.3848955105001900e+02;
  RealCameraPoses[4] << -9.5299400762696398e-02, 3.2600856773223291e-01, 4.2293639756994184e-02, -5.4833977982790829e+00, -3.9973987238308567e+01, 1.3746679864274185e+02;
  RealCameraPoses[5] << -5.2518669896608325e-03, 1.8729378068993427e-01, 2.3869010745811567e-01, 7.56063672, -38.9056610001283, 135.9400025953;
  RealCameraPoses[6] << 6.8232533405613333e-02, 1.7053718702560988e-01, 3.8316770039900216e-01, -14.956344793, -39.12189694169, 139.41003269007;
  RealCameraPoses[7] << -5.4208352654412867e-03, 5.5297615513488119e-01, 1.3804937029128120e-01, 16.791123587, -23.811561374564, 135.9976538618;

  //gernerate image points,隔点采集2D投影，模拟sparse BA
  int inc[num_view] = { 1,2,1,2, };

  //每张图像的3D点序号,sparse BA
 // std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic> > v_ImageP2d;
 // std::vector<std::vector<int> > v_IdxPt;
  for (int n = 0; n < num_view; n++)
  {
    //put all params to a vector
    Eigen::VectorXd params(15);
    params << RealCameraParams, RealCameraPoses[n];

    //fill index
    std::vector<int> id_in_this_image;
    if (inc[n] == 0) inc[n] = 1;
    for (int i = 0; i < ObjectP3d.cols(); i += inc[n])
    {
      id_in_this_image.push_back(i);
    }
    v_IdxPt.push_back(id_in_this_image);

    //fill image pnts
    int num_pt_in_this_image = id_in_this_image.size();
    Eigen::Matrix<double, 2, Eigen::Dynamic> ImageP2d;
    ImageP2d.resize(2, num_pt_in_this_image);
    for (int i = 0; i < num_pt_in_this_image; i++)
    {
      int id = id_in_this_image[i];
      Eigen::Vector2d P2d;
      P3d_to_P2d<double>(ObjectP3d.col(id), P2d, params);
      ImageP2d.col(i) = P2d;
    }
    v_ImageP2d.push_back(ImageP2d);
  }

  ///------------------------------------------------------------------
  cout << ">>>>>>>>>>>>>>>>>  Ground truth :\n";
  cout << "Intrinsic of cam: \n" << RealCameraParams.transpose() << endl;
  for (int n = 0; n < v_IdxPt.size(); n++)
    cout << "Pose of cam " << n << " :\n" << RealCameraPoses[n].transpose() << endl;
  cout << "\n";

  /*for (int n = 0; n < num_view; n++)
  {
  cout << "num in this image: "<< v_IdxPt[n].size()<<endl;

  for (int i = 0; i < v_IdxPt[n].size(); i++)
  std::cout << v_IdxPt[n][i] << "|";
  cout << std::endl;

  std::cout << v_ImageP2d[n] << std::endl;
  }*/

  //cout << ImageP2d << endl;
}

#include<chrono>
//multiview images LM法估计鱼眼相机内参外参
void test_fisheye_multiview_calibrate_LM()
{
  using namespace Eigen;
  using namespace std;

  Eigen::Matrix<double, 3, Eigen::Dynamic> ObjectP3d;
  std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic> >  v_ImageP2d;
  std::vector<std::vector<int> >  v_IdxPt;

  generate_image_projective_points(ObjectP3d, v_ImageP2d, v_IdxPt);
  int num_view = v_IdxPt.size();

  //--------------------------------init camera params and poses
  TypeCamIntrinsic InitCameraParams;//intrinsic
  TypeCamExtrinsic InitCameraPoses; //extrinsic

  InitCameraParams << 280, 280, 320, 240, 0, 0, 0, 0, 0;
  InitCameraPoses << 0, 0, 0.0001, 0, 0, 100;
  
  //------------------------------------------------------------------

  //all parameters need to campact to a vector
  int num_params = 9 + 6 * num_view;
  Eigen::VectorXd p(num_params);
  p << InitCameraParams, InitCameraPoses.replicate(num_view, 1);
  //cout << p.transpose() << endl;

  int num_all_image_points = 0;
  for (int n = 0; n < num_view; n++)
    num_all_image_points += v_IdxPt[n].size();
 
  cout << "\n>>>>>>>>>>>>>>>>> Calibration:\n";
 
  //输出是图像投影点的2倍(x,y两坐标)
  projective_functor foo(num_params, 2 * num_all_image_points);
  cout << "Size of Jacobian : " << 2 * num_all_image_points << " x " << num_params << endl;
  
  //设置3D-2D对应数据
  foo.ObjectP3d = ObjectP3d;
  foo.v_ImageP2d = v_ImageP2d;
  foo.v_IdxP2d = v_IdxPt;

  //test functor
  if (1)
  {
    VectorXd residual(2 * num_all_image_points);
    foo(p, residual);
    cout << "Init Energy: " << residual.norm() << endl;
  }

  auto eigen_begin = std::chrono::steady_clock::now();

  //试着切换这两种方法，自动方法比较快；有时候后面手动计算Jacobian反而可以收敛
#if 0
  //auto cal numerical diff 不调用自定义df
  Eigen::NumericalDiff<projective_functor> numdiff(foo);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<projective_functor>, double> lm(numdiff);
#else
  //calculate df by hand,调用自定义df
  Eigen::LevenbergMarquardt<projective_functor> lm(foo);
#endif

  lm.parameters.maxfev = 200;//max iter number
 //lm.parameters.gtol = 0.01;
 //lm.parameters.xtol = 0.1;

  //do computation
  int ret = lm.minimize(p);

  auto eigen_end = std::chrono::steady_clock::now();
  auto eigen_time = std::chrono::duration_cast<std::chrono::duration<double>>(eigen_end - eigen_begin).count();

  std::cout << "time (ms): " << 1000 * eigen_time << std::endl;

  //http://eigen.tuxfamily.org/dox/unsupported/group__NonLinearOptimization__Module.html

  cout << "number of iterations performed: " << lm.iter << endl;
  cout << "number of functions evaluation: " << lm.nfev << endl;
  cout << "number of jacobian evaluation: " << lm.njev << endl;
  // cout << "Result parameters: " << p.transpose() << endl;

  //test functor
  if (1)
  {
    VectorXd residual(2 * num_all_image_points);
    foo(p, residual);
    cout << "Final Energy: " << residual.norm() << endl;
  }
 
  cout << "\n>>>>>>>>>>>>>>>>> Result parameters:\n";
  TypeCamIntrinsic CameraIntrincs = p.head(9);
  cout << "Intrinsic of cam: \n" << CameraIntrincs.transpose() << endl;
  for (int n = 0; n < v_IdxPt.size(); n++)
  {
    TypeCamExtrinsic pose = p.segment(9 + n * 6, 6);
    cout << "Pose of cam " << n << " :\n" << pose.transpose() << endl;
  }
}

void main()
{ 
  test_fisheye_multiview_calibrate_LM();
}