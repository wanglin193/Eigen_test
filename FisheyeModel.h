#include <Eigen/Dense> 
#include <Eigen/Geometry> 
//model used in cv::fisheye, see
//https://docs.opencv.org/3.2.0/db/d58/group__calib3d__fisheye.html
//also called 'Equidistant' pinhole model in others (such as Kalibr)

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
  //if (angle < 1e-8) angle = 1e-8;

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

//create p3d in a plane (Apriltag in a paper)
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

