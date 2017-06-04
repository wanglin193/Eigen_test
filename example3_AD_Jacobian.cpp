#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
// ------------------------------------------------------------------------------------

//定义一个普通函数体,调用形式为foo<float>(1,2),如省略<float>,Scalar缺省为x/y的类型
//Scalar可以是double/float等常规类型
//自动微分时Scalar定义为矩阵Eigen::AutoDiffScalar<Eigen::MatrixXd>,
//其中MatrixXd可以替换成Vector2f或Matrix<float,2,3>等预定义类型
template<typename Scalar> 
Scalar foo(const Scalar& x, const Scalar& y)
{ 
 Scalar f = x*x + sin(y);
 //Scalar f = (x * 2 - pow(x, 2) + 2 * sqrt(y) - 4 * sin(x) + 2 * cos(y) - exp(-0.5*x*x));
 return f;
}
 
void test_autodiff_scalar ()
{
  using namespace std;

  double p[2] = {1.3,M_PI};
  double val = foo(p[0], p[1]);
  cout << "f(px,py)=" << val << endl;

  //定义两个自变量函数的导数
  typedef Eigen::AutoDiffScalar<Eigen::Vector2f> AD;
  //初始化两个AD变量,ax/ay,给定初值和对应的导数方向,方向用矢量表示
  //两种方法:
  //AD ax(p[0], Eigen::Vector2f::UnitX());
  //AD ay(p[1], Eigen::Vector2f::UnitY());
  Eigen::Matrix<float, 2, 1> direct0, direct1;
  direct0 << 1, 0;
  AD ax(p[0], direct0);
  direct1 << 0, 1;
  AD ay(p[1], direct1);

  //cout << "ax.value()=" << ax.value() << " ax.derivatives()=" << ax.derivatives() << endl;
  //计算自动微分，返回成员变量为函数值value()和微分矢量derivatives()
  AD res = foo(ax, ay);
  cout << "f(px,py) by AD: " << res.value() << "\nf'(px,py): \n" << res.derivatives()<< endl;
}

// ------------------------------------------------------------------------------------

//矢量作为输入变量,输出仍为标量
template<typename Vector> 
typename Vector::Scalar foo(const Vector& p)
{
  typedef typename Vector::Scalar Scalar;
  Scalar f;
  if (p.rows() == 2 && p.cols() == 1)//2d vector
    f = p(0)*p(0) + sin(p[1]);
  else
   f =(p - Vector(Scalar(-1), Scalar(1.))).norm() + (p.array() * p.array()).sum() + p.dot(p);
  return f;
}

void test_autodiff_vector()
{
  using namespace std;

  Eigen::Vector2f p;// = Eigen::Vector2f::Random();
  p << 1.3, M_PI;
  cout<<"f(p)=" << foo(p) << endl;
  //不好用
  typedef Eigen::AutoDiffScalar<Eigen::Vector2f> AD;
  typedef Eigen::Matrix<AD, 2, 1> VectorAD;
 
  VectorAD ap = p.cast<AD>();
  ap.x().derivatives() = Eigen::Vector2f::UnitX();
  ap.y().derivatives() = Eigen::Vector2f::UnitY();
  AD res = foo<VectorAD>(ap);

  auto v = foo<Eigen::Vector2f>(p);
  cout << "f(p) by AD=" << res.value() << "\nf'(p)=\n" << res.derivatives() << endl;
}

// ------------------------------------------------------------------------------------

//极简的AutoDiffJacobian例子.
//定义一个Functor，作为模板类AutoDiffJacobian<Functor>的特定化类型Functor
//所以AutoDiffJacobian中定义的几个变量必不可少,包括：
//1.矩阵类型InputType/ValueType/JacobianType
//2.输入输出矢量维度InputsAtCompileTime/ValuesAtCompileTime
//3.返回输入矢量维度的接口inputs()
//4.重载operator(),这样类adFunctor可以用adFunctor(x,y)的形式调用
struct adFunctor
{
  typedef Eigen::Matrix<double, 3, 1> InputType;
  typedef Eigen::Matrix<double, 2, 1> ValueType;
  typedef Eigen::Matrix<double, 2, 3> JacobianType;

  enum {
    InputsAtCompileTime = 3,
    ValuesAtCompileTime = 2
  };
  
  size_t inputs() const { return InputsAtCompileTime; }
  //size_t outputs() const { return ValuesAtCompileTime; }

  //Functor都有operator，这里必须是模板类型
  template <typename T>
  void operator() (const Eigen::Matrix<T, InputsAtCompileTime, 1>& input, Eigen::Matrix<T, ValuesAtCompileTime, 1>* _v) const
  {    
    Eigen::Matrix<T, ValuesAtCompileTime, 1>& output = *_v;
    output.setZero();
    for (int i = 0; i < inputs(); i++)
    {
      output[0] += log(input(i));
      output[1] += sqrt(input(i));
    } 
  }
}; 

void test_adFunctor_jacobian()
{
  //输入输出类型
  adFunctor::InputType in;
  in << 1, 2, 3;
  adFunctor::ValueType out;
  adFunctor::JacobianType j;
  
  //定义Functor
  adFunctor f;
  //常规调用,输入输出都是矢量
  f(in, &out);
  std::cout << " out : " << out.transpose() << std::endl;

  //自动Jacobian调用方法，adFunctor作为AutoDiffJacobian的类型输入
  Eigen::AutoDiffJacobian<adFunctor> adjac(f);
  adjac(in, &out, &j);
  std::cout << "Output: " << out.transpose() << std::endl;
  std::cout << "Jacobian:\n"<<j << std::endl;
}

// ------------------------------------------------------------------------------------

//稍微复杂的TestFunc1的例子.
//重载的void operator()包含手工计算Jacobian的版本
//矢量维度根据输入矢量自动决定
//Dynamic的意思和 Eigen::MatrixXd的定义Eigen::Matrix<double, Dynamic, Dynamic>中的一致，
//表示矩阵维度不在编译时确定。
template<typename Scalar, int NX = Dynamic, int NY = Dynamic>
struct TestFunc1
{ 
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };

  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  int m_inputs, m_values;

  TestFunc1() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  TestFunc1(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  template<typename T>
  void operator() (const Eigen::Matrix<T, InputsAtCompileTime, 1>& x, Eigen::Matrix<T, ValuesAtCompileTime, 1>* _v) const
  {
    Eigen::Matrix<T, ValuesAtCompileTime, 1>& v = *_v;

    v[0] = 2 * x[0] * x[0] + x[0] * x[1];
    v[1] = 3 * x[1] * x[0] + 0.5 * x[1] * x[1];
    if (inputs()>2)
    {
      v[0] += 0.5 * (x[2]);
      v[1] += x[2];
    }
    if (values()>2)
    {
      v[2] = 3 * x[1] * x[0] * x[0];
    }
    if (inputs()>2 && values()>2)
      v[2] *= x[2];
  }
  //手工计算Jacobian的版本，这里用解析法
  void operator() (const InputType& x, ValueType* v, JacobianType* _j) const
  {
    (*this)(x, v);

    if (_j)
    {
      JacobianType& j = *_j;

      j(0, 0) = 4 * x[0] + x[1];
      j(1, 0) = 3 * x[1];

      j(0, 1) = x[0];
      j(1, 1) = 3 * x[0] + 2 * 0.5 * x[1];

      if (inputs()>2)
      {
        j(0, 2) = 0.5;
        j(1, 2) = 1;
      }
      if (values()>2)
      {
        j(2, 0) = 3 * x[1] * 2 * x[0];
        j(2, 1) = 3 * x[0] * x[0];
      }
      if (inputs()>2 && values()>2)
      {
        j(2, 0) *= x[2];
        j(2, 1) *= x[2];

        j(2, 2) = 3 * x[1] * x[0] * x[0];
        j(2, 2) = 3 * x[1] * x[0] * x[0];
      }
    }
  }
};

//test auto diff jacobian
void test_autodiff_jacobian()
{ 
  //定义函数，指定输入输出数据类型和维度
  typedef TestFunc1<double, 3, 2> MyFuc; //简写为了方便
  MyFuc f;

  //定义输入输出矢量
  MyFuc::InputType x = MyFuc::InputType::Random();
  MyFuc::ValueType yref;
  MyFuc::JacobianType jref;

  //常规调用,输入输出都是矢量,调用手算Jacobian矩阵函数
  f(x, &yref, &jref);
  std::cout << ">> Jaobian by hand:\n";
  std::cout <<"Y:"<< yref.transpose() << "\n";
  std::cout <<"Jacobian:\n"<< jref << "\n";

  //自动微分jacobian调用
  MyFuc::ValueType y;
  MyFuc::JacobianType j;
  Eigen::AutoDiffJacobian<MyFuc> autoj(f);
  autoj(x, &y, &j);
  std::cout << ">> Jaobian by AutoDiffJacobian:\n";
  std::cout << "Y:" << y.transpose() << "\n";
  std::cout << "Jacobian:\n" << j << "\n";
} 
// ------------------------------------------------------------------------------------

void main()
{ 
  std::cout << "\n>>>>>>>>test_autodiff_scalar:" << std::endl;
  test_autodiff_scalar();

  std::cout << "\n>>>>>>>>test_autodiff_vector:" << std::endl; 
  test_autodiff_vector();

  std::cout << "\n>>>>>>>>test_adFunctor_jacobian:" << std::endl; 
  test_adFunctor_jacobian();

  std::cout << "\n>>>>>>>>test_autodiff_jacobian:" << std::endl;
  test_autodiff_jacobian();
} 