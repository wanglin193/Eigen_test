#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
// ------------------------------------------------------------------------------------

//����һ����ͨ������,������ʽΪfoo<float>(1,2),��ʡ��<float>,ScalarȱʡΪx/y������
//Scalar������double/float�ȳ�������
//�Զ�΢��ʱScalar����Ϊ����Eigen::AutoDiffScalar<Eigen::MatrixXd>,
//����MatrixXd�����滻��Vector2f��Matrix<float,2,3>��Ԥ��������
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

  //���������Ա��������ĵ���
  typedef Eigen::AutoDiffScalar<Eigen::Vector2f> AD;
  //��ʼ������AD����,ax/ay,������ֵ�Ͷ�Ӧ�ĵ�������,������ʸ����ʾ
  //���ַ���:
  //AD ax(p[0], Eigen::Vector2f::UnitX());
  //AD ay(p[1], Eigen::Vector2f::UnitY());
  Eigen::Matrix<float, 2, 1> direct0, direct1;
  direct0 << 1, 0;
  AD ax(p[0], direct0);
  direct1 << 0, 1;
  AD ay(p[1], direct1);

  //cout << "ax.value()=" << ax.value() << " ax.derivatives()=" << ax.derivatives() << endl;
  //�����Զ�΢�֣����س�Ա����Ϊ����ֵvalue()��΢��ʸ��derivatives()
  AD res = foo(ax, ay);
  cout << "f(px,py) by AD: " << res.value() << "\nf'(px,py): \n" << res.derivatives()<< endl;
}

// ------------------------------------------------------------------------------------

//ʸ����Ϊ�������,�����Ϊ����
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
  //������
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

//�����AutoDiffJacobian����.
//����һ��Functor����Ϊģ����AutoDiffJacobian<Functor>���ض�������Functor
//����AutoDiffJacobian�ж���ļ��������ز�����,������
//1.��������InputType/ValueType/JacobianType
//2.�������ʸ��ά��InputsAtCompileTime/ValuesAtCompileTime
//3.��������ʸ��ά�ȵĽӿ�inputs()
//4.����operator(),������adFunctor������adFunctor(x,y)����ʽ����
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

  //Functor����operator�����������ģ������
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
  //�����������
  adFunctor::InputType in;
  in << 1, 2, 3;
  adFunctor::ValueType out;
  adFunctor::JacobianType j;
  
  //����Functor
  adFunctor f;
  //�������,�����������ʸ��
  f(in, &out);
  std::cout << " out : " << out.transpose() << std::endl;

  //�Զ�Jacobian���÷�����adFunctor��ΪAutoDiffJacobian����������
  Eigen::AutoDiffJacobian<adFunctor> adjac(f);
  adjac(in, &out, &j);
  std::cout << "Output: " << out.transpose() << std::endl;
  std::cout << "Jacobian:\n"<<j << std::endl;
}

// ------------------------------------------------------------------------------------

//��΢���ӵ�TestFunc1������.
//���ص�void operator()�����ֹ�����Jacobian�İ汾
//ʸ��ά�ȸ�������ʸ���Զ�����
//Dynamic����˼�� Eigen::MatrixXd�Ķ���Eigen::Matrix<double, Dynamic, Dynamic>�е�һ�£�
//��ʾ����ά�Ȳ��ڱ���ʱȷ����
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
  //�ֹ�����Jacobian�İ汾�������ý�����
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
  //���庯����ָ����������������ͺ�ά��
  typedef TestFunc1<double, 3, 2> MyFuc; //��дΪ�˷���
  MyFuc f;

  //�����������ʸ��
  MyFuc::InputType x = MyFuc::InputType::Random();
  MyFuc::ValueType yref;
  MyFuc::JacobianType jref;

  //�������,�����������ʸ��,��������Jacobian������
  f(x, &yref, &jref);
  std::cout << ">> Jaobian by hand:\n";
  std::cout <<"Y:"<< yref.transpose() << "\n";
  std::cout <<"Jacobian:\n"<< jref << "\n";

  //�Զ�΢��jacobian����
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