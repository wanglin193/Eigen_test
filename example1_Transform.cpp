#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <iostream> 
using namespace Eigen;
using namespace std;

typedef double Scalar ;
typedef Eigen::Matrix<Scalar,6,1> Vec6; 
typedef Eigen::Matrix<Scalar,4,4> Mat4x4; 
typedef Eigen::Matrix<Scalar,3,1> Vec3;
typedef Eigen::Matrix<Scalar,Dynamic,3> VERT;
	
inline VERT applyRT( const VERT&v, const Mat4x4& RT ) 
{
	return (v.rowwise().homogeneous())*(RT.topRows(3).transpose());
}

void test_rot()
{
	VERT pt(4,3),pt_;
	pt<<0,0,0, 1,0,0, 0,1,0, 0,0,1;
	pt_ = 100*pt;
	
	cout<<"-----------------test_rot()------------------------"<<endl;
	cout<<"P3d:\n"<< pt<<endl;
		
	//1.
	Mat4x4 pose = Mat4x4::Identity();  
	Eigen::Transform<Scalar,3,Affine> rotX,rotY,rotZ;
	rotX = Eigen::AngleAxis<Scalar>( M_PI/2,Vec3(1,0,0) );
	rotY = Eigen::AngleAxis<Scalar>( M_PI/2,Vec3(0,1,0) );
	rotZ = Eigen::AngleAxis<Scalar>( M_PI/2,Vec3(0,0,1) );
	pose = (rotZ*rotY*rotX).matrix(); 
	cout<<"R:\n"<<pose<<endl;
	//3*3 R apply to p3d
	pt_ = (pose.block(0,0,3,3)*pt.transpose()).transpose();
	cout<<"P3d_rot:\n"<< pt_<<endl;
	
	//2.
	rotX = rotY = rotZ = Mat4x4::Identity();
	rotX(1,1)=rotX(2,2)=-1; 
	rotY(0,0)=rotY(2,2)=-1;
	rotZ(0,0)=rotZ(1,1)=-1;
	//R
	pose = (rotY*rotX).matrix();
	//T
	pose.block(0,3,3,1) = Vec3(100,80,-30);
	cout<<"RT:\n"<<pose<<endl;		 
	//4*4 RT apply to p3d
	pt_ = applyRT(pt,pose);
	cout<<"P3d_RT:\n"<< pt_<<endl;
}

void test_transform( Vec6 & TV )
{
	Mat4x4 RT = Mat4x4::Identity(); 
	cout<<"------------------test_transform()------------------"<<endl;
	cout<<"Rigid Transformation Vector:\n" <<TV.transpose()<<endl;

	//Rotation = Rz*Ry*Rx
	Scalar cx,cy,cz,sx,sy,sz;
	cx = cos(TV(0)); cy = cos(TV(1)); cz = cos(TV(2)); 
	sx = sin(TV(0)); sy = sin(TV(1)); sz = sin(TV(2)); 
	RT(0,0) = cy*cz; RT(0,1) = cz*sx*sy-cx*sz;	RT(0,2) = cx*cz*sy+sx*sz; RT(0,3) = TV(3);
	RT(1,0) = cy*sz; RT(1,1) = cx*cz+sx*sy*sz ; RT(1,2) = cx*sy*sz-cz*sx; RT(1,3) = TV(4);
	RT(2,0) = -sy;	 RT(2,1) = cy*sx ;			RT(2,2) = cx*cy ;		  RT(2,3) = TV(5);
	cout<<"\nEuler angles->Rotation matrix:\n";
	cout<<"RT Euler Rz*Ry*Rx:\n"<<RT<<endl;

	//Rotation = Rz*Ry*Rx
	Eigen::Transform<Scalar,3,Affine> trEuler;//Eigen::Transform<Scalar,3,Affine> = Eigen::Affine3f
	trEuler  = 
		Eigen::AngleAxis<Scalar>(TV(2), Vec3::UnitZ()) * // Eigen::Vector3f(0, 0, 1)) 
		Eigen::AngleAxis<Scalar>(TV(1), Vec3::UnitY()) * // Eigen::Vector3f(0, 1, 0)) 
		Eigen::AngleAxis<Scalar>(TV(0), Vec3::UnitX()) ; // Eigen::Vector3f(1, 0, 0)) 
	trEuler.translation() = TV.tail<3>(); 		
	RT = trEuler.matrix();		
	cout<<"RT Euler Rz*Ry*Rx with EIGEN :\n"<<RT<<endl;

	//Rotation = Rx*Ry*Rz
	trEuler  = 
		Eigen::AngleAxis<Scalar>(TV(0), Vec3::UnitX()) * 
		Eigen::AngleAxis<Scalar>(TV(1), Vec3::UnitY()) * 
		Eigen::AngleAxis<Scalar>(TV(2), Vec3::UnitZ()) ; 
	trEuler.translation() = TV.tail<3>(); 		
	RT = trEuler.matrix();		
	cout<<"RT Euler Rx*Ry*Rz with EIGEN :\n"<<RT<<endl;

	//Rodrigues equation:	
	Vec3 Axis =  TV.head<3>(); 	
	Scalar Angle = Axis.norm(); 
	Axis = Axis/Angle; 
	cout << "\nRodrigues equation:\n";
	cout << "Angle: " << Angle << endl; 
	cout << "Axis: " << Axis << endl; 

	//so(3)->SO(3),angle-axis->Rotation matrix
	Eigen::Transform<Scalar,3,Affine> transformation;
	transformation = Eigen::AngleAxis<Scalar>(Angle,Axis);		
	transformation.translation() = TV.tail<3>(); 
	RT = transformation.matrix();
	cout << "so(3)->SO(3):\n";
	cout<<"RT:\n"<<RT<<endl;

	//SO(3)->so(3),Rotation matrix -> angle-axis 
	Eigen::Matrix<Scalar,3,3> R = transformation.rotation();
	cout<<"R:\n"<<R<<endl; 

	Eigen::AngleAxis<Scalar> angle_test(R); //angle_test.matrix()
	Scalar ang = angle_test.angle();
	Vec3  axis = angle_test.axis(); 
	cout << "SO(3)->so(3):\n";
	cout << "Angle: " << ang <<  endl; 
	cout << "Axis: " << axis <<  endl;  

	//Quaternion:
	Eigen::Quaternion<Scalar> p,q,rotatedP; 
	q = Eigen::AngleAxis<Scalar>(Angle,Axis); 
	//q.normalize();
	cout<<"Quaternion scalar:"<<q.w()<<"\nQuaternion vector:\n"<<q.vec()<<endl ; 
	cout<<"R by Quaternion:\n"<<q.toRotationMatrix()<<endl; 

	//R From Two Vectors:
	Vec3 v0(7,2,8);
	Vec3 v1 = q*v0 ;
	rotatedP = Eigen::Quaternion<Scalar>::FromTwoVectors( v0, v1 );
	cout<<"R from 2 vectors :\n"<<rotatedP.matrix()<<endl; 
	return ;
}

/*
p = Eigen::AngleAxis<Scalar>(Angle,Axis); 
q = Eigen::AngleAxis<Scalar>(Angle,Axis); 
rotatedP = q * p * q.inverse();
cout<<"rotatedP :\n"<<rotatedP.toRotationMatrix()<<endl; 
cout<<"rotatedP :\n"<<p.matrix()*q.matrix()<<endl; 
*/

//Compile:
//set INCLUDE=C:\eigen3.2.8\;%INCLUDE%
//cl example1_Transform.cpp /MD /Ox /Ot /W3 /EHsc
void main()
{
	test_rot();
	 
	Vec6 vect1;

	vect1<< 0.0, M_PI/4, -M_PI/8, 100.19, 40.74,20.193;
	test_transform( vect1 );
	
	Scalar raw_data[6] ={0.6,0.3,-0.6, 100.19, 40.74,20.193};
	Map<Vec6> vect2(raw_data);
	test_transform( Vec6(vect2) ); 
	test_transform( Vec6(vect2/=100) );  

  //raw data has changed
	printf("raw_data r: %g %g %g\n",raw_data[0],raw_data[1],raw_data[2]);
	printf("raw_data t: %g %g %g\n",raw_data[3],raw_data[4],raw_data[5]);

	return;
}
