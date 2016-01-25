#include <iostream>
#include <math.h>
#include <dmtk/dmtk.h>
#include <dmtk/tensor.h>

using namespace std;
using namespace dmtk;

//-------------------------------------------------------------

inline Matrix<double> 
eye(int n)
{
    Matrix<double> Identity(n,n);
    Identity = I<double>();
    return Identity;
}

//----------------------------------------------------------

void print_blocks(int l,int r)
{
    cout << "**********************************"<< endl;
    cout << "LEFT SIZE = " << l << endl;
    cout << "RIGHT SIZE = " << r << endl;
    while (l--)
        cout << "X ";
    cout << "* * ";
    while (r--)
        cout << "X ";
    cout << endl;
    return;
}

//----------------------------------------------------------
class DMRGBlock
{
    public:
        int size;
        Matrix<double> H; // block Hamiltonian
        Vector<Matrix<double> > sz; // Sz operators 
        Vector<Matrix<double> > splus; // S+ operators

        Matrix<double> U; // transformation matrix

        DMRGBlock() { resize(1); }
        DMRGBlock(int _size) { resize(_size); }

        DMRGBlock &resize(int n)
          {
            size = n;
            sz.resize(n);
            splus.resize(n);
            return *this;
          }
};

class DMRGSystem
{
    public:
        Matrix<double> sz0; // single site Sz
        Matrix<double> splus0; // single site S+
        Vector<DMRGBlock> BlockL; // left block 
        Vector<DMRGBlock> BlockR; // right block 

        Matrix<double> psi;	// g.s. wave function
        Matrix<double> rho;	// density matrix
        Matrix<double> seed;	// seed wave function for lanczos
        double energy;
        double error;
        int nsites;
        int right_size;
        int left_size;
        
        DMRGSystem(int _nsites)
          {
            sz0.resize(2,2);
            splus0.resize(2,2);
            nsites = _nsites;
            BlockL.resize(nsites);
            BlockR.resize(nsites);

            // initialize Hamiltonian:
            sz0 = 0.0;
            splus0 = 0.0;    
            BlockL[0].H = sz0;    
            BlockR[0].H = sz0;    
            BlockL[0].U = sz0;    
            BlockR[0].U = sz0;    
            // single-site operators
            sz0(0,0)         = -0.5;
            sz0(1,1)         =  0.5;
            splus0(1,0)      =  1.0;
            BlockL[0].sz[0] = sz0;
            BlockR[0].sz[0] = sz0;
            BlockL[0].splus[0] = splus0;
            BlockR[0].splus[0] = splus0;
          }

        void BuildBlockLeft(int _iter);        
        void BuildBlockRight(int _iter);        
        void GroundState(bool use_seed);
        void DensityMatrix(int _position);        
        void Truncate(int _position, int _m);        
        void BuildSeed(int direction);
        void Measure();
        void Reflect()
          {
            BlockR[right_size].resize(left_size+1);
            for(int i = 0; i <= right_size; i++){
              BlockR[right_size].sz[i] = BlockL[left_size].sz[i];
              BlockR[right_size].splus[i] = BlockL[left_size].splus[i];
            }
          }
};

void
DMRGSystem::BuildBlockLeft(int _iter)
{
    left_size = _iter;
    BlockL[left_size].resize(left_size+1);
    Matrix<double> HL = BlockL[left_size-1].H;
    Matrix<double> szL = BlockL[left_size-1].sz[left_size-1];
    Matrix<double> splusL = BlockL[left_size-1].splus[left_size-1];
    int dim_l = HL.cols();
    Matrix<double> I_left = eye(dim_l);
    Matrix<double> I2 = eye(2);
    //enlarge left block:
    BlockL[left_size].H  = tensor(HL,I2) +
                     tensor(szL,sz0) +
                     0.5*tensor(splusL,splus0.t()) +
                     0.5*tensor(splusL.t(),splus0);


    for(int i = 0; i < left_size; i++){
      BlockL[left_size].splus[i] = tensor(BlockL[left_size-1].splus[i],I2);
      BlockL[left_size].sz[i] = tensor(BlockL[left_size-1].sz[i],I2);
    }

    BlockL[left_size].splus[left_size] = tensor(I_left,splus0);
    BlockL[left_size].sz[left_size] = tensor(I_left,sz0);
}

void
DMRGSystem::BuildBlockRight(int _iter)
{
    right_size = _iter;
    BlockR[right_size].resize(right_size+1);
    Matrix<double> HR = BlockR[right_size-1].H;
    Matrix<double> szR = BlockR[right_size-1].sz[right_size-1];
    Matrix<double> splusR = BlockR[right_size-1].splus[right_size-1];
    int dim_r = HR.cols();
    Matrix<double> I_right= eye(dim_r);
    Matrix<double> I2 = eye(2);
    //enlarge right block:
    BlockR[right_size].H = tensor(I2,HR) +
                     tensor(sz0,szR) +
                     0.5* tensor(splus0.t(),splusR) +
                     0.5* tensor(splus0,splusR.t());

    for(int i = 0; i < right_size; i++){
      BlockR[right_size].splus[i] = tensor(I2,BlockR[right_size-1].splus[i]);
      BlockR[right_size].sz[i] = tensor(I2,BlockR[right_size-1].sz[i]);
    }

    BlockR[right_size].splus[right_size] = tensor(splus0,I_right);
    BlockR[right_size].sz[right_size] = tensor(sz0,I_right) ;
}

void
DMRGSystem::GroundState(bool use_seed = false)
{
    double ev;
    Vector<double> lanczos_a(100);
    Vector<double> lanczos_b(100);
    int dim_l = BlockL[left_size].H.cols();
    int dim_r = BlockR[right_size].H.cols();
    psi.resize(dim_r,dim_l);
    int maxiter = -1;

//    cout << psi.rows() << " " << psi.cols() << endl;
//    cout << seed.rows() << " " << seed.cols() << endl;
    lanczos<double, DMRGSystem, Matrix<double> >(*this, psi, seed, ev, lanczos_a, lanczos_b, maxiter, 1.e-7, use_seed, true, "vectors.dat");

    energy = ev;
}

void
DMRGSystem::DensityMatrix(int _position)
{
    int dim_l = BlockL[left_size].H.cols();
    int dim_r = BlockR[right_size].H.cols();
    // Calculate density matrix
    if(_position == LEFT){
      rho = product(psi,psi.t());
    } else {
      rho = product(psi.t(),psi);
    }
}

void
DMRGSystem::Truncate(int _position, int _m)
{
    // diagonalize rho
    Vector<double> rho_eig(rho.cols());
    Matrix<double> rho_evec(rho);
    rho_evec.diagonalize(rho_eig); 
    // calculate the truncation error for a given number of states m
    for(int i = 0; i < rho.cols(); i++) cout << "RHO EIGENVALUE " << i << " = " << rho_eig(i) << endl; 
    error = 0.;
    if (_m < rho_eig.size())
      for (int i = 0; i < rho_eig.size()-_m ; i++) error += rho_eig(i);
    cout << "Truncation error = " << error <<endl;

    if (rho.cols() > _m)
      rho_evec = rho_evec(Range(rho.cols()-_m,rho.cols()-1,1),Range(0,rho.rows()-1));

    // perform transformation:
    Matrix<double> U = rho_evec.t();
    Matrix<double> aux2;
    if(_position == LEFT){
      BlockL[left_size].U = rho_evec;
      aux2 = product(BlockL[left_size].H,rho_evec);
      BlockL[left_size].H = product(U,aux2);

      for(int i = 0; i <= left_size; i++){
        aux2 = product(BlockL[left_size].splus[i],rho_evec);
        BlockL[left_size].splus[i] = product(U,aux2);
        aux2 = product(BlockL[left_size].sz[i],rho_evec);
        BlockL[left_size].sz[i] = product(U,aux2);
      }
    } else {
      BlockR[right_size].U = rho_evec;
      aux2 = product(BlockR[right_size].H,rho_evec);
      BlockR[right_size].H = product(U,aux2);

      for(int i = 0; i <= right_size; i++){
        aux2 = product(BlockR[right_size].splus[i],rho_evec);
        BlockR[right_size].splus[i] = product(U,aux2);
        aux2 = product(BlockR[right_size].sz[i],rho_evec);
        BlockR[right_size].sz[i] = product(U,aux2);
      }
    }
}

//----------------------------------------------------------

void
DMRGSystem::BuildSeed(int dir) 
{
  int dim_l = BlockL[left_size-1].H.cols();
  int dim_r = BlockR[right_size-1].H.cols();
  Tensor<double> psi_new(dim_l,2,2,dim_r);

  if(dir == LEFT2RIGHT){
    if(left_size == 1) {
      seed = psi;
      return;
    } else {
      const Matrix<double> &UL = BlockL[left_size-1].U;  
      const Matrix<double> &UR = BlockR[right_size].U;  
      int old_dim_l = BlockL[left_size-2].U.cols();
      int old_dim_r = BlockR[right_size].U.cols();
// We copy the old g.s. (in matrix form) into the tensor
      Tensor<double> psi_old(old_dim_l*2,1,1,2*old_dim_r); 
      for(int i = 0; i < 2*old_dim_l; i++)
        for(int j = 0; j < 2*old_dim_r; j++) psi_old(i,0,0,j) = psi(j,i);
// We first transform the left part
      psi_old.resize(old_dim_l*2,1,2,old_dim_r);
      Tensor<double> psi_aux(dim_l,1,2,old_dim_r);
      for(int i3 = 0; i3 < 2; i3++){    
        for(int i4 = 0; i4 < old_dim_r; i4++){    
          Vector<double> vaux = psi_old(Range(0,old_dim_l*2-1),0,i3,i4);
          psi_aux(Range(0,dim_l-1),0,i3,i4) = product(UL.t(),vaux);
        }
      }
// We now transform the right part
      psi_aux.resize(dim_l,2,1,old_dim_r);
      psi_new.resize(dim_l,2,1,2*dim_r);
      for(int i1 = 0; i1 < dim_l; i1++){
        for(int i2 = 0; i2 < 2; i2++){
          Vector<double> vaux = psi_aux(i1,i2,0,Range(0,old_dim_r-1));
          psi_new(i1,i2,0,Range(0,2*dim_r-1)) = product(UR,vaux);
        }
      } 
    }
  } else {
    if(right_size == 1) {
      seed = psi;
      return;
    } else {
      const Matrix<double> &UL = BlockL[left_size].U;  
      const Matrix<double> &UR = BlockR[right_size-1].U;  
      int old_dim_l = BlockL[left_size].U.cols();
      int old_dim_r = BlockR[right_size-2].U.cols();
      cout << old_dim_l << " " << old_dim_r << endl;
// We copy the old g.s. (in matrix form) into the tensor
      Tensor<double> psi_old(old_dim_l*2,1,1,2*old_dim_r); 
      for(int i = 0; i < 2*old_dim_l; i++)
        for(int j = 0; j < 2*old_dim_r; j++) psi_old(i,0,0,j) = psi(j,i);
// We first transform the right part
      psi_old.resize(old_dim_l,2,1,2*old_dim_r);
      Tensor<double> psi_aux(old_dim_l,2,1,dim_r);
      for(int i1 = 0; i1 < old_dim_l; i1++){    
        for(int i2 = 0; i2 < 2; i2++){    
          Vector<double> vaux = psi_old(i1,i2,0,Range(0,old_dim_r*2-1));
          psi_aux(i1,i2,0,Range(0,dim_r-1)) = product(UR.t(),vaux);
        }
      }
// We now transform the left part
      psi_aux.resize(old_dim_l,1,2,dim_r);
      psi_new.resize(dim_l*2,1,2,dim_r);
      for(int i3 = 0; i3 < 2; i3++){
        for(int i4 = 0; i4 < dim_r; i4++){
          Vector<double> vaux = psi_aux(Range(0,old_dim_l-1),0,i3,i4);
          psi_new(Range(0,2*dim_l-1),0,i3,i4) = product(UL,vaux);
        }
      } 
    }
  }

  psi_new.resize(dim_l*2,1,1,2*dim_r);
  seed.resize(2*dim_r,2*dim_l);
  for(int i = 0; i < 2*dim_l; i++)
    for(int j = 0; j < 2*dim_r; j++) seed(j,i) = psi_new(i,0,0,j);
}

//----------------------------------------------------------

double
measure(const Matrix<double> &op, const Matrix<double> &psi, int pos)
{
  double res = 0;
  Matrix<double> aux(psi); //result

  if(pos == LEFT)
    aux = product(op,psi);
  else
    aux = product(psi,op.t());

  Vector<double> v1 = aux.as_vector();
  Vector<double> v2 = psi.as_vector();
  res = product(v1,v2);
  return res;
}

double
measure(const Matrix<double> &op_left, const Matrix<double> &op_right, const Matrix<double> &psi)
{
  double res = 0;
  Matrix<double> aux(psi); //result

  Matrix <double> kk = op_right.t();
  aux = product(op_left,psi);
  aux = product(aux,op_right.t());

  Vector<double> v1 = aux.as_vector();
  Vector<double> v2 = psi.as_vector();
  res = product(v1,v2);
  return res;
}

void
DMRGSystem::Measure()
{
  const DMRGBlock &BL = BlockL[left_size];
  const DMRGBlock &BR = BlockR[right_size];

  for(int i = 0; i <= left_size; i++)
    cout << "Sz(" << i << ") = " << measure(BL.sz[i],psi,LEFT) << endl;

  for(int i = 0; i <= right_size; i++)
    cout << "Sz(" << nsites-i-1 << ") = " << measure(BR.sz[i],psi,RIGHT) << endl;

  for(int i = 0; i <= left_size; i++)
    for(int j = 0; j <= right_size; j++)
      cout << "Sz(" << i << ")Sz(" << nsites-j-1 << ") = " << measure(BL.sz[i],BR.sz[j],psi) << endl;
}
//----------------------------------------------------------
namespace dmtk
{

Matrix<double>
product(DMRGSystem &S, const Matrix<double> & psi) 
{
    int left_size = S.left_size;
    int right_size = S.right_size;
    Matrix<double> HL = S.BlockL[left_size].H;
    Matrix<double> szL = S.BlockL[left_size].sz[left_size];
    Matrix<double> splusL = S.BlockL[left_size].splus[left_size];
    Matrix<double> HR = S.BlockR[right_size].H;
    Matrix<double> szR = S.BlockR[right_size].sz[right_size];
    Matrix<double> splusR = S.BlockR[right_size].splus[right_size];
    int dim_l = HL.cols();
    int dim_r = HR.cols();
    Matrix<double> npsi(psi); //result

    npsi = product(HL,psi);
    npsi += product(psi,HR.t());

    Matrix<double> tmat(dim_l,dim_r);
    // Sz.Sz
    tmat= product(psi,szR.t());
    npsi += product(szL,tmat);
    // S+.S-
    tmat= product(psi,splusR)*0.5;
    npsi += product(splusL,tmat);
    // S-.S+
    tmat= product(psi,splusR.t())*0.5;
    npsi += product(splusL.t(),tmat);

    return npsi;
}

} // namespace dmtk
//----------------------------------------------------------

int main()
{
    // PARAMETERS-----------------------------------------------------------
    int nsites,n_states_to_keep,n_sweeps;
    cout << "Number of sites: ";
    cin  >> nsites;
    cout << "Number of states to keep: ";
    cin  >> n_states_to_keep;
    cout << "Number of sweeps in finite DMRG: ";
    cin  >> n_sweeps;
    // Operators:
    DMRGSystem S(nsites);
    
    //------------------------------------
    // WARMUP: Infinite DMRG sweep
    //------------------------------------
    
    for (int n = 1; n < nsites/2; n++){ // do infinite size dmrg
        cout << "WARMUP ITERATION " << n << endl;
        print_blocks(n,n);
        // Create HL and HR by adding the single sites to the two blocks
        S.BuildBlockLeft(n);
        S.BuildBlockRight(n);
        // find smallest eigenvalue and eigenvector
        S.GroundState();
        // Calculate density matrix
        S.DensityMatrix(LEFT);
        // Truncate
        S.Truncate(LEFT,n_states_to_keep);
        // Reflect
        S.DensityMatrix(RIGHT);
        S.Truncate(RIGHT,n_states_to_keep);
    }
    for(int iter = nsites/2; iter < nsites - 2; iter++){
        cout << "LEFT-TO-RIGHT ITERATION " << iter << endl;
        print_blocks(iter,nsites-iter-2);
        // Create HL and HR by adding the single sites to the two blocks
        S.BuildBlockLeft(iter);
        S.BuildBlockRight(nsites-iter-2);
        // find smallest eigenvalue and eigenvector
        S.GroundState();
        // Calculate density matrix
        S.DensityMatrix(LEFT);
        // Truncate
        S.Truncate(LEFT,n_states_to_keep);
    }
    cout << "*************************************" << endl;
    cout << "Start sweeps" << endl;
    cout << "*************************************"<< endl;
    int first_iter = 1;
    for (int sweep = 1; sweep <= n_sweeps; sweep++){
        for(int iter = first_iter; iter < nsites - 2; iter++){
            cout << "RIGHT-TO-LEFT ITERATION " << iter << endl;
            print_blocks(nsites-iter-2,iter);
            // Create HL and HR by adding the single sites to the two blocks
            S.BuildBlockRight(iter);
            S.BuildBlockLeft(nsites-iter-2);
            // find smallest eigenvalue and eigenvector
            S.BuildSeed(RIGHT2LEFT);
            S.GroundState(true);
            // Measure correlations
            S.Measure();
            // Calculate density matrix
            S.DensityMatrix(RIGHT);
            // Truncate
            S.Truncate(RIGHT,n_states_to_keep);
        }
        for(int iter = first_iter; iter < nsites - 2; iter++){
            cout << "LEFT-TO-RIGHT ITERATION " << iter << endl;
            print_blocks(iter,nsites-iter-2);
            // Create HL and HR by adding the single sites to the two blocks
            S.BuildBlockLeft(iter);
            S.BuildBlockRight(nsites-iter-2);
            // find smallest eigenvalue and eigenvector
            S.BuildSeed(LEFT2RIGHT);
            S.GroundState(true);
            // Measure correlations
            S.Measure();
            // Calculate density matrix
            S.DensityMatrix(LEFT);
            // Truncate
            S.Truncate(LEFT,n_states_to_keep);
        }
    }


    cout << "*************************************"<<endl;
    return 0;
}



