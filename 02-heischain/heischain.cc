#include <iostream>
#include <dmtk/dmtk.h>

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

class DMRGSystem
{
    public:
        Matrix<double> sz0; // single site Sz
        Matrix<double> splus0; // single site S+
        Vector<Matrix<double> > HL; // left block Hamiltonian
        Vector<Matrix<double> > HR; // right block Hamiltonian
        Vector<Matrix<double> > szL; // left block Sz 
        Vector<Matrix<double> > szR; // right block Sz 
        Vector<Matrix<double> > splusL; // left block S+
        Vector<Matrix<double> > splusR; // right block S+

        Matrix<double> psi;	// g.s. wave function
        Matrix<double> rho;	// density matrix
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
            HL.resize(nsites);
            HR.resize(nsites);
            szL.resize(nsites);
            splusL.resize(nsites);
            szR.resize(nsites);
            splusR.resize(nsites);

            // initialize Hamiltonian:
            sz0 = 0.0;
            splus0 = 0.0;    
            HL[0] = sz0;    
            HR[0] = sz0;     
            // single-site operators
            sz0(0,0)         = -0.5;
            sz0(1,1)         =  0.5;
            splus0(1,0)      =  1.0;
            szR[0] = sz0;
            szL[0] = sz0;
            splusR[0] = splus0;
            splusL[0] = splus0;
          }

        void BuildBlockLeft(int _iter);        
        void BuildBlockRight(int _iter);        
        void GroundState();
        void DensityMatrix(int _position);        
        void Truncate(int _position, int _m);        
        void Reflect()
          {
            szR[right_size] = szL[left_size];
            splusR[right_size] = splusL[left_size];
          }
};

void
DMRGSystem::BuildBlockLeft(int _iter)
{
    left_size = _iter;
    int dim_l = HL[left_size-1].cols();
    Matrix<double> I_left = eye(dim_l);
    Matrix<double> I2 = eye(2);
    //enlarge left block:
    HL[left_size]  = tensor(HL[left_size-1],I2) +
                     tensor(szL[left_size-1],sz0) +
                     0.5*tensor(splusL[left_size-1],splus0.t()) +
                     0.5*tensor(splusL[left_size-1].t(),splus0);
    splusL[left_size] = tensor(I_left,splus0);
    szL[left_size] = tensor(I_left,sz0);

}

void
DMRGSystem::BuildBlockRight(int _iter)
{
    right_size = _iter;
    int dim_r = HR[right_size-1].cols();
    Matrix<double> I_right= eye(dim_r);
    Matrix<double> I2 = eye(2);
    //enlarge right block:
    HR[right_size] = tensor(I2,HR[right_size-1]) +
                     tensor(sz0,szR[right_size-1]) +
                     0.5* tensor(splus0.t(),splusR[right_size-1]) +
                     0.5* tensor(splus0,splusR[right_size-1].t());
    splusR[right_size] = tensor(splus0,I_right);
    szR[right_size] = tensor(sz0,I_right) ;
}

void
DMRGSystem::GroundState()
{
    double ev;
    Vector<double> lanczos_a(100);
    Vector<double> lanczos_b(100);
    int dim_l = HL[left_size].cols();
    int dim_r = HR[right_size].cols();
    psi.resize(dim_r,dim_l);
    Matrix<double> evec(dim_r,dim_l);
    int maxiter = -1;

    lanczos<double, DMRGSystem, Matrix<double> >(*this, psi, evec, ev, lanczos_a, lanczos_b, maxiter, 1.e-7, false, true, "vectors.dat");

    energy = ev;
}

void
DMRGSystem::DensityMatrix(int _position)
{
    int dim_l = HL[left_size].cols();
    int dim_r = HR[right_size].cols();
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
      aux2 = product(HL[left_size],rho_evec);
      HL[left_size] = product(U,aux2);
      aux2 = product(splusL[left_size],rho_evec);
      splusL[left_size] = product(U,aux2);
      aux2 = product(szL[left_size],rho_evec);
      szL[left_size] = product(U,aux2);
    } else {
      aux2 = product(HR[right_size],rho_evec);
      HR[right_size] = product(U,aux2);
      aux2 = product(splusR[right_size],rho_evec);
      splusR[right_size] = product(U,aux2);
      aux2 = product(szR[right_size],rho_evec);
      szR[right_size] = product(U,aux2);
    }
}

//----------------------------------------------------------
namespace dmtk
{

Matrix<double>
product(DMRGSystem &S, const Matrix<double> & psi) 
{
    int left_size = S.left_size;
    int right_size = S.right_size;
    int dim_l = S.HL[left_size].cols();
    int dim_r = S.HR[right_size].cols();
    Matrix<double> npsi(psi); //result

    npsi = product(S.HL[left_size],psi);
                     
    npsi += product(psi,S.HR[right_size].t());

    Matrix<double> tmat(dim_l,dim_r);
    // Sz.Sz
    tmat= product(psi,S.szR[right_size].t());
    npsi += product(S.szL[left_size],tmat);
    // S+.S-
    tmat= product(psi,S.splusR[right_size])*0.5;
    npsi += product(S.splusL[left_size],tmat);
    // S-.S+
    tmat= product(psi,S.splusR[right_size].t())*0.5;
    npsi += product(S.splusL[left_size].t(),tmat);

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
    cout << "*************************************" << endl;
    cout << "Start sweeps" << endl;
    cout << "*************************************"<< endl;
    int first_iter = nsites/2;
    for (int sweep = 1; sweep <= n_sweeps; sweep++){
        for(int iter = first_iter; iter < nsites - 3; iter++){
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
            // We copy the left blocks onto the right blocks
        }
        first_iter = 1;
        for(int iter = first_iter; iter < nsites - 3; iter++){
            cout << "RIGHT-TO-LEFT ITERATION " << iter << endl;
            print_blocks(nsites-iter-2,iter);
            // Create HL and HR by adding the single sites to the two blocks
            S.BuildBlockRight(iter);
            S.BuildBlockLeft(nsites-iter-2);
            // find smallest eigenvalue and eigenvector
            S.GroundState();
            // Calculate density matrix
            S.DensityMatrix(RIGHT);
            // Truncate
            S.Truncate(RIGHT,n_states_to_keep);
            // We copy the left blocks onto the right blocks
        }
    }
    cout << "*************************************"<<endl;
    return 0;
}



