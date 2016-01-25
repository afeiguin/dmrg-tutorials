#include <iostream>
#include <dmtk/dmtk.h>

using namespace std;
using namespace dmtk;

#define POW2(i) (1 << i)

//-------------------------------------------------------------

inline Matrix<double> 
eye(int n)
{
    Matrix<double> Identity(n,n);
    Identity = I<double>();
    return Identity;
}

//----------------------------------------------------------

void print_blocks(int l)
{
    cout << "**********************************"<< endl;
    cout << "LEFT SIZE = " << l << endl;
    while (l--)
        cout << "X ";
    cout << "* \n";
    return;
}     

//----------------------------------------------------------

void
GroundState(Matrix<double> & H)
{
    double ev;
    Vector<double> lanczos_a(100);
    Vector<double> lanczos_b(100);
    Vector<double> psi(H.rows());
    Vector<double> evec(H.rows());
    int maxiter = -1;

    lanczos<double, Matrix<double>, Vector<double> >(H, psi, evec, ev, lanczos_a, lanczos_b, maxiter, 1.e-7, false, true, "vectors.dat");
}

//----------------------------------------------------------

int main()
{
    // PARAMETERS-----------------------------------------------------------
    int nsites;
    cout << "Number of sites: ";
    cin  >> nsites;

    // Single site operators
    Matrix<double> sz0(2,2); // single site Sz
    Matrix<double> splus0(2,2); // single site S+
    sz0(0,0)         = -0.5;
    sz0(1,1)         =  0.5;
    splus0(1,0)      =  1.0;
 
    Matrix<double> term_szsz(4,4); //auxiliary matrix to store Sz.Sz 
    term_szsz = tensor(sz0,sz0);

    Matrix<double> term_spsm(4,4); //auxiliary matrix to store 1/2 S+.S- 
    term_spsm = tensor(splus0,splus0.t())*0.5;
 
    // Hamiltonian 
    int maxdim = POW2(nsites);
    cout << "MAXDIM = " << maxdim << endl;
    Matrix<double> H(2,2); // Hamiltonian matrix
    H = 0.;
    for(int i = 1; i < nsites; i++){;
       int diml = POW2(i);
       int dim = diml*2;
       print_blocks(i);
       cout << "ADDING SITE " << i << " DIML= " << diml << " DIM= " << diml*2 << endl;

       Matrix<double> Ileft(diml,diml);
       Ileft = eye(diml);

       Matrix<double> aux(dim,dim); 
       aux = tensor(H,eye(2)); 
       H = aux;
 
       // Sz.Sz
       H = H + tensor(Ileft,term_szsz);

       // (1/2)(Sp.Sm + Sm.Sp) 
       H = H + tensor(Ileft,term_spsm);
       H = H + tensor(Ileft,term_spsm.t());
       GroundState(H);  //Diagonalize the matrix
    }

    return 0;
}



