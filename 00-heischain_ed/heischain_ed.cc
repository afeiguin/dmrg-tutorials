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
    // PARAMETERS
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
    int maxdim = POW2(nsites); // 2^N states
    cout << "MAXDIM = " << maxdim << endl;
    Matrix<double> H(maxdim,maxdim); // Hamiltonian matrix

    for(int i = 1; i < nsites; i++){;
       // We add the term for the interaction S_i.S_{i+1}

       int diml = POW2(i-1);        // 2^(i-1)
       int dimr = POW2(nsites-i-1); // 2^(nsites-i-1)

       print_blocks(i-1,nsites-i-1);
       cout << "SITE " << i << " DIML= " << diml << " DIMR= " << dimr << endl;

       Matrix<double> Ileft(diml,diml),Iright(dimr,dimr);
       Ileft = eye(diml);
       Iright = eye(dimr);
 
       Matrix<double> aux(2*diml,2*diml); // auxiliary matrix to store the term
       // Sz.Sz
       aux = tensor(Ileft,term_szsz);
       H = H + tensor(aux,Iright);

       // (1/2)(Sp.Sm + Sm.Sp) 
       aux = tensor(Ileft,term_spsm);
       H = H + tensor(aux,Iright);
       aux = tensor(Ileft,term_spsm.t());
       H = H + tensor(aux,Iright);
    }

    GroundState(H); //Diagonalize the matrix
    return 0;
}



