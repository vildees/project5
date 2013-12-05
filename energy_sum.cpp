#include <iostream>
#include <math.h>
#include <iomanip>
#include <armadillo>
#include <omp.h>

#include "planet.h"
#include "constants.h"
#include "gaussiandeviate.cpp"

using namespace std;
using namespace arma;

class Solver {

public:
    vec A;
    vec dAdt;
    vec M;

    Solver(vec A, vec dAdt, vec M);

    void SolveSystem(vec A);
    void LeapFrog();
    vec kinetic_energy(vec A);
    vec potential_energy(vec A);
};

int N      = 300;                                  // number of planets
int D      = 3;                                    // dimensionality
int values = 2*D;

int    n   = 1000;
double t_i = 0.0 ;                             // initial time
double t_f = 5.;                               // final time in years
double dt  = (t_f-t_i)/n;

double R0 = 20;

double sigma = 1.0;         // Standard deviation
double mu    = 10.0;        // Mean of the distribution (centered around 10 solar masses)

double epsilon = 0.; //0.15

int main() {

    vec A    = zeros(values*N);
    vec dAdt = zeros(values*N);
    vec M    = zeros(N);

    double u, v, w;
    double x, y, z;
    double r, theta, phi;

    long idum = -1;

    // initial conditions

    for (int i=0; i<N; i++) {

        u = ran2(&idum);
        v = ran2(&idum);
        w = ran2(&idum);

        phi   = 2*pi*w;
        r     = R0*cbrt(u);
        theta = acos(1-2*v);

        x = r*sin(theta)*cos(phi);
        y = r*sin(theta)*sin(phi);
        z = r*cos(theta);

        int j = values*i;

        A(j)   = x;                     // x - position
        A(j+1) = y;                     // y - position
        A(j+2) = z;                     // z - position
        A(j+3) = 0.0;                     // vx - velocity
        A(j+4) = 0.0;                     // vy - velocity
        A(j+5) = 0.0;                     // vz - velocity

        M(i) = sigma*gaussian_deviate(&idum) + mu;
    }


    Solver R = Solver(A, dAdt, M);
    R.LeapFrog();
}

Solver:: Solver(vec A, vec dAdt, vec M) {
    this->A = A;
    this->dAdt = dAdt;
    this->M = M;
}

void Solver::SolveSystem(vec A) {

    double x, y, z, r;
    int vi, vj;
    double G = pi*pi*R0*R0*R0 / (8*mu*N);
    //vec dAdt2 = zeros(6*N);

    //#pragma omp parallel for private(vi,x,y,z,r,vj) shared(dAdt,A)
    for(int i=0; i<N; i++) {
        vi = values*i;

        dAdt(vi+3) = 0.0;
        dAdt(vi+4) = 0.0;
        dAdt(vi+5) = 0.0;

        dAdt(vi)   = A(vi+3);                  // dx/dt = vx
        dAdt(vi+1) = A(vi+4);                  // dy/dt = vy
        dAdt(vi+2) = A(vi+5);                  // dz/dt = vz

        for(int j = 0; j < N; j++) {
            vj = values*j;

            if (j != i) {
                x = A(vi)   - A(vj);
                y = A(vi+1) - A(vj+1);
                z = A(vi+2) - A(vj+2);

                r = sqrt(x*x + y*y + z*z);

                dAdt(vi+3) += -(G*M[j]/(r*r*r + epsilon*epsilon))*x;      // dvx/dt
                dAdt(vi+4) += -(G*M[j]/(r*r*r + epsilon*epsilon))*y;      // dvy/dt
                dAdt(vi+5) += -(G*M[j]/(r*r*r + epsilon*epsilon))*z;      // dvz/dt
            }
        }
    }
}



void Solver::LeapFrog() {

    double x, y, z, vx, vy, vz, ax, ay, az, ax_new, ay_new, az_new;

    vec vx_new(N);
    vec vy_new(N);
    vec vz_new(N);

    fstream outFile;

    outFile.open("energy.dat", ios::out);

    for(int j=0; j<n; j++) {
        cout << float(j) / n << endl;
        Solver::SolveSystem(A);

        for(int i=0; i<N; i++) {

            int vi = values*i;

            x = A(vi);
            y = A(vi+1);
            z = A(vi+2);

            vx = dAdt(vi);
            vy = dAdt(vi+1);
            vz = dAdt(vi+2);

            ax = dAdt(vi+3);                        // ax(t)
            ay = dAdt(vi+4);                        // ay(t)
            az = dAdt(vi+5);                        // az(t)

            vx_new(i) =  vx + (dt/2)*ax;            // vx(t+dt/2)
            vy_new(i) =  vy + (dt/2)*ay;            // vy(t+dt/2)
            vz_new(i) =  vz + (dt/2)*az;            // vz(t+dt/2)

            A(vi)   = x + dt*vx_new(i);             // x(t+dt)
            A(vi+1) = y + dt*vy_new(i);             // y(t+dt)
            A(vi+2) = z + dt*vz_new(i);             // z(t+dt)
        }

        Solver::SolveSystem(A);

        for(int i=0; i<N; i++) {

            int vi = values*i;

            ax_new = dAdt(vi+3);                   // ax(t+dt)
            ay_new = dAdt(vi+4);                   // ay(t+dt)
            az_new = dAdt(vi+5);                   // az(t+dt)

            A(vi+3) = vx_new(i) + (dt/2)*ax_new;    // vx(t+h)
            A(vi+4) = vy_new(i) + (dt/2)*ay_new;    // vy(t+h)
            A(vi+5) = vz_new(i) + (dt/2)*az_new;    // vz(t+h)
        }

        vec K = Solver::kinetic_energy(A);
        vec P = Solver::potential_energy(A);

      vec E = K + P;

        double kin = 0.0;
        double pot = 0.0;

        for(int k = 0; k < N; k++) {
            if (E(k) < 0) {
                kin += K(k);
                pot += P(k);
            }
            if (E(k) > 0) {
                pot -= P(k);
            }
        }

        outFile << kin << " " << pot << endl;
    }

    outFile.close();
}

vec Solver::kinetic_energy(vec A) {

    double vx, vy, vz;
    double v;

    vec kin_energy = zeros(N);

    for(int i=0; i<N; i++) {

        double vi = values*i;

        vx = A(vi+3);
        vy = A(vi+4);
        vz = A(vi+5);

        v = sqrt(vx*vx + vy*vy + vz*vz);

        kin_energy(i) = 0.5*M(i)*v*v;
    }

    return kin_energy;
}

vec Solver::potential_energy(vec A) {

    double x, y, z, r;
    double G = pi*pi*R0*R0*R0 / (8*mu*N);

    vec pot_energy = zeros(N);

    for(int i=0; i<N; i++) {
        int vi = values*i;

        for(int j = 0; j < N; j++) {
            int vj = values*j;

            if (j != i) {
                x = A(vi)   - A(vj);
                y = A(vi+1) - A(vj+1);
                z = A(vi+2) - A(vj+2);

                r = sqrt(x*x + y*y + z*z);

                pot_energy(i) = - G*M(i)*M(j) / r;
            }
        }
    }

    return 0.5*pot_energy;
}
