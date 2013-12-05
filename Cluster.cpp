#include <iostream>
#include <math.h>
#include <iomanip>
#include <armadillo>
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
    void RungeKutta4();
    void LeapFrog();
};

int N = 100;                                     // number of planets
int D = 3;                                     // dimensionality
int values = 2*D;

int n = 1000;
double t_i = 0.0 ;                             // initial time
double t_f = 10.;                             // final time in years
double dt = (t_f-t_i)/n;

double R0 = 20;

double sigma = 1.0;         // Standard deviation
double mu = 10.0;           // Mean of the distribution (centered around 10 solar masses)


int main() {

    vec A = zeros(values*N);
    vec dAdt = zeros(values*N);
    vec M = zeros(N);

    double u, v, w;
    double x, y, z;
    double r, theta, phi;

    long idum = -1;

    // initial conditions

    for (int i=0; i<N; i++) {

        u = ran2(&idum);
        v = ran2(&idum);
        w = ran2(&idum);

        phi = 2*pi*w;
        r = R0*cbrt(u);
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
    //R.RungeKutta4();
    R.LeapFrog();
}

Solver:: Solver(vec A, vec dAdt, vec M) {
    this->A = A;
    this->dAdt = dAdt;
    this->M = M;
}

void Solver::SolveSystem(vec A) {

    double x, y, z, r;

    double G = pi*pi*R0*R0*R0 / (8*mu*N);

    for(int i=0; i<N; i++) {
        int vi = values*i;

        dAdt(vi+3) = 0.0;
        dAdt(vi+4) = 0.0;
        dAdt(vi+5) = 0.0;

        dAdt(vi)   = A(vi+3);                  // dx/dt = vx
        dAdt(vi+1) = A(vi+4);                  // dy/dt = vy
        dAdt(vi+2) = A(vi+5);                  // dz/dt = vz

        for(int j = 0; j < N; j++) {
            int vj = values*j;

            if (j != i) {
                x = A(vi)   - A(vj);
                y = A(vi+1) - A(vj+1);
                z = A(vi+2) - A(vj+2);

                r = sqrt(x*x + y*y + z*z);

                dAdt(vi+3) += -(G*M[j]/(r*r*r))*x;      // dvx/dt
                dAdt(vi+4) += -(G*M[j]/(r*r*r))*y;      // dvy/dt
                dAdt(vi+5) += -(G*M[j]/(r*r*r))*z;      // dvz/dt
            }
        }
    }
}

void Solver:: RungeKutta4() {

    vec k1, k2, k3, k4;
    vec C = zeros(values*N);

    fstream outFile;
    outFile.open("RK4_Cluster.dat", ios::out);

    for(int i=0; i<N; i++) {
        outFile << A(values*i) << " " << A(values*i+1) << " " << A(values*i+2) << " ";
    }
    outFile << endl;

    for(int i=0; i<n; i++) {

        Solver::SolveSystem(A);

        k1 = dAdt;
        C = A + k1*(dt/2);

        Solver::SolveSystem(C);

        k2 = dAdt;
        C = A + k2*(dt/2);

        Solver::SolveSystem(C);

        k3 = dAdt;
        C = A + k3*dt;

        Solver::SolveSystem(C);

        k4 = dAdt;

        this->A += (dt/6.)*(k1 + 2*(k2+k3) + k4);

        for(int j=0; j<N; j++) {

            outFile << A(values*j) << " " << A(values*j+1) << " " << A(values*j+2) << " ";
        }

        outFile << endl;
    }

    outFile.close();

}


void Solver::LeapFrog() {

    double x, y, z, vx, vy, vz, ax, ay, az, ax_new, ay_new, az_new;

    vec vx_new(N);
    vec vy_new(N);
    vec vz_new(N);

    fstream outFile;
    outFile.open("LeapFrog_Cluster.dat", ios::out);

    for(int j=0; j<n; j++) {

        Solver::SolveSystem(A);

        for(int i=0; i<N; i++) {

            int vi = values*i;

            x = A(vi);
            y = A(vi+1);
            z = A(vi+2);

            outFile << x << " " << y << " " << z << " ";

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

        outFile << endl;

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
    }

    outFile.close();
}



