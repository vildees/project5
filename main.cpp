/*
   D - dimensionality
   n - number of objects
   A - vector containing the position and velocities for all the objects (length : 2*D*n)
   dAdt - the time derivatives of A
   B - initial conditions (posisition and velocity) for the different objects

*/

#include <iostream>
#include <math.h>
#include <iomanip>
#include <armadillo>
#include "planet.h"
#include "constants.h"

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

int n = 2;                                     // number of planets
int D = 3;                                     // dimensionality
int values = 2*D;

int N = 1000;
double t_i = 0.0 ;                             // initial time
double t_f = 2.;                             // final time in years
double dt = (t_f-t_i)/N;

int main() {

    vec A = zeros(values*n);
    vec dAdt = zeros(values*n);
    vec M = zeros(n);

    planet Earth = planet(1.0, 0.0, 0.0, 0.0, 2.*pi, 0.0, M_earth);
    planet Sun = planet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, M_sun);
    planet B[n];

    B[0] = Earth;
    B[1] = Sun;

    // initial conditions

    for (int i=0; i<n; i++) {

        int j = values*i;

        if (D == 2) {
            A(j) = B[i].x0;                       // x - position
            A(j+1) = B[i].y0;                     // y - position
            A(j+2) = B[i].vx;                     // vx - velocity
            A(j+3) = B[i].vy;                     // vy - velocity

            M[i] = B[i].M;                        // mass
        }

        if (D == 3) {
            A(j)   = B[i].x0;                     // x - position
            A(j+1) = B[i].y0;                     // y - position
            A(j+2) = B[i].z0;                     // z - position
            A(j+3) = B[i].vx;                     // vx - velocity
            A(j+4) = B[i].vy;                     // vy - velocity
            A(j+5) = B[i].vz;                     // vz - velocity

            M(i) = B[i].M;
        }
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

    if (D == 2) {

        for(int i=0; i<n; i++) {
            int vi = values*i;

            dAdt(vi)   = A(vi+2);                  // dx/dt = vx
            dAdt(vi+1) = A(vi+3);                  // dy/dt = vy

            for (int j = 0; j < n; j++ ) {
                int vj = values*j;

                if (j != i) {
                    x = A(vi) - A(vj);
                    y = A(vi+1) - A(vj+1);

                    r = sqrt(x*x + y*y);

                    dAdt(vi+2) += -(G*M[j]/(r*r*r))*x;                // dvx/dt
                    dAdt(vi+3) += -(G*M[j]/(r*r*r))*y;                // dvy/dt
                }
            }
        }
    }

    if (D == 3) {

        for(int i=0; i<n; i++) {
            int vi = values*i;

            dAdt(vi+3) = 0.0;
            dAdt(vi+4) = 0.0;
            dAdt(vi+5) = 0.0;

            dAdt(vi)   = A(vi+3);                  // dx/dt = vx
            dAdt(vi+1) = A(vi+4);                  // dy/dt = vy
            dAdt(vi+2) = A(vi+5);                  // dz/dt = vz

            for(int j = 0; j < n; j++) {
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

}

void Solver:: RungeKutta4() {

    vec k1, k2, k3, k4;
    vec C = zeros(values*n);

    fstream outFile;
    outFile.open("RK4.dat", ios::out);

    for(int i=0; i<n; i++) {
        outFile << A(values*i) << " " << A(values*i+1) << " " << A(values*i+2) << " ";
    }
    outFile << endl;

    for(int i=0; i<N; i++) {

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

        for(int j=0; j<n; j++) {

           outFile << A(values*j) << " " << A(values*j+1) << " " << A(values*j+2) << " ";
        }

        outFile << endl;
    }

    outFile.close();

}


void Solver::LeapFrog() {

    double x, y, z, vx, vy, vz, ax, ay, az, ax_new, ay_new, az_new;

    vec vx_new(n);
    vec vy_new(n);
    vec vz_new(n);

    fstream outFile;
    outFile.open("LeapFrog.dat", ios::out);

    for(int j=0; j<N; j++) {

        Solver::SolveSystem(A);

        for(int i=0; i<n; i++) {

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

        for(int i=0; i<n; i++) {

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


