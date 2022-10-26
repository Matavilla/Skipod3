#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#include "mpi.h"

double fun() {
    double x = 2.0 * ((float) rand() / RAND_MAX) - 1.0;
    double y = 2.0 * ((float) rand() / RAND_MAX) - 1.0;
    double z = 1.0 * ((float) rand() / RAND_MAX);
    double value = 0;
    if (z >= sqrt(x*x + y*y))
        value = sqrt(x * x + y * y);
    return value;
}

int main(int argc, char **argv) {
    double eps = atof(argv[1]);
    const double solution = M_PI / 6;

    MPI_Init(&argc, &argv);
    const double startTime = MPI_Wtime();

    int wrank, wsize;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    srand(wrank + wsize + 50);
    
    unsigned long long gNumPoints = 0;
    double gSum = 0.0;
    double gTime = 0.0;
    double gSolution = 0;

    int flag = 1;
    unsigned long long lNumPoints = 1;
    while (flag) {
        //std::cout << flag << " " << gSum << " " << gNumPoints << " " << gSolution << std::endl;
        double lSum = 0.0;
        lNumPoints = 10000;
        for (unsigned i = 0; i < lNumPoints; i++)
            lSum += fun();
        
        double tmp1 = 0;
        unsigned long long tmp2 = 0;
        MPI_Reduce(&lSum, &tmp1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&lNumPoints, &tmp2, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        gSum += tmp1;
        gNumPoints += tmp2;
        if (wrank == 0) {
            gSolution = 4 * gSum / gNumPoints;
            if (fabs(gSolution - solution) < eps)
                flag = 0;
        }
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    const double lTime = MPI_Wtime() - startTime;
    MPI_Reduce(&lTime, &gTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (wrank == 0) {
        std::cout << "Ans: " << gSolution << std::endl << "Error: " << fabs(gSolution - solution) << std::endl
                  << "N: " << gNumPoints << std::endl << "Time: " << gTime << std::endl;
    }
    MPI_Finalize();
    return 0;
}
