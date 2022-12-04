#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "mpi.h"

double Lx = 0;
double Ly = 0;
double Lz = 0;

double hx = 0;
double hy = 0;
double hz = 0;

size_t N = 0;

double T = 0;
double tau = 0;
double K = 0;

std::vector< std::vector<double> > u(3);

void SetGlobalVar(char **argv) {
    double L = atof(argv[1]);
    Lx = L;
    Ly = L;
    Lz = L;
    N = atoi(argv[2]);

    hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;

    T = 2.0;
    K = 1000;
    tau = T / K;
}

size_t GetIndex(size_t i, size_t j, size_t k) {
    return (i * (N + 2) + j) * (N + 2) + k;
}

double Solution(double x, double y, double z, double t = 0.0) {
    const double a = M_PI * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz));
    return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a * t + 2.0 * M_PI);
}

double laplace(double* u, size_t i, size_t j, size_t k) {
    return (u[GetIndex(i - 1, j, k)] - 2 * u[GetIndex(i, j, k)] + u[GetIndex(i + 1, j, k)]) / (hx * hx) +
           (u[GetIndex(i, j - 1, k)] - 2 * u[GetIndex(i, j, k)] + u[GetIndex(i, j + 1, k)]) / (hy * hy) +
           (u[GetIndex(i, j, k - 1)] - 2 * u[GetIndex(i, j, k)] + u[GetIndex(i, j, k + 1)]) / (hz * hz);
}

void FillBoundary(size_t step) {
    for (size_t i = 0; i <= N; i++) {
        for (size_t j = 0; j <= N; j++) {
            u[step % 3][GetIndex(0, i, j)] = 0;
            u[step % 3][GetIndex(N, i, j)] = 0;

            u[step % 3][GetIndex(i, 0, j)] = 0;
            u[step % 3][GetIndex(i, N, j)] = 0;
        }
    }
    for (size_t i = 1; i < N; i++) {
        for (size_t j = 1; j < N; j++) {
            if (step <= 1) {
                u[step % 3][GetIndex(i, j, 0)] = Solution(i * hx, j * hy, 0, step * tau);
                u[step % 3][GetIndex(i, j, N)] = Solution(i * hx, j * hy, Lz, step * tau);
            } else {
                size_t ind = GetIndex(i, j, N);
                u[(step - 1) % 3][GetIndex(i, j, N + 1)] = u[(step - 1) % 3][GetIndex(i, j, 1)];
                u[step % 3][ind] = 2 * u[(step - 1) % 3][ind] - u[(step - 2) % 3][ind] + tau * tau * laplace(&u[(step - 1) % 3][0], i, j, N);

                u[step % 3][GetIndex(i, j, 0)] = u[step % 3][GetIndex(i, j, N)];
            }
        }
    }
}

void InitFirstSteps() {
    FillBoundary(0);
    FillBoundary(1);

    for (size_t i = 1; i < N; i++)
        for (size_t j = 1; j < N; j++)
            for (size_t k = 1; k < N; k++)
                u[0][GetIndex(i, j, k)] = Solution(i * hx, j * hy, k * hz);

    for (size_t i = 1; i < N; i++)
        for (size_t j = 1; j < N; j++)
            for (size_t k = 1; k < N; k++)
                u[1][GetIndex(i, j, k)] = u[0][GetIndex(i, j, k)] + 0.5 * tau * tau * laplace(&u[0][0], i, j, k);
}

void CalculateError(size_t step) {
    double err = 0.0;
    size_t i_t = 0, j_t = 0, k_t = 0;
    for (size_t i = 0; i <= N; i++)
        for (size_t j = 0; j <= N; j++)
            for (size_t k = 0; k <= N; k++)
            {
                if (fabs(u[step % 3][GetIndex(i, j, k)] - Solution((i) * hx, (j) * hy, k * hz, step * tau)) > err)
                {
                    i_t = i;
                    j_t = j;
                    k_t = k;
                }
                err = std::max(err, fabs(u[step % 3][GetIndex(i, j, k)] - Solution(i * hx, j * hy, k * hz, step * tau)));
            }
    std::cout << "Max Error on step = " << step << ": " << err << " " << i_t << " " << j_t << " " << k_t << std::endl;
}

int main(int argc, char **argv) {
    SetGlobalVar(argv);

    u[0].resize((N + 2) * (N + 2) * (N + 2));
    u[1].resize((N + 2) * (N + 2) * (N + 2));
    u[2].resize((N + 2) * (N + 2) * (N + 2));

    InitFirstSteps();

    for (size_t step = 2; step <= 20; step++)
    {
        for (size_t i = 1; i < N; i++)
        {
            for (size_t j = 1; j < N; j++)
            {
                for (size_t k = 1; k < N; k++)
                {
                    size_t ind = GetIndex(i, j, k);
                    u[step % 3][ind] = 2 * u[(step + 2) % 3][ind] - u[(step + 1) % 3][ind] + tau * tau * laplace(&u[(step + 2) % 3][0], i, j, k);
                }
            }
        }
        FillBoundary(step);
    }

    CalculateError(20);

    return 0;
}
