#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "omp.h"

#define size_t unsigned

double Lx = 0;
double Ly = 0;
double Lz = 0;

double hx = 0;
double hy = 0;
double hz = 0;

size_t N = 0;

size_t Nx, Ny;
size_t xStart;
size_t yStart;

double T = 0;
double tau = 0;
double K = 0;

int wrank, wsize, rowRank, rowSize, colRank, colSize;
MPI_Comm newComm, rowComm, colComm; 

double* u[3];

double* exchangeData[8];

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
    K = 10000;
    tau = T / K;
}

void CreateNewTopology() {
    size_t n_x = 0, n_y = 0;
    if (wsize == 1) {
        n_x = 1;
        n_y = 1;
    } else if (wsize > 4 && wsize % 4 == 0) {
        n_x = wsize / 4;
        n_y = 4;
    } else if (wsize % 2 == 0) {
        n_x = wsize / 2;
        n_y = 2;
    }

    int dims[2], periods[2], subdims[2];
    dims[0] = n_x;
    dims[1] = n_y;
    periods[0] = periods[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &newComm);

    subdims[0] = 0;
    subdims[1] = 1;
    MPI_Cart_sub(newComm, subdims, &colComm);

    subdims[0] = 1;
    subdims[1] = 0;
    MPI_Cart_sub(newComm, subdims, &rowComm);

    MPI_Comm_rank(rowComm, &rowRank);
    MPI_Comm_size(rowComm, &rowSize);
    MPI_Comm_rank(colComm, &colRank);
    MPI_Comm_size(colComm, &colSize);

    Nx = N / rowSize;
    if (rowRank == 0)
        Nx += N % rowSize;
    Ny = N / colSize;

    xStart = rowRank * Nx;
    if (rowRank != 0)
        xStart += N % rowSize;

    yStart = colRank * Ny;

    u[0] = malloc((Nx + 2) * (Ny + 2) * (N + 2) * 8);
    u[1] = malloc((Nx + 2) * (Ny + 2) * (N + 2) * 8);
    u[2] = malloc((Nx + 2) * (Ny + 2) * (N + 2) * 8);

    exchangeData[0] = malloc((Ny + 1) * (N + 1) * 8);
    exchangeData[1] = malloc((Ny + 1) * (N + 1) * 8);
    exchangeData[2] = malloc((Ny + 1) * (N + 1) * 8);
    exchangeData[3] = malloc((Ny + 1) * (N + 1) * 8);
    exchangeData[4] = malloc((Nx + 1) * (N + 1) * 8);
    exchangeData[5] = malloc((Nx + 1) * (N + 1) * 8);
    exchangeData[6] = malloc((Nx + 1) * (N + 1) * 8);
    exchangeData[7] = malloc((Nx + 1) * (N + 1) * 8);
}

void FreeData() {
    free(u[0]);
    free(u[1]);
    free(u[2]);
    free(exchangeData[0]);
    free(exchangeData[1]);
    free(exchangeData[2]);
    free(exchangeData[3]);
    free(exchangeData[4]);
    free(exchangeData[5]);
    free(exchangeData[6]);
    free(exchangeData[7]);
}

double Solution(double x, double y, double z, double t) {
    const double a = M_PI * sqrt(1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz));
    return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a * t + 2.0 * M_PI);
}

void SendRecvAndWait(size_t num, size_t size, size_t dest, MPI_Comm* comm) {
    MPI_Request requests[2];
    MPI_Status stat[2];
    MPI_Isend(exchangeData[num], size, MPI_DOUBLE, dest, 0, *comm, &requests[0]);
    MPI_Irecv(exchangeData[num + 1], size, MPI_DOUBLE, dest, 0, *comm, &requests[1]);
    MPI_Waitall(2, requests, stat);
}

void ExchangeUp(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (colRank + 1 == colSize)
    return;

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i <= Nx; i++)
        for (size_t k = 0; k <= N; k++)
            exchangeData[6][i * (N + 1) + k] = uNew[i][Ny][k];

    SendRecvAndWait(6, (Nx + 1) * (N + 1), colRank + 1, &colComm);

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i <= Nx; i++)
        for (size_t k = 0; k <= N + 1; k++)
            uNew[i][Ny + 1][k] = exchangeData[7][i * (N + 1) + k];
}

void ExchangeDown(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (colRank == 0)
        return;

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i <= Nx; i++)
        for (size_t k = 0; k <= N; k++)
            exchangeData[4][i * (N + 1) + k] = uNew[i][1][k];

    SendRecvAndWait(4, (Nx + 1) * (N + 1), colRank - 1, &colComm);

#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i <= Nx; i++)
        for (size_t k = 0; k <= N ; k++)
            uNew[i][0][k] = exchangeData[5][i * (N + 1) + k];
}

void ExchangeRight(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (rowRank + 1 == rowSize)
        return;

#pragma omp parallel for num_threads(4)
    for (size_t j = 0; j <= Ny; j++)
        for (size_t k = 0; k <= N; k++)
            exchangeData[2][j * (N + 1) + k] = uNew[Nx][j][k];

    SendRecvAndWait(2, (Ny + 1) * (N + 1), rowRank + 1, &rowComm);

#pragma omp parallel for num_threads(4)
    for (size_t j = 0; j <= Ny; j++)
        for (size_t k = 0; k <= N; k++)
            uNew[Nx + 1][j][k] = exchangeData[3][j * (N + 1) + k];
}

void ExchangeLeft(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (rowRank == 0)
        return;

#pragma omp parallel for num_threads(4)
    for (size_t j = 0; j <= Ny; j++)
        for (size_t k = 0; k <= N; k++)
            exchangeData[0][j * (N + 1) + k] = uNew[1][j][k];

    SendRecvAndWait(0, (Ny + 1) * (N + 1), rowRank - 1, &rowComm);

#pragma omp parallel for num_threads(4)
    for (size_t j = 0; j <= Ny; j++)
            for (size_t k = 0; k <= N; k++)
                uNew[0][j][k] = exchangeData[1][j * (N + 1) + k];
}

void ExchangeData(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (wsize == 1)
        return;

    ExchangeLeft(uNew);
    ExchangeRight(uNew);
    ExchangeDown(uNew);
    ExchangeUp(uNew);
}

double CalculateError(const size_t step, double uNew[Nx + 2][Ny + 2][N + 2]) {
    double err = 0.0;
#pragma omp parallel for num_threads(4), reduction(max:err)
    for (size_t i = 0; i <= Nx; i++)
        for (size_t j = 0; j <= Ny; j++)
            for (size_t k = 0; k <= N; k++)
		        if (fabs(uNew[i][j][k] - Solution((xStart + i) * hx, (yStart + j) * hy, k * hz, step * tau)) > err)
			        err = fabs(uNew[i][j][k] - Solution((xStart + i) * hx, (yStart + j) * hy, k * hz, step * tau));
    return err;
}

void FillXYBoundary(double uNew[Nx + 2][Ny + 2][N + 2]) {
    if (rowRank == 0) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j <= Ny; j++)
            for (size_t k = 0; k <= N; k++)
                uNew[0][j][k] = 0;
    }

    if (rowRank + 1 == rowSize) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j <= Ny; j++)
            for (size_t k = 0; k <= N; k++)
                uNew[Nx][j][k] = 0;
    }
    
    if (colRank == 0) {
#pragma omp parallel for num_threads(4)
        for (size_t i = 0; i <= Nx; i++)
            for (size_t k = 0; k <= N; k++)
                uNew[i][0][k] = 0;
    }

    if (colRank + 1 == colSize) {
#pragma omp parallel for num_threads(4)
        for (size_t i = 0; i <= Nx; i++)
            for (size_t k = 0; k <= N; k++)
                uNew[i][Ny][k] = 0;
    }
}

void FillZBoundary(size_t step, double uNew[Nx + 2][Ny + 2][N + 2]) {
#pragma omp parallel for num_threads(4)
    for (size_t i = 1; i <= Nx; i++)
        for (size_t j = 1; j <= Ny; j++)
        {
            if (xStart + i == N || yStart + j == N)
                continue;
            uNew[i][j][0] = Solution((xStart + i) * hx, (yStart + j) * hy, 0, step * tau);
            uNew[i][j][N] = Solution((xStart + i) * hx, (yStart + j) * hy, 0, step * tau);
        }
}

void InitFirstSteps(double uPPrev[Nx + 2][Ny + 2][N + 2], double uPrev[Nx + 2][Ny + 2][N + 2]) {
    {
        FillXYBoundary(uPPrev);
        FillZBoundary(0, uPPrev);

        for (size_t i = 1; i <= Nx; i++)
#pragma omp parallel for num_threads(4)
            for (size_t j = 1; j <= Ny; j++)
                for (size_t k = 1; k < N; k++)
                {
                    if (xStart + i == N || yStart + j == N)
                        continue;
                    uPPrev[i][j][k] = Solution((xStart + i) * hx, (yStart + j) * hy, k * hz, 0);
                }
        
        ExchangeData(uPPrev);
    }
    {
        FillXYBoundary(uPrev);
        FillZBoundary(1, uPrev);

        for (size_t i = 1; i <= Nx; i++)
#pragma omp parallel for num_threads(4)
            for (size_t j = 1; j <= Ny; j++)
                for (size_t k = 1; k < N; k++)
                {
                    if (xStart + i == N || yStart + j == N)
                        continue;
                    uPrev[i][j][k] = uPPrev[i][j][k] + 0.5 * tau * tau * (
                                    (uPPrev[i - 1][j][k] - 2 * uPPrev[i][j][k] + uPPrev[i + 1][j][k]) / (hx * hx) +
                                    (uPPrev[i][j - 1][k] - 2 * uPPrev[i][j][k] + uPPrev[i][j + 1][k]) / (hy * hy) +
                                    (uPPrev[i][j][k - 1] - 2 * uPPrev[i][j][k] + uPPrev[i][j][k + 1]) / (hz * hz));
                }
    
        ExchangeData(uPrev);
    }
}

void OMPKernel(double uNew[Nx + 2][Ny + 2][N + 2], 
               double uPrev[Nx + 2][Ny + 2][N + 2],
               double uPPrev[Nx + 2][Ny + 2][N + 2],
               double* maxTimeKernelFor,
               double* maxTimeKernelBoundary)
{
    double timeFor = MPI_Wtime();
    for (size_t i = 1; i <= Nx; i++)
#pragma omp parallel for 
        for (size_t j = 1; j <= Ny; j++)
        {
            uPrev[i][j][N + 1] = uPrev[i][j][1];
            for (size_t k = 1; k <= N; k++)
            {
                if (xStart + i == N || yStart + j == N)
                    continue;
                uNew[i][j][k] = 2 * uPrev[i][j][k] - uPPrev[i][j][k] + tau * tau * (
                                (uPrev[i - 1][j][k] - 2 * uPrev[i][j][k] + uPrev[i + 1][j][k]) / (hx * hx) +
                                (uPrev[i][j - 1][k] - 2 * uPrev[i][j][k] + uPrev[i][j + 1][k]) / (hy * hy) +
                                (uPrev[i][j][k - 1] - 2 * uPrev[i][j][k] + uPrev[i][j][k + 1]) / (hz * hz));

            }
            uNew[i][j][0] = uNew[i][j][N];
        }
    timeFor = MPI_Wtime() - timeFor;
    if (timeFor > *maxTimeKernelFor)
        *maxTimeKernelFor = timeFor;

    double timeBoundary = MPI_Wtime();
    FillXYBoundary(uNew);
    timeBoundary = MPI_Wtime() - timeBoundary;
    if (timeFor > *maxTimeKernelBoundary)
        *maxTimeKernelBoundary = timeBoundary;
}

int main(int argc, char **argv) {
    SetGlobalVar(argv);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    CreateNewTopology();

	printf("%u %u %u %u\n", rowRank, rowSize, colRank, colSize);
    const double startTime = MPI_Wtime();
    double timeInit = 0, maxTimeKernel = 0, maxTimeExchange = 0, maxTimeKernelFor = 0, maxTimeKernelBoundary = 0;
    {
        timeInit = MPI_Wtime();
        InitFirstSteps((double (*)[Ny + 2][N + 2]) u[0], (double (*)[Ny + 2][N + 2]) u[1]);
        timeInit = MPI_Wtime() - timeInit;

        for (size_t step = 2; step <= 20; step++)
        {
            double timeKernel = MPI_Wtime();
            OMPKernel((double (*)[Ny + 2][N + 2]) u[step % 3],
                      (double (*)[Ny + 2][N + 2]) u[(step - 1) % 3],
                      (double (*)[Ny + 2][N + 2]) u[(step - 2) % 3],
                      &maxTimeKernelFor,
                      &maxTimeKernelBoundary);
            timeKernel = MPI_Wtime() - timeKernel;
            if (timeKernel > maxTimeKernel)
                maxTimeKernel = timeKernel;

            double timeExchange = MPI_Wtime();
            ExchangeData((double (*)[Ny + 2][N + 2]) u[step % 3]);
            timeExchange = MPI_Wtime() - timeExchange;
            if (timeExchange > maxTimeExchange)
                maxTimeExchange = timeExchange;
        }
    }

    double eps = CalculateError(20, (double (*)[Ny + 2][N + 2]) u[20 % 3]);
    double maxEps = 0;
    MPI_Reduce(&eps, &maxEps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double time = MPI_Wtime() - startTime;
    double maxTime = 0;
    MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (wrank == 0)
    {
        printf("Error: %.10lf Time: %lf\n", maxEps, maxTime);
        printf("Init: %.10lf Kernel: %.10lf Exchange: %.10lf\n", timeInit, maxTimeKernel, maxTimeExchange);
        printf("KernelFor: %.10lf KernelBoundary: %.10lf KernelExchange: %.10lf\n", maxTimeKernelFor, maxTimeKernelBoundary, maxTimeKernel - maxTimeKernelFor - maxTimeKernelBoundary);
    }
    FreeData();
    MPI_Finalize();
    return 0;
}
