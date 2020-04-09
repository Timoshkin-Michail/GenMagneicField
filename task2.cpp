#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

#include <mpi.h>
#include <fftw3-mpi.h>
#include <chrono>
#include "VectorFunction.h"

void initialize(VectorFunction &magnetic_field, VectorFunction &velocity_field) {
    const double coef = 2 / std::sqrt(3);
    std::vector<std::function<double(const double, const double, const double)>> velocity_functions;
    velocity_functions.emplace_back([=](const double, const double x2, const double x3) {
        return coef * std::sin(x2) * std::cos(x3);
    });
    velocity_functions.emplace_back([=](const double x1, const double, const double x3) {
        return coef * std::sin(x3) * std::cos(x1);
    });
    velocity_functions.emplace_back([=](const double x1, const double x2, const double) {
        return coef * std::sin(x1) * std::cos(x2);
    });

    velocity_field.initialize(velocity_functions);

    std::vector<std::function<double(const double, const double, const double)>> magnetic_functions;
    magnetic_functions.emplace_back([](const double x1, const double x2, const double x3) {
        return std::sin(x1 - 2 * x2 + 3 * x3);
    });
    magnetic_functions.emplace_back([](const double x1, const double x2, const double x3) {
        return std::cos(-x1 - x3 + 5 * x2);
    });
    magnetic_functions.emplace_back([](const double x1, const double x2, const double x3) {
        return std::sin(-3 * x1 - x2 + x3);
    });

    magnetic_field.initialize(magnetic_functions);
    magnetic_field.forward_transformation();
}


int main(int argc, char *argv[]) {
    const double DOUBLE_PI = std::acos(-1) * 2;
    const long time_steps = std::strtol(argv[1], nullptr, 0);
    const ptrdiff_t N = std::strtol(argv[2], nullptr, 0);
    const double tau = std::strtod(argv[3], nullptr),
            eta = std::strtod(argv[4], nullptr);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fftw_mpi_init();

    ptrdiff_t alloc_local, local_dim0_size, local_dim0_start;
    alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD, &local_dim0_size, &local_dim0_start);
    //std::ofstream ofile(argv[5]);
    const auto begin = std::chrono::steady_clock::now();

    {
        VectorFunction magnetic_field{Modes::COMPLEXES_REALS, N, tau, eta, 0, DOUBLE_PI, alloc_local, local_dim0_size,
                                      local_dim0_start, rank, size},
                velocity_field{Modes::REALS, N, tau, eta, 0, DOUBLE_PI, alloc_local, local_dim0_size, local_dim0_start,
                               rank, size},
                rotor_field{Modes::COMPLEXES_REALS, N, tau, eta, 0, DOUBLE_PI, alloc_local, local_dim0_size,
                            local_dim0_start, rank, size},
                auxiliary_field{Modes::ONE_COMPLEX, N, tau, eta, 0, DOUBLE_PI, alloc_local, local_dim0_size,
                                local_dim0_start, rank, size};

        initialize(magnetic_field, velocity_field);
        magnetic_field.forward_transformation();

        double cur_energy;
        for (int step = 0; step < time_steps; ++step) {
            magnetic_field.correction(auxiliary_field); // div = 0
            magnetic_field.backward_transformation();
            magnetic_field.do_step(velocity_field, rotor_field);
            cur_energy = magnetic_field.energy_fourier();
            if (cur_energy > 1e285) {
                break;
            }
        }
    }
    //ofile.close();
    const auto end = std::chrono::steady_clock::now();
    long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &elapsed_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&elapsed_ms, nullptr, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        std::cout << elapsed_ms << '\n';
        //ofile << cur_energy << '\n';
    }
    MPI_Finalize();
    return 0;
}


