#include <mpi.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>
#include <functional>
#include <fftw3-mpi.h>

typedef std::vector<std::function<double(const double, const double, const double)>> vector_function;

struct Configuration {
    const ptrdiff_t N;
    const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
    const double RANGE_LEFT, RANGE_RIGHT;
    const ptrdiff_t alloc_local, local_n0, local_0_start;
    double *indeces;
    int rank, size;
    vector_function b;
    std::vector<vector_function> d_b;
    vector_function rot_b;

    Configuration(const ptrdiff_t N_,
                  const double rng_left_, const double rng_right_,
                  const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_, const ptrdiff_t local_0_start_,
                  const int rank_, const int size_) :
            N{N_},
            INDEX_LEFT{-N / 2 + 1}, INDEX_RIGHT{N / 2 + 1}, //  [left, right)
            RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_},
            alloc_local{alloc_local_}, local_n0{local_n0_}, local_0_start{local_0_start_},
            rank{rank_}, size{size_} {
        indeces = new double[N];
        for (ptrdiff_t i = 0; i <= N / 2; ++i) {
            indeces[i] = i;
        }
        for (ptrdiff_t i = N / 2 + 1; i < N; ++i) {
            indeces[i] = (double) i - N;
        }

        //b1
        b.emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(x1 - 2 * x2 + 3 * x3));
        });
        //b2
        b.emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::cos(-x1 - x3));
        });
        //b3
        b.emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(-3 * x1 - x2 + x3));
        });


        d_b = std::vector<vector_function>(3);
        // d_b1
        d_b[0].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * std::cos(1 * x1 - 2 * x2 + 3 * x3);
        }); // derivative of the first argument
        d_b[0].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * -2 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
        }); // derivative of the second argument
        d_b[0].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 3 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
        }); // derivative of the third argument

        // d_b2
        d_b[1].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3);
        }); // derivative of the first argument
        d_b[1].emplace_back([](const double x1, const double x2, const double x3) {
            return 0;
        }); // derivative of the second argument
        d_b[1].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3);
        }); // derivative of the third argument

        // d_b3
        d_b[2].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(-3 * x1 - x2 + x3)) * -3 * std::cos(-3 * x1 - x2 + x3);
        }); // derivative of the first argument
        d_b[2].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(-3 * x1 - x2 + x3)) * -std::cos(-3 * x1 - x2 + x3);
        }); // derivative of the second argument
        d_b[2].emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(-3 * x1 - x2 + x3)) * std::cos(-3 * x1 - x2 + x3);
        }); // derivative of the third argument

        // rotor b
        rot_b.emplace_back([](const double x1, const double x2, const double x3) {
            return -std::exp(std::sin(-3 * x1 - x2 + x3)) * std::cos(-3 * x1 - x2 + x3) -
                   std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3);
        });
        rot_b.emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 3 * std::cos(x1 - 2 * x2 + 3 * x3) +
                   std::exp(std::sin(-3 * x1 - x2 + x3)) * 3 * std::cos(-3 * x1 - x2 + x3);
        });
        rot_b.emplace_back([](const double x1, const double x2, const double x3) {
            return std::exp(std::cos(-x1 - x3)) * std::sin(-x1 - x3) +
                   std::exp(std::sin(x1 - 2 * x2 + 3 * x3)) * 2 * std::cos(x1 - 2 * x2 + 3 * x3);
        });

    }

    ~Configuration() {
        delete[] indeces;
    }
};


void derivative_of_function(fftw_complex *ptr, const Configuration &config, const int num_of_dimension) {
    double coef;
    for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < config.N; ++j) {
            for (ptrdiff_t k = 0; k < config.INDEX_RIGHT; ++k) { //  [0, N/2 + 1)
                std::swap(ptr[(i * config.N + j) * (config.N / 2 + 1) + k][0],
                          ptr[(i * config.N + j) * (config.N / 2 + 1) + k][1]);
                if (num_of_dimension == 0) {
                    coef = config.indeces[config.local_0_start + i];
                } else if (num_of_dimension == 1) {
                    coef = config.indeces[j];
                } else {
                    coef = k;
                }
                ptr[(i * config.N + j) * (config.N / 2 + 1) + k][0] *= -coef;
                ptr[(i * config.N + j) * (config.N / 2 + 1) + k][1] *= coef;
            }
        }
    }
}

void rotor(fftw_complex *rot,
           const fftw_complex *cross_p_l, const fftw_complex *cross_p_r,
           const Configuration &config, const int num_of_dimension) {
    double coef_l, coef_r;
    for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < config.N; ++j) {
            for (ptrdiff_t k = 0; k < config.INDEX_RIGHT; ++k) {
                const ptrdiff_t idx = (i * config.N + j) * (config.N / 2 + 1) + k;
                if (num_of_dimension == 0) {
                    coef_l = config.indeces[j];
                    coef_r = k;
                } else if (num_of_dimension == 1) {
                    coef_l = k;
                    coef_r = config.indeces[config.local_0_start + i];
                } else {
                    coef_l = config.indeces[config.local_0_start + i];
                    coef_r = config.indeces[j];
                }
                rot[idx][0] = -cross_p_l[idx][1] * coef_l + cross_p_r[idx][1] * coef_r;
                rot[idx][1] = cross_p_l[idx][0] * coef_l - cross_p_r[idx][0] * coef_r;
            }
        }
    }
}


double field_energy_physical(const double *ptr_1,
                             const double *ptr_2,
                             const double *ptr_3,
                             const Configuration &config) {
    double energy = 0.;
    for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < config.N; ++j) {
            for (ptrdiff_t k = 0; k < config.N; ++k) {
                const ptrdiff_t idx = (i * config.N + j) * (2 * (config.N / 2 + 1)) + k;
                energy += ptr_1[idx] * ptr_1[idx] +
                          ptr_2[idx] * ptr_2[idx] +
                          ptr_3[idx] * ptr_3[idx];
            }
        }
    }
    energy /= (2.0 * config.N * config.N * config.N);
    return energy;
}

double field_energy_fourier(const fftw_complex *ptr_1,
                            const fftw_complex *ptr_2,
                            const fftw_complex *ptr_3,
                            const Configuration &config) {
    double energy = 0.;
    for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
        for (ptrdiff_t j = 0; j < config.N; ++j) {
            ptrdiff_t idx = (i * config.N + j) * (config.N / 2 + 1);
            energy += 0.5 * (ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
                             ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
                             ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
            for (ptrdiff_t k = 1; k < config.INDEX_RIGHT; ++k) {
                ++idx;
                energy += (ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
                           ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
                           ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
            }
        }
    }
    energy /= (double) (config.N * config.N * config.N);
    return energy;
}


void fill_real(double *vec[3], const Configuration &config) {
    for (ptrdiff_t dir = 0; dir < 3; ++dir) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            const double x1 = config.RANGE_RIGHT * (config.local_0_start + (double) i) / config.N;
            for (ptrdiff_t j = 0; j < config.N; ++j) {
                const double x2 = config.RANGE_RIGHT * j / config.N;
                for (ptrdiff_t k = 0; k < config.N; ++k) {
                    const double x3 = config.RANGE_RIGHT * k / config.N;
                    vec[dir][(i * config.N + j) * (2 * (config.N / 2 + 1)) + k] = config.b[dir](x1, x2, x3);
                }
            }
        }
    }
}

void test_derivative(const Configuration &config) {
    const ptrdiff_t N = config.N;
    fftw_plan forward_plan[3], backward_plan[3];
    double *vec_r[3];
    fftw_complex *vec_c[3];
    for (int dir = 0; dir < 3; ++dir) {
        vec_r[dir] = fftw_alloc_real(2 * config.alloc_local);
        vec_c[dir] = fftw_alloc_complex(config.alloc_local);
        forward_plan[dir] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[dir], vec_c[dir], MPI_COMM_WORLD, FFTW_MEASURE);
        backward_plan[dir] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[dir], vec_r[dir], MPI_COMM_WORLD, FFTW_MEASURE);
    }

    for (int num = 0; num < 3; ++num) {
        fill_real(vec_r, config);
        fftw_execute(forward_plan[0]);
        fftw_execute(forward_plan[1]);
        fftw_execute(forward_plan[2]);

        derivative_of_function(vec_c[0], config, num);
        derivative_of_function(vec_c[1], config, num);
        derivative_of_function(vec_c[2], config, num);

        fftw_execute(backward_plan[0]);
        fftw_execute(backward_plan[1]);
        fftw_execute(backward_plan[2]);

        for (auto &dir : vec_r) {
            for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
                for (ptrdiff_t j = 0; j < config.N; ++j) {
                    for (ptrdiff_t k = 0; k < config.N; ++k) {
                        dir[(i * N + j) * (2 * (N / 2 + 1)) + k] /= 1.0 * N * N * N;
                    }
                }
            }
        }
        double max_err = 0.;
        for (int dir = 0; dir < 3; ++dir) {
            for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
                const double x1 = config.RANGE_RIGHT * (config.local_0_start + (double) i) / config.N;
                for (ptrdiff_t j = 0; j < config.N; ++j) {
                    const double x2 = config.RANGE_RIGHT * j / config.N;
                    for (ptrdiff_t k = 0; k < config.N; ++k) {
                        const double x3 = config.RANGE_RIGHT * k / config.N;
                        max_err = std::max(max_err, std::abs(vec_r[dir][(i * N + j) * (2 * (N / 2 + 1)) + k] -
                                                             config.d_b[dir][num](x1, x2, x3)));
                    }
                }
            }
        }

        if (config.rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&max_err, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        if (config.rank == 0) {
            if (num == 0) {
                std::cout << "Max error in derivative calculation for direction:" << '\n';
            }
            std::cout << "b" << num + 1 << "  " << max_err << '\n';
        }
    }

    for (int dir = 0; dir < 3; ++dir) {
        fftw_free(vec_r[dir]);
        fftw_free(vec_c[dir]);
        fftw_destroy_plan(forward_plan[dir]);
        fftw_destroy_plan(backward_plan[dir]);
    }
}

void test_rotor(const Configuration &config) {
    const ptrdiff_t N = config.N;
    fftw_plan forward_plan[3];
    double *vec_r[3];
    fftw_complex *vec_c[3];
    for (int dir = 0; dir < 3; ++dir) {
        vec_r[dir] = fftw_alloc_real(2 * config.alloc_local);
        vec_c[dir] = fftw_alloc_complex(config.alloc_local);
        forward_plan[dir] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[dir], vec_c[dir], MPI_COMM_WORLD, FFTW_MEASURE);
    }
    fill_real(vec_r, config);


    fftw_complex *rotor_c[3];
    double *rotor_r[3];
    fftw_plan rot_c_to_r[3];
    for (int dir = 0; dir < 3; ++dir) {
        rotor_r[dir] = fftw_alloc_real(2 * config.alloc_local);
        rotor_c[dir] = fftw_alloc_complex(config.alloc_local);
        rot_c_to_r[dir] = fftw_mpi_plan_dft_c2r_3d(N, N, N, rotor_c[dir], rotor_r[dir], MPI_COMM_WORLD, FFTW_MEASURE);
    }

    fftw_execute(forward_plan[0]);
    fftw_execute(forward_plan[1]);
    fftw_execute(forward_plan[2]);

    for (auto &dir : vec_c) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < config.N; ++j) {
                for (ptrdiff_t k = 0; k < config.INDEX_RIGHT; ++k) {
                    dir[(i * N + j) * (N / 2 + 1) + k][0] /= N * std::sqrt(N);
                    dir[(i * N + j) * (N / 2 + 1) + k][1] /= N * std::sqrt(N);
                }
            }
        }
    }

    rotor(rotor_c[0], vec_c[2], vec_c[1], config, 0);
    rotor(rotor_c[1], vec_c[0], vec_c[2], config, 1);
    rotor(rotor_c[2], vec_c[1], vec_c[0], config, 2);

    fftw_execute(rot_c_to_r[0]);
    fftw_execute(rot_c_to_r[1]);
    fftw_execute(rot_c_to_r[2]);

    for (auto &dir : rotor_r) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < config.N; ++j) {
                for (ptrdiff_t k = 0; k < config.N; ++k) {
                    dir[(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * std::sqrt(N);
                }
            }
        }
    }

    double max_diff = 0.;
    for (int dir = 0; dir < 3; ++dir) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            const double x1 = config.RANGE_RIGHT * (config.local_0_start + (double) i) / config.N;
            for (ptrdiff_t j = 0; j < config.N; ++j) {
                const double x2 = config.RANGE_RIGHT * j / config.N;
                for (ptrdiff_t k = 0; k < config.N; ++k) {
                    const double x3 = config.RANGE_RIGHT * k / config.N;
                    max_diff = std::max(max_diff, std::abs(
                            rotor_r[dir][(i * N + j) * (2 * (N / 2 + 1)) + k] - config.rot_b[dir](x1, x2, x3)));
                }
            }
        }
    }

    if (config.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (config.rank == 0) {
        std::cout << "Max error in rotor calculation:" << '\n';
        std::cout << max_diff << '\n';
    }
    for (int q = 0; q < 3; ++q) {
        fftw_free(vec_r[q]);
        fftw_free(vec_c[q]);
        fftw_free(rotor_r[q]);
        fftw_free(rotor_c[q]);
        fftw_destroy_plan(rot_c_to_r[q]);
        fftw_destroy_plan(forward_plan[q]);
    }
}

void test_divergence(const Configuration &config) {
    const ptrdiff_t N = config.N;
    fftw_plan forward_plan[3], backward_plan[3];
    double *vec_r[3];
    fftw_complex *vec_c[3];
    for (int q = 0; q < 3; ++q) {
        vec_r[q] = fftw_alloc_real(2 * config.alloc_local);
        vec_c[q] = fftw_alloc_complex(config.alloc_local);
        forward_plan[q] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
        backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
    }

    fill_real(vec_r, config);

    fftw_execute(forward_plan[0]);
    fftw_execute(forward_plan[1]);
    fftw_execute(forward_plan[2]);

    derivative_of_function(vec_c[0], config, 0);
    derivative_of_function(vec_c[1], config, 1);
    derivative_of_function(vec_c[2], config, 2);

    fftw_execute(backward_plan[0]);
    fftw_execute(backward_plan[1]);
    fftw_execute(backward_plan[2]);

    for (auto &dir : vec_r) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < config.N; ++j) {
                for (ptrdiff_t k = 0; k < config.N; ++k) {
                    dir[(i * N + j) * (2 * (N / 2 + 1)) + k] /= 1.0 * N * N * N;
                }
            }
        }
    }

    double max_diff = 0.;
    for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
        const double x1 = config.RANGE_RIGHT * (config.local_0_start + (double) i) / config.N;
        for (ptrdiff_t j = 0; j < config.N; ++j) {
            const double x2 = config.RANGE_RIGHT * j / config.N;
            for (ptrdiff_t k = 0; k < config.N; ++k) {
                const double x3 = config.RANGE_RIGHT * k / config.N;
                max_diff = std::max(max_diff, std::abs(
                        vec_r[0][(i * N + j) * (2 * (N / 2 + 1)) + k] +
                        vec_r[1][(i * N + j) * (2 * (N / 2 + 1)) + k] +
                        vec_r[2][(i * N + j) * (2 * (N / 2 + 1)) + k] -
                        (config.d_b[0][0](x1, x2, x3) +
                         config.d_b[1][1](x1, x2, x3) +
                         config.d_b[2][2](x1, x2, x3))));
            }
        }
    }

    if (config.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&max_diff, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    if (config.rank == 0) {
        std::cout << "Max error in divergence calculation:" << '\n';
        std::cout << max_diff << '\n';
    }

    for (int q = 0; q < 3; ++q) {
        fftw_free(vec_r[q]);
        fftw_free(vec_c[q]);
        fftw_destroy_plan(forward_plan[q]);
        fftw_destroy_plan(backward_plan[q]);
    }
}

void test_energy(const Configuration &config) {
    const ptrdiff_t N = config.N;
    fftw_plan forward_plan[3], backward_plan[3];
    double *vec_r[3];
    fftw_complex *vec_c[3];
    for (int q = 0; q < 3; ++q) {
        vec_r[q] = fftw_alloc_real(2 * config.alloc_local);
        vec_c[q] = fftw_alloc_complex(config.alloc_local);
        forward_plan[q] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
        backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
    }
    fill_real(vec_r, config);

    double energy = field_energy_physical(vec_r[0], vec_r[1], vec_r[2], config);
    if (config.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (config.rank == 0) {
        std::cout << "Energy in physical space" << '\n';
        std::cout << energy << '\n';
    }

    fftw_execute(forward_plan[0]);
    fftw_execute(forward_plan[1]);
    fftw_execute(forward_plan[2]);

    for (auto &dir : vec_c) {
        for (ptrdiff_t i = 0; i < config.local_n0; ++i) {
            for (ptrdiff_t j = 0; j < N; ++j) {
                for (ptrdiff_t k = 0; k < config.INDEX_RIGHT; ++k) {
                    dir[(i * N + j) * (N / 2 + 1) + k][0] /= N * std::sqrt(N);
                    dir[(i * N + j) * (N / 2 + 1) + k][1] /= N * std::sqrt(N);
                }
            }
        }
    }

    energy = field_energy_fourier(vec_c[0], vec_c[1], vec_c[2], config);
    if (config.rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (config.rank == 0) {
        std::cout << "Energy in Fourier space" << '\n';
        std::cout << energy << '\n';
    }

    for (int q = 0; q < 3; ++q) {
        fftw_free(vec_r[q]);
        fftw_free(vec_c[q]);
        fftw_destroy_plan(forward_plan[q]);
        fftw_destroy_plan(backward_plan[q]);
    }
}


int main(int argc, char *argv[]) {

    const ptrdiff_t N = std::strtol(argv[1], nullptr, 0);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    fftw_mpi_init();

    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD, &local_n0, &local_0_start);
    if (rank == 0) {
        printf("N: %ld\n", N);
        printf("Alloc_local: %ld\n", alloc_local);
    }

    const Configuration config{N, 0, std::acos(-1) * 2, alloc_local, local_n0, local_0_start, rank, size};

    test_derivative(config);
    test_rotor(config);
    test_divergence(config);
    test_energy(config);

    MPI_Finalize();
    return 0;
}
