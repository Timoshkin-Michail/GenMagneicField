#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "VectorFunction.h"

VectorFunction::VectorFunction(const Modes mode, const ptrdiff_t N, const double TAU, const double ETA,
                               const double left_range, const double right_range,
                               const ptrdiff_t alloc_local, const ptrdiff_t local_dim0_size,
                               const ptrdiff_t local_dim0_start,
                               const int rank, const int size) :
        mode{mode},
        N{N},
        TAU{TAU},
        ETA{ETA},
        INDEX_LEFT{-N / 2 + 1},
        INDEX_RIGHT{N / 2 + 1},
        LEFT_RANGE{left_range},
        RIGHT_RANGE{right_range},
        alloc_local{alloc_local}, local_dim0_size{local_dim0_size}, local_dim0_start{local_dim0_start},
        NORMALIZATION_CONSTANT{std::sqrt(N * N * N)},
        rank{rank},
        size{size} {
    switch (mode) {
        case Modes::COMPLEXES_REALS:
            for (int i = 0; i < 3; ++i) {
                vec_r[i] = fftw_alloc_real(2 * alloc_local);
                vec_c[i] = fftw_alloc_complex(alloc_local);
                forward_plan[i] = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[i], vec_c[i], MPI_COMM_WORLD, FFTW_MEASURE);
                backward_plan[i] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[i], vec_r[i], MPI_COMM_WORLD, FFTW_MEASURE);
            }
            break;
        case Modes::COMPLEXES:
            for (int i = 0; i < 3; ++i) {
                vec_c[i] = fftw_alloc_complex(alloc_local);
            }
            break;
        case Modes::REALS:
            for (int i = 0; i < 3; ++i) {
                vec_r[i] = fftw_alloc_real(2 * alloc_local);
            }
            break;
        case Modes::ONE_COMPLEX:
            vec_c[0] = fftw_alloc_complex(alloc_local);
    }
    indexes = new ptrdiff_t[N];
    for (ptrdiff_t i = 0; i <= N / 2; ++i) {
        indexes[i] = i;
    }
    for (ptrdiff_t i = N / 2 + 1; i < N; ++i) {
        indexes[i] = i - N;
    }
}

VectorFunction::~VectorFunction() {
    delete[] indexes;
    switch (mode) {
        case Modes::COMPLEXES_REALS:
            for (int i = 0; i < 3; ++i) {
                fftw_free(vec_r[i]);
                fftw_free(vec_c[i]);
                fftw_destroy_plan(forward_plan[i]);
                fftw_destroy_plan(backward_plan[i]);
            }
            break;
        case Modes::COMPLEXES:
            for (int i = 0; i < 3; ++i) {
                fftw_free(vec_c[i]);
            }
            break;
        case Modes::REALS:
            for (int i = 0; i < 3; ++i) {
                fftw_free(vec_r[i]);
            }
            break;
        case Modes::ONE_COMPLEX:
            fftw_free(vec_c[0]);
    }
}

void
VectorFunction::initialize(std::vector<std::function<double(const double, const double, const double)>> functions) {
    double cur_x, cur_y, cur_z;
    ptrdiff_t idx;
    for (ptrdiff_t q = 0; q < 3; ++q) {
        for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
            for (ptrdiff_t j = 0; j < N; ++j) {
                for (ptrdiff_t k = 0; k < N; ++k) {
                    idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
                    cur_x = LEFT_RANGE + (RIGHT_RANGE - LEFT_RANGE) * (local_dim0_start + i) / N;
                    cur_y = LEFT_RANGE + (RIGHT_RANGE - LEFT_RANGE) * j / N;
                    cur_z = LEFT_RANGE + (RIGHT_RANGE - LEFT_RANGE) * k / N;
                    vec_r[q][idx] = functions[q](cur_x, cur_y, cur_z);
                }
            }
        }
    }
}

void VectorFunction::forward_transformation() {
    for (int i = 0; i < 2; i++) {
        fftw_execute(forward_plan[i]);
    }

    ptrdiff_t idx;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                idx = (i * N + j) * (N / 2 + 1) + k;
                for (int t = 0; t < 2; t++) {
                    vec_c[t][idx][0] /= NORMALIZATION_CONSTANT;
                    vec_c[t][idx][1] /= NORMALIZATION_CONSTANT;
                }
            }
        }
    }
}

void VectorFunction::backward_transformation() {
    for (int i = 0; i < 2; i++) {
        fftw_execute(backward_plan[i]);
    }

    ptrdiff_t idx;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < N; ++k) {
                idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
                for (int t = 0; t < 2; t++) {
                    vec_r[t][idx] /= NORMALIZATION_CONSTANT;
                }
            }
        }
    }
}

void VectorFunction::cross_product(const VectorFunction &velocity_field, const VectorFunction &magnetic_field) {
    ptrdiff_t idx;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < N; ++k) {
                idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
                vec_r[0][idx] = velocity_field.vec_r[1][idx] * magnetic_field.vec_r[2][idx] -
                                velocity_field.vec_r[2][idx] * magnetic_field.vec_r[1][idx];

                vec_r[1][idx] = velocity_field.vec_r[2][idx] * magnetic_field.vec_r[0][idx] -
                                velocity_field.vec_r[0][idx] * magnetic_field.vec_r[2][idx];

                vec_r[2][idx] = velocity_field.vec_r[0][idx] * magnetic_field.vec_r[1][idx] -
                                velocity_field.vec_r[1][idx] * magnetic_field.vec_r[0][idx];
            }
        }
    }

}

void VectorFunction::divergence(const VectorFunction &source_field) {
    double coef_0, coef_1, coef_2;
    ptrdiff_t idx;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                idx = (i * N + j) * (N / 2 + 1) + k;
                coef_0 = static_cast<double>(indexes[local_dim0_start + i]);
                coef_1 = static_cast<double>(indexes[j]);
                coef_2 = static_cast<double>(k);
                vec_c[0][idx][0] = -(coef_0 * source_field.vec_c[0][idx][1] +
                                     coef_1 * source_field.vec_c[1][idx][1] +
                                     coef_2 * source_field.vec_c[2][idx][1]);

                vec_c[0][idx][1] = coef_0 * source_field.vec_c[0][idx][0] +
                                   coef_1 * source_field.vec_c[1][idx][0] +
                                   coef_2 * source_field.vec_c[2][idx][0];
            }
        }
    }
}

void VectorFunction::rotor(const VectorFunction &velocity_field, const VectorFunction &magnetic_field) {
    cross_product(velocity_field, magnetic_field);

    forward_transformation();

    double idx_0_l, idx_0_r, idx_1_l, idx_1_r, idx_2_l, idx_2_r;
    double val_0_r, val_0_i, val_1_r, val_1_i, val_2_r, val_2_i;
    ptrdiff_t idx;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                idx = (i * N + j) * (N / 2 + 1) + k;
                idx_0_l = indexes[j];
                idx_0_r = k;
                idx_1_l = k;
                idx_1_r = indexes[local_dim0_start + i];
                idx_2_l = indexes[local_dim0_start + i];
                idx_2_r = indexes[j];
                val_0_r = vec_c[0][idx][0];
                val_0_i = vec_c[0][idx][1];
                val_1_r = vec_c[1][idx][0];
                val_1_i = vec_c[1][idx][1];
                val_2_r = vec_c[2][idx][0];
                val_2_i = vec_c[2][idx][1];
                vec_c[0][idx][0] = -val_2_i * idx_0_l + val_1_i * idx_0_r;
                vec_c[0][idx][1] = val_2_r * idx_0_l - val_1_r * idx_0_r;
                vec_c[1][idx][0] = -val_0_i * idx_1_l + val_2_i * idx_1_r;
                vec_c[1][idx][1] = val_0_r * idx_1_l - val_2_r * idx_1_r;
                vec_c[2][idx][0] = -val_1_i * idx_2_l + val_0_i * idx_2_r;
                vec_c[2][idx][1] = val_1_r * idx_2_l - val_0_r * idx_2_r;
            }
        }
    }
}

void VectorFunction::do_step(const VectorFunction &velocity_field, VectorFunction &rotor_field) {

    rotor_field.rotor(velocity_field, *this);

    ptrdiff_t idx;
    double k_0, k_1, k_2, sum_k;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                idx = (i * N + j) * (N / 2 + 1) + k;
                k_0 = static_cast<double>(indexes[local_dim0_start + i]);
                k_1 = static_cast<double>(indexes[j]);
                k_2 = static_cast<double>(k);
                sum_k = k_0 * k_0 + k_1 * k_1 + k_2 * k_2;
                vec_c[0][idx][0] += (-ETA * sum_k * vec_c[0][idx][0] + rotor_field.vec_c[0][idx][0]) * TAU;
                vec_c[0][idx][1] += (-ETA * sum_k * vec_c[0][idx][1] + rotor_field.vec_c[0][idx][1]) * TAU;
                vec_c[1][idx][0] += (-ETA * sum_k * vec_c[1][idx][0] + rotor_field.vec_c[1][idx][0]) * TAU;
                vec_c[1][idx][1] += (-ETA * sum_k * vec_c[1][idx][1] + rotor_field.vec_c[1][idx][1]) * TAU;
                vec_c[2][idx][0] += (-ETA * sum_k * vec_c[2][idx][0] + rotor_field.vec_c[2][idx][0]) * TAU;
                vec_c[2][idx][1] += (-ETA * sum_k * vec_c[2][idx][1] + rotor_field.vec_c[2][idx][1]) * TAU;
            }
        }
    }

}

double VectorFunction::energy_fourier() const {
    double energy = 0.;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            ptrdiff_t idx = (i * N + j) * (N / 2 + 1);
            energy += 0.5 * (vec_c[0][idx][0] * vec_c[0][idx][0] + vec_c[0][idx][1] * vec_c[0][idx][1] +
                             vec_c[1][idx][0] * vec_c[1][idx][0] + vec_c[1][idx][1] * vec_c[1][idx][1] +
                             vec_c[2][idx][0] * vec_c[2][idx][0] + vec_c[2][idx][1] * vec_c[2][idx][1]);
            for (ptrdiff_t k = 1; k < INDEX_RIGHT; ++k) {
                ++idx;
                energy += (vec_c[0][idx][0] * vec_c[0][idx][0] + vec_c[0][idx][1] * vec_c[0][idx][1] +
                           vec_c[1][idx][0] * vec_c[1][idx][0] + vec_c[1][idx][1] * vec_c[1][idx][1] +
                           vec_c[2][idx][0] * vec_c[2][idx][0] + vec_c[2][idx][1] * vec_c[2][idx][1]);
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return energy;
}

void VectorFunction::correction(VectorFunction &vector_function) {
    vector_function.divergence(*this);

    ptrdiff_t idx;
    double local_max = 0.;
    for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
        for (ptrdiff_t j = 0; j < N; ++j) {
            for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                idx = (i * N + j) * (N / 2 + 1) + k;
                local_max = std::max(local_max, std::sqrt(
                        vector_function.vec_c[0][idx][0] * vector_function.vec_c[0][idx][0] +
                        vector_function.vec_c[0][idx][1] * vector_function.vec_c[0][idx][1]));
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (local_max >= EPSILON) {
        double k_0, k_1, sum_ks;
        for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
            for (ptrdiff_t j = 0; j < N; ++j) {
                for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
                    idx = (i * N + j) * (N / 2 + 1) + k;
                    k_0 = indexes[local_dim0_start + i];
                    k_1 = indexes[j];

                    if (std::abs(k_0) + std::abs(k_1) + k < EPSILON) {
                        for (int t = 0; t < 2; t++) {
                            vec_c[t][idx][0] = 0;
                            vec_c[t][idx][1] = 0;
                        }
                    } else {
                        sum_ks = k_0 * k_0 + k_1 * k_1 + k * k;
                        vec_c[0][idx][0] += -k_0 * vector_function.vec_c[0][idx][1] / sum_ks;
                        vec_c[0][idx][1] += k_0 * vector_function.vec_c[0][idx][0] / sum_ks;
                        vec_c[1][idx][0] += -k_1 * vector_function.vec_c[0][idx][1] / sum_ks;
                        vec_c[1][idx][1] += k_1 * vector_function.vec_c[0][idx][0] / sum_ks;
                        vec_c[2][idx][0] += -k * vector_function.vec_c[0][idx][1] / sum_ks;
                        vec_c[2][idx][1] += k * vector_function.vec_c[0][idx][0] / sum_ks;
                    }
                }
            }
        }
    }
}