#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include "adaptive_quadtree.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace fmm {

std::vector<Complex> computeExactForces(const std::vector<Source>& sources) { // Naive method
    std::vector<Complex> exact_forces(sources.size(), Complex{0.0, 0.0});
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < sources.size(); ++i) {
        double tx = sources[i].position.real();
        double ty = sources[i].position.imag();
        double fx = 0.0;
        double fy = 0.0;

        for (size_t j = 0; j < sources.size(); ++j) {
            if (i == j) continue;
            const auto& src = sources[j];
            double dx = tx - src.position.real();
            double dy = ty - src.position.imag();
            double r2 = dx * dx + dy * dy;
            
            double factor = -src.q / r2;
            fx += factor * dx;
            fy += factor * dy;
        }
        exact_forces[i] = Complex{fx, fy};
    }
    return exact_forces;
}


struct ErrorData {
    double l2_relative_error;  
    double max_absolute_error;
    double mean_absolute_error;
};

ErrorData evaluateSimulationError(const std::vector<Source>& sources, const std::vector<Complex>& approx_forces, size_t sample_size = 1000) {
    size_t N = sources.size();
    size_t M = std::min(sample_size, N);

    double sum_diff_sq = 0.0;
    double sum_exact_sq = 0.0;
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;

    size_t gap = N / M;
    if (gap == 0) gap = 1;

    #pragma omp parallel for reduction(+:sum_diff_sq, sum_exact_sq, sum_abs_err) reduction(max:max_abs_err)
    for (size_t k = 0; k < M; ++k) {
        size_t i = k * gap;
        if (i >= N) continue;

        const auto& target = sources[i];
        double tx = target.position.real();
        double ty = target.position.imag();

        double exact_fx = 0.0;
        double exact_fy = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const auto& src = sources[j];
            double dx = tx - src.position.real();
            double dy = ty - src.position.imag();
            double r2 = dx * dx + dy * dy;
            
            double factor = -src.q / r2;
            exact_fx += factor * dx;
            exact_fy += factor * dy;
        }

        double approx_fx = approx_forces[i].real();
        double approx_fy = approx_forces[i].imag();

        double diff_x = approx_fx - exact_fx;
        double diff_y = approx_fy - exact_fy;
        double diff_sq = diff_x * diff_x + diff_y * diff_y;
        double exact_sq = exact_fx * exact_fx + exact_fy * exact_fy;

        double abs_err = std::sqrt(diff_sq);

        sum_diff_sq += diff_sq;
        sum_exact_sq += exact_sq;
        sum_abs_err += abs_err;
        
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }

    ErrorData report;
    report.l2_relative_error = (sum_exact_sq > 1e-12) ? std::sqrt(sum_diff_sq / sum_exact_sq) : 0.0;
    report.max_absolute_error = max_abs_err;
    report.mean_absolute_error = sum_abs_err / M;

    return report;
}

} // namespace fmm

#endif // DIAGNOSTICS_HPP