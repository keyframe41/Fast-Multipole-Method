#ifndef MULTIPOLE_EXPANSION_HPP
#define MULTIPOLE_EXPANSION_HPP

#include "math_utils.hpp"
#include <vector>
#include <complex>
#include <span>

namespace fmm {

class MultipoleExpansion : public SeriesExpansion {
public:
    MultipoleExpansion() : SeriesExpansion() {}

    template<typename Iterator>
    MultipoleExpansion(const Complex &center, int order, Iterator begin, Iterator end) 
        : SeriesExpansion(center, order) 
    {
        for (auto it = begin; it != end; it++) {
            const auto &s = *it;
            Complex q_i = s.q;
            Complex z_i_to_z_c = s.position - this->center; 
            Complex temp_pow = z_i_to_z_c;

            (*this)(0) += q_i;
            for (size_t j = 1; j <= order; j++) {
                (*this)(j) -= q_i * temp_pow; 
                temp_pow *= z_i_to_z_c;
            }
        }

        for (size_t i = 1; i <= order; i++) (*this)(i) /= static_cast<double>(i); 
    }

    MultipoleExpansion(const Complex &center, std::span<const MultipoleExpansion* const> expansions)
        : SeriesExpansion(center, expansions.empty() ? 0 : expansions[0]->order) 
    {
        for (const auto *me : expansions) {
            Complex disp = me->center - this->center;
            std::array<Complex, MAXP> shifted = me->shift(disp);

            for (size_t i = 0; i <= this->order; i++) 
                this->coefficients[i] += shifted[i];
        }
    }

    std::array<Complex, MAXP> shift (const Complex &disp) const {
        std::array<Complex, MAXP> shifted;
        shifted.fill(Complex{0.0, 0.0});

        std::array<Complex, MAXP> shift_pow;
        shift_pow[0] = Complex{1.0, 0.0};

        for (size_t i = 1; i <= this->order; i++) shift_pow[i] = shift_pow[i - 1] * disp; // disp is ALWAYS one of 4 vectors, precomputation maybe possible
        
        double Q = (*this)(0).real();
        shifted[0] = Q;

        for (size_t r = 1; r <= this->order; r++) {
            shifted[r] = -Q * shift_pow[r] / static_cast<double>(r);
            for (int k = 1; k <= r; k++) {
                shifted[r] += (*this)(k) * shift_pow[r - k] * SeriesExpansion::binom_table(r - 1, k - 1);
            }
        }

        return shifted;
    }

    double evaluatePotential (const Complex &point) const override {
        Complex z_to_z_c = point - this->center;
        Complex z_to_z_c_inv = fast_inverse(z_to_z_c);

        Complex poly_sum = (*this)(this->order);
        for (size_t i = this->order - 1; i >= 1; i--) {
            poly_sum = poly_sum * z_to_z_c_inv + (*this)(i);
        }
        poly_sum *= z_to_z_c_inv;

        double Q = (*this)(0).real();
        Complex result = Q * std::log(z_to_z_c) + poly_sum;

        return result.real();
    }

    Complex evaluateForce (const Complex &point) const override {
        Complex z_to_z_c = point - this->center;
        Complex z_to_z_c_inv = fast_inverse(z_to_z_c);

        Complex poly_sum = static_cast<double>(this->order) * (*this)(this->order);
        for (size_t i = this->order - 1; i >= 1; i--) {
            poly_sum = poly_sum * z_to_z_c_inv + static_cast<double>(i) * (*this)(i);
        }

        double Q = (*this)(0).real();
        Complex result = (Q - poly_sum * z_to_z_c_inv) * z_to_z_c_inv;

        return Complex(-result.real(), result.imag());
    }
};

}

#endif

