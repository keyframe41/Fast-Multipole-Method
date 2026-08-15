#ifndef LOCAL_EXPANSION_HPP
#define LOCAL_EXPANSION_HPP

#include "math_utils.hpp"
#include "multipole_expansion.hpp"
#include <vector>
#include <complex>
#include <span>

namespace fmm {


    class LocalExpansion : public SeriesExpansion {
    public:
        LocalExpansion() : SeriesExpansion() {}
        LocalExpansion(const Complex &center, int order) : SeriesExpansion(center, order) {}

        template<typename Iterator>
        LocalExpansion(const Complex &center, int order, Iterator begin, Iterator end) 
            : SeriesExpansion(center, order) 
        {
            for (auto it = begin; it != end; ++it) {
                const auto& s = *it;
                double q_i = s.q;
                Complex z_i_to_z_c = s.position - this->center;
                Complex z_i_to_z_c_inv = fast_inverse(z_i_to_z_c); 
                Complex temp_pow = z_i_to_z_c_inv;

                (*this)(0) += q_i * std::log(-z_i_to_z_c); 
                for (size_t j = 1; j <= order; ++j) {
                    (*this)(j) -= q_i * temp_pow; 
                    temp_pow *= z_i_to_z_c_inv; 
                }
            }

            for (size_t j = 1; j <= order; ++j) (*this)(j) /= static_cast<double>(j);
        }

        LocalExpansion (const Complex &center, const MultipoleExpansion &me) 
            : SeriesExpansion(center, me.order) {
            this->coefficients = multipoleToLocal(me);
        }

        LocalExpansion (const Complex &center, std::span<const MultipoleExpansion* const> mes) 
            : SeriesExpansion(center, mes.empty() ? 0 : mes[0]->order)  {
            for (const auto *me : mes) *this += LocalExpansion(center, *me);
        }

        LocalExpansion (const Complex &center, const LocalExpansion &le)
            : SeriesExpansion(center, le.order) {
            Complex disp = le.center - this->center;
            this->coefficients = le.shift(disp);
        }

        std::array<Complex, MAXP> multipoleToLocal (const MultipoleExpansion &me) const {
            Complex disp = me.center - this->center;
            std::array<Complex, MAXP> coefficients;
            coefficients.fill(Complex{0.0, 0.0});

            Complex disp_inv = fast_inverse(disp);
            std::array<Complex, MAXP> disp_inv_pow;
            disp_inv_pow[0] = Complex{1.0, 0.0};
            for (size_t i = 1; i <= this->order; i++)  disp_inv_pow[i] = disp_inv_pow[i - 1] * disp_inv;

            coefficients[0] = me(0) * std::log(-disp);
            for (size_t k = 1; k <= this->order; k++) {
                double sign = (k & 1) ? -1.0 : 1.0;
                coefficients[0] += sign * me(k) * disp_inv_pow[k];
            }

            for (size_t j = 1; j <= this->order; j++) {
                coefficients[j] = -me(0) / static_cast<double>(j);

                for (size_t k = 1; k <= this->order; k++) {
                    double sign = (k & 1) ? -1.0 : 1.0;
                    double binom = SeriesExpansion::binom_table(j + k - 1, k - 1);
                    coefficients[j] += sign * me(k) * disp_inv_pow[k] * binom;
                }

                coefficients[j] *= disp_inv_pow[j];
            }

            return coefficients;
        }

        std::array<Complex, MAXP> shift (const Complex &disp) const { // Horner's rule magic
            std::array<Complex, MAXP> shifted = this->coefficients;

            for (size_t j = 0; j < this->order; j++) {
                for (size_t k = this->order - j - 1; k < this->order; k++)
                    shifted[k] -= disp * shifted[k + 1];
            }

            return shifted;
        }

        double evaluatePotential (const Complex &point) const override {
            Complex disp = point - this->center;
            Complex result = (*this)(this->order);

            for (int k = this->order - 1; k >= 0; k--) {
                result = result * disp + (*this)(k);
            }

            return result.real();
        }

        Complex evaluateForce (const Complex &point) const override {
            Complex disp = point - this->center;

            Complex result = static_cast<double>(this->order) * (*this)(this->order);
            for (size_t k = this->order - 1; k >= 1; k--) {
                result = result * disp + static_cast<double>(k) * (*this)(k);
            }

            return Complex{-result.real(), result.imag()};
        }
        
    };

}

#endif