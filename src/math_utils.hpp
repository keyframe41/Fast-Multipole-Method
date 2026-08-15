#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <complex>
#include <vector>
#include <limits>

#include "adaptive_quadtree.hpp"

namespace fmm {

constexpr int MAXP = 30 + 1;

class BinomialTable {
public:
    int max_order = 0;
    std::vector<double> table;

    void init (int order) {
        if (order <= max_order) return;

        max_order = order;
        table.assign(((max_order + 1) * (max_order + 2)) / 2, 0.0);

        for (size_t n = 0; n <= max_order; n++) {
            table[getIndex(n, 0)] = 1.0;
            table[getIndex(n, n)] = 1.0;
            for (size_t k = 1; k < n; k++) 
                table[getIndex(n, k)] = table[getIndex(n - 1, k - 1)] + table[getIndex(n - 1, k)];
        }
    }

    inline unsigned getIndex (unsigned n, unsigned k) const {
        return (n * (n + 1)) / 2 + k;
    }

    inline double operator()(int n, int k) const {
        return table[getIndex(n, k)];
    }
};

class SeriesExpansion {
public:
    Complex center;
    int order;
    std::array<Complex, MAXP> coefficients;

    inline static BinomialTable binom_table;

    SeriesExpansion() : center(0.0, 0.0), order(0) { coefficients.fill(Complex{0.0, 0.0}); }
    SeriesExpansion(const Complex &center, int order) : center(center), order(order) {
        coefficients.fill(Complex{0.0, 0.0});
        binom_table.init(2 * order);
    }

    virtual ~SeriesExpansion() = default;

    inline SeriesExpansion &operator+=(const SeriesExpansion &other) {
        for (size_t i = 0; i <= order; i++)
            coefficients[i] += other.coefficients[i];

        return *this;
    }

    inline Complex &operator()(unsigned n) { return coefficients[n]; }
    inline const Complex &operator()(unsigned n) const { return coefficients[n]; }

    virtual double evaluatePotential (const Complex &point) const = 0;
    virtual Complex evaluateForce (const Complex &point) const = 0;

};

std::vector<Source> readFile (const std::string &filename) {
    std::ifstream ifile(filename);
    if (!ifile.is_open()) throw std::runtime_error("Could not open file: " + filename);

    std::string line;
    std::vector<Source> sources;

    double x, y, charge;
    while (std::getline(ifile, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ss >> x >> y >> charge;
        sources.emplace_back(x, y, charge);
    }
    return sources;
}

void toFile (const std::vector<Source> &sources, const std::string &filename) {
    std::ofstream ofile(filename);
    if (!ofile.is_open()) throw std::runtime_error("Could not open file: " + filename);

    ofile << std::setprecision(std::numeric_limits<double>::digits10) << std::scientific;
    for (const auto &s : sources) {
        ofile << s.position.real() << "\t" << s.position.imag() << "\t" << s.q << "\n";
    }
    ofile.close();
}

inline Complex fast_inverse (Complex& z) {
    double norm_sq = z.real() * z.real() + z.imag() * z.imag();
    return Complex(z.real() / norm_sq, -z.imag() / norm_sq);
}

}


#endif