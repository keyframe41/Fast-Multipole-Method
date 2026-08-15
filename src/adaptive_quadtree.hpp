#ifndef QUADTREE_H
#define QUADTREE_H

#include <iostream>
#include <algorithm>
#include <complex>
#include <cmath>
#include <vector>
#include <queue>
#include <array>
#include <cstdint>

using Complex = std::complex<double>; 

namespace fmm {

constexpr uint32_t NULL_NODE = 0xFFFFFFFF;


    struct Source {
        Complex position;
        Complex velocity;
        double q;

        Source() : position(0.0, 0.0), velocity(0.0, 0.0), q(0.0) {}
        Source(double x, double y, double q)
            : position(x, y), velocity(0.0, 0.0), q(q) {}

    };

    struct BaseNode {
        Complex center;
        double box_length;
        std::size_t level;
        bool is_leaf;

        uint32_t parent;
        std::array<uint32_t, 4> children;

        BaseNode() : parent(NULL_NODE), children({NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE}) {}
        ~BaseNode() = default;
        
        bool adjacent (const BaseNode &other) const;

    };

    template<typename Node_T>
    class QuadTree {
    public:
        std::vector<Node_T> arena; // memory pool

        uint32_t root_id;
        std::size_t height;
        static const int n_children = 4;
        const std::array<Complex, 4> child_directions;

        QuadTree() : root_id(NULL_NODE), height(0), 
            child_directions({Complex(-1, -1), Complex(-1, 1), Complex(1, -1), Complex(1, 1)}) {};
        virtual ~QuadTree() = default;
        
        static std::tuple<Complex, Complex> getDataRange(const std::vector<Source> &points);

        void BFS(const std::function<void(uint32_t)> &processNode);
    };

    inline bool BaseNode::adjacent (const BaseNode &other) const {
        double dx = std::abs(this->center.real() - other.center.real());
        double dy = std::abs(this->center.imag() - other.center.imag());
        
        double tol = std::min(this->box_length, other.box_length) / 2.0;
        double lim = (this->box_length + other.box_length) / 2.0 + tol;

        return dx <= lim && dy <= lim;
    }

    template<typename Node_T>
    inline void QuadTree<Node_T>::BFS(const std::function<void(uint32_t)>& func) {
        if (this->root_id == NULL_NODE) return;

        std::queue<uint32_t> node_queue;
        node_queue.push(this->root_id);

        while (!node_queue.empty()) {
            uint32_t cur = node_queue.front();
            node_queue.pop();

            func(cur);

            for (std::size_t i = 0; i < 4; i++) {
                if (this->arena[cur].children[i] != NULL_NODE) {
                    node_queue.push(this->arena[cur].children[i]);
                }
            }
        }
    }

    template<typename Node_T>
    inline std::tuple<Complex, Complex> QuadTree<Node_T>::getDataRange(
        const std::vector<Source>& points) {

        if (points.empty()) return {0.0, 0.0};

        double min_x = points[0].position.real(), max_x = points[0].position.real();
        double min_y = points[0].position.imag(), max_y = points[0].position.imag();
        
        for (const auto &p : points) {
            min_x = std::min(min_x, p.position.real());
            max_x = std::max(max_x, p.position.real());
            min_y = std::min(min_y, p.position.imag());
            max_y = std::max(max_y, p.position.imag());
        }

        double pad_x = std::min((max_x - min_x) * 1e-5, 1e-5);
        double pad_y = std::min((max_y - min_y) * 1e-5, 1e-5);

        return std::make_tuple(Complex(min_x - pad_x, min_y - pad_y), 
                            Complex(max_x + pad_x, max_y + pad_y));
    }

}

#endif