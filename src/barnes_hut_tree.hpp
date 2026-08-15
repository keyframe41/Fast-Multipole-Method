#ifndef BARNES_HUT_TREE_H
#define BARNES_HUT_TREE_H

#include "adaptive_quadtree.hpp"
#include <cassert>
#include <omp.h>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>

namespace fmm {

struct BhNode : public BaseNode {
    Complex center_of_mass;
    double total_mass = 0.0;

    uint32_t start_id = 0;
    uint32_t num_sources = 0;

    BhNode() : BaseNode() {}
};

class BhTree : public QuadTree<BhNode> {
public:
    std::vector<Source> &sources;
    std::vector<std::vector<uint32_t>> level_indices;

    size_t max_sources_per_leaf;
    double theta; // Multipole acceptance criterion

    std::vector<Complex> forces;

    BhTree(std::vector<Source> &src, size_t max_s_p_l, double theta) 
        : sources(src), max_sources_per_leaf(max_s_p_l), theta(theta) 
    {}

    void buildTree() {
        if (sources.empty()) return;
        arena.resize(sources.size() * 4);
        active_nodes = 0;

        level_indices.clear();

        auto [l_bound, u_bound] = getDataRange(sources);

        double box_len = std::max(u_bound.real() - l_bound.real(), u_bound.imag() - l_bound.imag());
        Complex center((l_bound.real() + u_bound.real()) / 2.0, (l_bound.imag() + u_bound.imag()) / 2.0);

        bool is_leaf = (sources.size() <= max_sources_per_leaf);
        root_id = allocateNode(center, box_len, 0, NULL_NODE, is_leaf);

        arena[root_id].start_id = 0;
        arena[root_id].num_sources = static_cast<uint32_t>(sources.size());

        level_indices.push_back({root_id});

        if (is_leaf) {
            this->height = 0;
            computeCenterOfMass();
            computeForces();
            return;
        }

        sortTree(root_id);
        computeCenterOfMass();
        computeForces();
    }

    std::vector<std::pair<Complex, double>> getBoxGeometries() {
        std::vector<std::pair<Complex, double>> boxes;
        BFS([this, &boxes](uint32_t node_id) {
            boxes.push_back({arena[node_id].center, arena[node_id].box_length});
        });
        return boxes;
    }

private:
    std::size_t active_nodes = 0;

    uint32_t allocateNode(Complex center, double box_len, size_t level, uint32_t parent, bool is_leaf) {
        uint32_t id;
        #pragma omp atomic capture
        {
            id = active_nodes;
            active_nodes++;
        }
        
        assert(id < arena.size() && "Arena size exceeded! Adjust initial size of arena in buildTree().");

        arena[id].center = center;
        arena[id].box_length = box_len;
        arena[id].level = level;
        arena[id].parent = parent;
        arena[id].is_leaf = is_leaf;
        arena[id].children.fill(NULL_NODE);
        arena[id].center_of_mass = Complex{0.0, 0.0};
        arena[id].total_mass = 0.0;
        arena[id].start_id = 0;
        arena[id].num_sources = 0;
        
        return id;
    }

    void sortTree(uint32_t initial_root) {
        unsigned depth = 0;
        bool subdivide = true;
        std::vector<uint32_t> prev_internal_nodes = {initial_root};
        
        while (subdivide) {
            depth++;

            size_t max_children = prev_internal_nodes.size() * 4;
            std::vector<uint32_t> next_level(max_children, NULL_NODE);
            std::vector<uint32_t> next_internal_nodes(max_children, NULL_NODE);

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < prev_internal_nodes.size(); i++) {
                uint32_t parent_id = prev_internal_nodes[i];

                double cx = arena[parent_id].center.real();
                double cy = arena[parent_id].center.imag();
                double child_len = arena[parent_id].box_length / 2.0;

                uint32_t start_id = arena[parent_id].start_id;
                uint32_t count = arena[parent_id].num_sources;

                auto begin = sources.begin() + start_id;
                auto end = begin + count;

                auto mid_x = std::partition(begin, end,
                    [cx](const Source &s) { return s.position.real() < cx; });

                auto mid_y1 = std::partition(begin, mid_x,
                    [cy](const Source &s) { return s.position.imag() < cy; });
                auto mid_y2 = std::partition(mid_x, end,
                    [cy](const Source &s) { return s.position.imag() < cy; });

                std::array<uint32_t, 4> q_counts = {
                    static_cast<uint32_t>(std::distance(begin, mid_y1)),
                    static_cast<uint32_t>(std::distance(mid_y1, mid_x)),
                    static_cast<uint32_t>(std::distance(mid_x, mid_y2)),
                    static_cast<uint32_t>(std::distance(mid_y2, end))
                };
                
                uint32_t child_start_id = start_id;

                for (size_t c = 0; c < 4; c++) {
                    uint32_t c_count = q_counts[c];

                    if (c_count > 0)  {
                        Complex child_center = arena[parent_id].center + child_directions[c] * (child_len / 2.0);

                        bool is_leaf = (c_count <= max_sources_per_leaf) || (depth >= 20);

                        uint32_t child_id = allocateNode(child_center, child_len, depth, parent_id, is_leaf);

                        arena[child_id].start_id = child_start_id;
                        arena[child_id].num_sources = c_count;
                        arena[parent_id].children[c] = child_id;

                        next_level[i * 4 + c] = child_id;

                        if (!is_leaf) next_internal_nodes[i * 4 + c] = child_id;
                    }
                    child_start_id += c_count;
                }
            }

            next_level.erase(std::remove(next_level.begin(), next_level.end(), NULL_NODE), next_level.end());
            next_internal_nodes.erase(std::remove(next_internal_nodes.begin(), next_internal_nodes.end(), NULL_NODE), next_internal_nodes.end());
            
            if (!next_level.empty()) level_indices.push_back(next_level);
            subdivide = !next_internal_nodes.empty();
            prev_internal_nodes = std::move(next_internal_nodes);
        }
        this->height = depth;
    }

    void computeCenterOfMass() {
        for (int depth = this->height; depth >= 0; depth--) {
            const auto &level = level_indices[depth];

            #pragma omp parallel for
            for (size_t i = 0; i < level.size(); i++) {
                uint32_t node_id = level[i];
                BhNode &node = arena[node_id];

                double mass = 0.0;
                Complex c_o_mass{0.0, 0.0};

                if (node.is_leaf) {
                    for (uint32_t s_id = node.start_id; s_id < node.start_id + node.num_sources; s_id++) {
                        double q = sources[s_id].q;
                        mass += q;
                        c_o_mass += q * sources[s_id].position;
                    }
                    node.total_mass = mass;
                    if (mass > 1e-12) node.center_of_mass = c_o_mass / mass;
                    else node.center_of_mass = node.center;
                } else {
                    for (size_t c = 0; c < 4; c++) {
                        uint32_t child_id = node.children[c];
                        if (child_id == NULL_NODE) continue;
                        
                        double m_child = arena[child_id].total_mass;
                        mass += m_child;
                        c_o_mass += m_child * arena[child_id].center_of_mass;
                    }
                    node.total_mass = mass;
                    if (mass > 1e-12) node.center_of_mass = c_o_mass / mass;
                    else node.center_of_mass = node.center;
                }
            }
        }
    }

    void computeForces() {
        forces.assign(sources.size(), Complex{0.0, 0.0});

        const double inv_theta_sq = 1.0 / (theta * theta);

        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t t_id = 0; t_id < sources.size(); t_id++) {
            const Source &target = sources[t_id];
            const double tx = target.position.real();
            const double ty = target.position.imag();

            double fx = 0.0;
            double fy = 0.0;

            uint32_t stack[128];
            int stack_ptr = 0;
            stack[stack_ptr++] = root_id;

            while (stack_ptr > 0) {
                uint32_t cur_id = stack[--stack_ptr];
                const BhNode &node = arena[cur_id];
                if (node.num_sources == 0) continue;

                double dx = tx - node.center_of_mass.real();
                double dy = ty - node.center_of_mass.imag();
                double r2 = dx * dx + dy * dy;

                if (node.is_leaf) {
                    for (uint32_t s_id = node.start_id; s_id < node.start_id + node.num_sources; s_id++) {
                        if (t_id == s_id) continue;
                        const Source &src = sources[s_id];
                        
                        double pdx = tx - src.position.real();
                        double pdy = ty - src.position.imag();
                        r2 = pdx * pdx + pdy * pdy;
                        double factor = -src.q / r2;
                        fx += factor * pdx;
                        fy += factor * pdy;
                    }
                } else {
                    // Approximate if node is sufficiently far
                    double s = node.box_length;
                    double threshold = s * s * inv_theta_sq;

                    if (r2 > 1e-24 && threshold < r2) {
                        r2 += 10.0;
                        double factor = -node.total_mass / r2;
                        fx += factor * dx;
                        fy += factor * dy;
                    } else {
                        if (node.children[0] != NULL_NODE) stack[stack_ptr++] = node.children[0];
                        if (node.children[1] != NULL_NODE) stack[stack_ptr++] = node.children[1];
                        if (node.children[2] != NULL_NODE) stack[stack_ptr++] = node.children[2];
                        if (node.children[3] != NULL_NODE) stack[stack_ptr++] = node.children[3];
                    }
                }
            }

            forces[t_id] = Complex{fx, fy};
        }
    }
};

}

#endif