#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "fmm_tree.hpp"
#include "barnes_hut_tree.hpp"
#include "diagnostics.hpp"

int main()
{
    // ============================================================
    // Window settings
    // ============================================================

    constexpr unsigned int screen_size = 1380;
    constexpr unsigned int frame_rate = 60;

    sf::RenderWindow window(
        sf::VideoMode({screen_size, screen_size}),
        "Simulation"
    );

    window.setFramerateLimit(frame_rate);

    // ============================================================
    // Simulation settings
    // ============================================================

    constexpr double dt = 0.001;
    constexpr double orbital_speed = 200.0;

    constexpr std::size_t cluster_particle_count = 40000;
    constexpr std::size_t uniform_particle_count = 10000;

    // ============================================================
    // Font
    // ============================================================

    const std::string font_path =
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf";

    if (!std::filesystem::exists(font_path))
    {
        std::cerr
            << "ERROR: Font not found:\n"
            << "  " << font_path << "\n\n"
            << "Install JetBrains Mono with:\n"
            << "  sudo apt install fonts-jetbrains-mono\n";

        return EXIT_FAILURE;
    }

    sf::Font font(font_path);

    // ============================================================
    // Random number generation
    // ============================================================

    std::mt19937 gen(42);

    const double center_coordinate =
        static_cast<double>(screen_size) / 2.0;

    const double cluster_sigma =
        static_cast<double>(screen_size) / 10.0;

    std::normal_distribution<double> cluster_distribution(
        center_coordinate,
        cluster_sigma
    );

    std::uniform_real_distribution<double> uniform_distribution(
        100.0,
        static_cast<double>(screen_size) - 100.0
    );

    // ============================================================
    // Create particles
    // ============================================================

    std::vector<fmm::Source> sources;

    sources.reserve(
        cluster_particle_count +
        uniform_particle_count
    );

    Complex center{
        center_coordinate,
        center_coordinate
    };

    // ------------------------------------------------------------
    // Dense central cluster
    // ------------------------------------------------------------

    for (std::size_t i = 0;
         i < cluster_particle_count;
         ++i)
    {
        double x = cluster_distribution(gen);
        double y = cluster_distribution(gen);

        while (
            x < 100.0 ||
            x > static_cast<double>(screen_size) - 100.0
        )
        {
            x = cluster_distribution(gen);
        }

        while (
            y < 100.0 ||
            y > static_cast<double>(screen_size) - 100.0
        )
        {
            y = cluster_distribution(gen);
        }

        sources.emplace_back(
            x,
            y,
            10.0
        );
    }

    // ------------------------------------------------------------
    // Uniform particles
    // ------------------------------------------------------------

    for (std::size_t i = 0;
         i < uniform_particle_count;
         ++i)
    {
        sources.emplace_back(
            uniform_distribution(gen),
            uniform_distribution(gen),
            1.0
        );
    }

    std::cout
        << "Created "
        << sources.size()
        << " particles.\n";

    // ============================================================
    // Initial orbital velocities
    // ============================================================

    for (auto& source : sources)
    {
        Complex displacement =
            source.position - center;

        const double radius =
            std::abs(displacement);

        if (radius > 1.0)
        {
            Complex tangent{
                -displacement.imag(),
                displacement.real()
            };

            tangent /= radius;

            source.velocity =
                tangent * orbital_speed;
        }
    }

    // ============================================================
    // FMM tree
    // ============================================================

    fmm::FmmTree tree(
        sources,
        60,
        10
    );

    // Alternative Barnes-Hut implementation:
    //
    // fmm::BhTree tree(
    //     sources,
    //     1,
    //     2.5
    // );

    std::cout << "Building initial FMM tree...\n";

    tree.buildTree();

    std::cout << "FMM tree ready.\n";

    // ============================================================
    // Rendering settings
    // ============================================================

    constexpr bool draw_boxes = false;

    const sf::Color particle_color =
        sf::Color::Cyan;

    const sf::Color box_color(
        66,
        191,
        245
    );

    // ============================================================
    // Diagnostics
    // ============================================================

    constexpr int frame_limit = 1800;

    int total_frames = 0;

    float total_time = 0.0f;
    float max_time = 0.0f;
    float min_time = 1000000.0f;

    std::vector<float> times(
        frame_limit
    );

    std::vector<double> error_l2(
        frame_limit
    );

    std::vector<double> error_abs(
        frame_limit
    );

    double average_error_l2 = 0.0;
    double average_error_abs = 0.0;
    double max_error = 0.0;

    // Prevent unused-variable warnings until diagnostics are enabled.
    (void)error_l2;
    (void)error_abs;
    (void)average_error_l2;
    (void)average_error_abs;
    (void)max_error;

    // ============================================================
    // Frame timer
    // ============================================================

    sf::Clock fpstimer;

    // ============================================================
    // Main simulation loop
    // ============================================================

    while (window.isOpen())
    {
        // --------------------------------------------------------
        // Events
        // --------------------------------------------------------

        while (const std::optional event =
                   window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (
                const auto* key_pressed =
                    event->getIf<sf::Event::KeyPressed>()
            )
            {
                if (
                    key_pressed->scancode ==
                    sf::Keyboard::Scancode::Escape
                )
                {
                    window.close();
                }
            }
        }

        // --------------------------------------------------------
        // Start frame timer
        // --------------------------------------------------------

        fpstimer.restart();

        // --------------------------------------------------------
        // Clear window
        // --------------------------------------------------------

        window.clear(
            sf::Color::Black
        );

        // ========================================================
        // First half of velocity Verlet integration
        // ========================================================

        #pragma omp parallel for schedule(static)
        for (int i = 0;
             i < static_cast<int>(sources.size());
             ++i)
        {
            sources[i].velocity +=
                0.5 *
                tree.forces[i] *
                dt;

            sources[i].position +=
                sources[i].velocity *
                dt;
        }

        // ========================================================
        // Rebuild FMM tree
        // ========================================================

        tree.buildTree();

        // ========================================================
        // Optional exact-force diagnostics
        // ========================================================

        /*
        auto exact_forces =
            fmm::computeExactForces(sources);

        fmm::ErrorData error =
            fmm::evaluateSimulationError(
                sources,
                tree.forces,
                1000
            );
        */

        // ========================================================
        // Draw FMM boxes
        // ========================================================

        if (draw_boxes)
        {
            const auto boxes =
                tree.getBoxGeometries();

            sf::VertexArray box_vertices(
                sf::PrimitiveType::Lines,
                boxes.size() * 8
            );

            #pragma omp parallel for schedule(static)
            for (int i = 0;
                 i < static_cast<int>(boxes.size());
                 ++i)
            {
                const std::size_t id =
                    static_cast<std::size_t>(i) * 8;

                const float cx =
                    static_cast<float>(
                        boxes[i].first.real()
                    );

                const float cy =
                    static_cast<float>(
                        boxes[i].first.imag()
                    );

                const float half_size =
                    static_cast<float>(
                        boxes[i].second / 2.0
                    );

                const sf::Vector2f top_left{
                    cx - half_size,
                    cy - half_size
                };

                const sf::Vector2f top_right{
                    cx + half_size,
                    cy - half_size
                };

                const sf::Vector2f bottom_right{
                    cx + half_size,
                    cy + half_size
                };

                const sf::Vector2f bottom_left{
                    cx - half_size,
                    cy + half_size
                };

                box_vertices[id] =
                    sf::Vertex{
                        top_left,
                        box_color
                    };

                box_vertices[id + 1] =
                    sf::Vertex{
                        top_right,
                        box_color
                    };

                box_vertices[id + 2] =
                    sf::Vertex{
                        top_right,
                        box_color
                    };

                box_vertices[id + 3] =
                    sf::Vertex{
                        bottom_right,
                        box_color
                    };

                box_vertices[id + 4] =
                    sf::Vertex{
                        bottom_right,
                        box_color
                    };

                box_vertices[id + 5] =
                    sf::Vertex{
                        bottom_left,
                        box_color
                    };

                box_vertices[id + 6] =
                    sf::Vertex{
                        bottom_left,
                        box_color
                    };

                box_vertices[id + 7] =
                    sf::Vertex{
                        top_left,
                        box_color
                    };
            }

            window.draw(
                box_vertices
            );
        }

        // ========================================================
        // Second half of velocity Verlet integration
        // ========================================================

        #pragma omp parallel for schedule(static)
        for (int i = 0;
             i < static_cast<int>(sources.size());
             ++i)
        {
            sources[i].velocity +=
                0.5 *
                tree.forces[i] *
                dt;
        }

        // ========================================================
        // Create particle vertex array
        // ========================================================

        sf::VertexArray particle_vertices(
            sf::PrimitiveType::Points,
            sources.size()
        );

        // ========================================================
        // Fill particle vertex array
        // ========================================================

        #pragma omp parallel for schedule(static)
        for (int i = 0;
             i < static_cast<int>(sources.size());
             ++i)
        {
            const float x =
                static_cast<float>(
                    sources[i].position.real()
                );

            const float y =
                static_cast<float>(
                    sources[i].position.imag()
                );

            particle_vertices[i] =
                sf::Vertex{
                    sf::Vector2f{x, y},
                    particle_color
                };
        }

        // --------------------------------------------------------
        // Draw particles
        // --------------------------------------------------------

        window.draw(
            particle_vertices
        );

        // ========================================================
        // Frame timing
        // ========================================================

        const float milliseconds =
            static_cast<float>(
                fpstimer
                    .getElapsedTime()
                    .asMicroseconds()
            ) / 1000.0f;

        // ========================================================
        // Draw timing information
        // ========================================================

        sf::Text timing_text(
            font,
            std::to_string(milliseconds) + " ms",
            24
        );

        timing_text.setFillColor(
            sf::Color::White
        );

        timing_text.setPosition(
            {10.0f, 10.0f}
        );

        window.draw(
            timing_text
        );

        // ========================================================
        // Display frame
        // ========================================================

        window.display();

        // ========================================================
        // Timing statistics
        // ========================================================

        if (total_frames < frame_limit)
        {
            times[total_frames] =
                milliseconds;
        }

        total_time += milliseconds;

        max_time =
            std::max(
                max_time,
                milliseconds
            );

        min_time =
            std::min(
                min_time,
                milliseconds
            );

        ++total_frames;

        // Uncomment to stop after frame_limit frames:
        //
        // if (total_frames >= frame_limit)
        // {
        //     break;
        // }
    }

    // ============================================================
    // Final statistics
    // ============================================================

    if (total_frames > 0)
    {
        const float average_time =
            total_time /
            static_cast<float>(total_frames);

        std::cout
            << "\nSimulation finished.\n"
            << "Frames: "
            << total_frames
            << '\n'
            << "Average frame time: "
            << average_time
            << " ms\n"
            << "Minimum frame time: "
            << min_time
            << " ms\n"
            << "Maximum frame time: "
            << max_time
            << " ms\n";
    }

    return EXIT_SUCCESS;
}
