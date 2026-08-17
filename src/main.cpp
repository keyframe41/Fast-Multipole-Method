#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <optional>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <omp.h>

#include "fmm_tree.hpp"
#include "barnes_hut_tree.hpp"
#include "diagnostics.hpp"

int main() {
    sf::Color p_color = sf::Color::Cyan;
    const int screen_size = 1380;
    sf::RenderWindow window(sf::VideoMode({screen_size, screen_size}), "Simulation");
    window.setFramerateLimit(60);

    // Simulation settings
    const double dt = 0.001;
    const int BUILD_EVERY_N_FRAMES = 4; // 1 = every frame, 2 = every second frame, etc.

    sf::Clock frameTimer;
    sf::Clock phaseClock;
    sf::Clock uiTimer;

    // Font loading with fallbacks
    sf::Font font;
    bool canDrawText = false;
    {
        const std::vector<std::string> fontCandidates = {
            "assets/fonts/JetBrainsMonoNL-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "/Library/Fonts/JetBrainsMonoNL-Regular.ttf"
        };

        for (const auto& path : fontCandidates) {
            if (font.openFromFile(path)) {
                std::cout << "Loaded font: " << path << "\n";
                canDrawText = true;
                break;
            }
        }

        if (!canDrawText) {
            std::cerr << "Warning: no font loaded. Overlay text disabled.\n";
        }
    }

    // Initial particle setup
    std::mt19937 gen(42);
    std::normal_distribution<double> cluster(1.0 * screen_size / 2, 1.0 * screen_size / 10);
    std::uniform_real_distribution<double> uniform(100.0, screen_size * 1.0 - 100.0);

    std::vector<fmm::Source> sources;
    sources.reserve(50000);

    Complex center{1.0 * screen_size / 2, 1.0 * screen_size / 2};
    for (size_t i = 0; i < 40000; i++) {
        double x = cluster(gen), y = cluster(gen);
        while (x < 100 || x > screen_size - 100) x = cluster(gen);
        while (y < 100 || y > screen_size - 100) y = cluster(gen);
        sources.emplace_back(x, y, 10.0);
    }
    for (size_t i = 0; i < 10000; i++)
        sources.emplace_back(uniform(gen), uniform(gen), 1.0);

    const double orbital_speed = 200.0;
    for (auto& s : sources) {
        Complex disp = s.position - center;
        double r = std::abs(disp);
        if (r > 1.0) {
            Complex tangent{-disp.imag(), disp.real()};
            tangent /= r;
            s.velocity = tangent * orbital_speed;
        }
    }

    // Build initial tree/forces
    fmm::FmmTree tree(sources, 60, 10);
    // fmm::BhTree tree(sources, 1, 2.5);
    tree.buildTree();

    bool drawBoxes = false;

    // Reused draw buffers
    sf::VertexArray particle_va(sf::PrimitiveType::Points, sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        particle_va[i].color = p_color;
    }

    sf::Text overlay(font, "", 20);
    overlay.setFillColor(sf::Color::White);
    overlay.setPosition({10.f, 8.f});

    // Timing stats
    int tot_frames = 0;
    float tIntegrateMs = 0.f, tBuildMs = 0.f, tRenderMs = 0.f, tFrameMs = 0.f;
    float emaBuildMs = 0.f;
    float maxBuildMs = 0.f;
    constexpr float emaAlpha = 0.10f;

    while (window.isOpen()) {
        frameTimer.restart();

        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) window.close();
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) window.close();
            }
        }

        window.clear(sf::Color::Black);

        // Phase 1: integrate (half-kick + drift)
        phaseClock.restart();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(sources.size()); i++) {
            sources[i].velocity += 0.5 * tree.forces[i] * dt;
            sources[i].position += sources[i].velocity * dt;
        }
        tIntegrateMs = static_cast<float>(phaseClock.getElapsedTime().asMicroseconds()) / 1000.0f;

        // Phase 2: rebuild tree (optionally decimated)
        phaseClock.restart();
        if (BUILD_EVERY_N_FRAMES <= 1 || (tot_frames % BUILD_EVERY_N_FRAMES) == 0) {
            tree.buildTree();
        }
        tBuildMs = static_cast<float>(phaseClock.getElapsedTime().asMicroseconds()) / 1000.0f;

        // Build-time stats
        if (tot_frames == 0) emaBuildMs = tBuildMs;
        else emaBuildMs = (1.0f - emaAlpha) * emaBuildMs + emaAlpha * tBuildMs;
        maxBuildMs = std::max(maxBuildMs, tBuildMs);

        // Phase 3: second half-kick + draw prep + draw
        phaseClock.restart();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(sources.size()); i++) {
            sources[i].velocity += 0.5 * tree.forces[i] * dt;
        }

        // Updating vertex buffer is often smoother single-threaded
        for (size_t i = 0; i < sources.size(); ++i) {
            particle_va[i].position = sf::Vector2f(
                static_cast<float>(sources[i].position.real()),
                static_cast<float>(sources[i].position.imag())
            );
        }

        if (drawBoxes) {
            auto boxes = tree.getBoxGeometries();
            sf::VertexArray box_va(sf::PrimitiveType::Lines, boxes.size() * 8);
            sf::Color box_color(66, 191, 245);

            for (size_t i = 0; i < boxes.size(); i++) {
                int id_b = static_cast<int>(i) * 8;

                float cx = static_cast<float>(boxes[i].first.real());
                float cy = static_cast<float>(boxes[i].first.imag());
                float len = static_cast<float>(boxes[i].second / 2.0);

                sf::Vector2f tl(cx - len, cy - len);
                sf::Vector2f tr(cx + len, cy - len);
                sf::Vector2f br(cx + len, cy + len);
                sf::Vector2f bl(cx - len, cy + len);

                box_va[id_b    ] = sf::Vertex{tl, box_color};
                box_va[id_b + 1] = sf::Vertex{tr, box_color};
                box_va[id_b + 2] = sf::Vertex{tr, box_color};
                box_va[id_b + 3] = sf::Vertex{br, box_color};
                box_va[id_b + 4] = sf::Vertex{br, box_color};
                box_va[id_b + 5] = sf::Vertex{bl, box_color};
                box_va[id_b + 6] = sf::Vertex{bl, box_color};
                box_va[id_b + 7] = sf::Vertex{tl, box_color};
            }
            window.draw(box_va);
        }

        window.draw(particle_va);

        tRenderMs = static_cast<float>(phaseClock.getElapsedTime().asMicroseconds()) / 1000.0f;
        tFrameMs = static_cast<float>(frameTimer.getElapsedTime().asMicroseconds()) / 1000.0f;

        // Update overlay at ~10 Hz
        if (canDrawText && uiTimer.getElapsedTime().asMilliseconds() >= 100) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << "frame: " << tFrameMs << " ms\n"
                << "integrate: " << tIntegrateMs << " ms\n"
                << "buildTree: " << tBuildMs << " ms"
                << " (ema " << emaBuildMs << ", max " << maxBuildMs << ")\n"
                << "render: " << tRenderMs << " ms\n"
                << "build every N: " << BUILD_EVERY_N_FRAMES;
            overlay.setString(oss.str());
            uiTimer.restart();
        }

        if (canDrawText) {
            window.draw(overlay);
        }

        window.display();
        tot_frames++;
    }

    return 0;
}
