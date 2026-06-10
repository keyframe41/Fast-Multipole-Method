#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>

#include <omp.h>

#include "fmm_tree.hpp"

int main() {
    // std::cout << "Max OpenMP Threads Available: " << omp_get_max_threads() << std::endl;
    // #pragma omp parallel
	// {
    //     int id = omp_get_thread_num();
    //     #pragma omp critical
    //     {
    //         std::cout << "CPU core " << id << "available" << std::endl;
    //     }
    // }
    // std::cout << "--- OPENMP TEST FINISHED ---" << std::endl;

	// sf::Color p_color = sf::Color{167, 65, 250};	
	sf::Color p_color = sf::Color::Cyan;	

	sf::RenderWindow window(sf::VideoMode({1000, 1000}) , "Simulation");
	const int frame_rate = 60;
    window.setFramerateLimit(frame_rate);

	const double dt = 0.001;

	sf::Clock fpstimer;
	const sf::Font font("/Library/Fonts/JetBrainsMonoNL-Regular.ttf");

	std::mt19937 gen(42);
	std::normal_distribution<double> cluster(500.0, 100.0);
	std::uniform_real_distribution<double> uniform(100.0, 900.0);
	std::uniform_real_distribution<double> uniform_center(-100.0, 100.0);

	std::vector<fmm::PointSource> sources;
	Complex center{500.0, 500.0};
	for (size_t i = 0; i < 100000; i++) {
		double x = cluster(gen), y = cluster(gen);
		while (x < 100 or x > 900) x = cluster(gen);
		while (y < 100 or y > 900) y = cluster(gen);
		sources.emplace_back(x, y, 10.0);
	}
	for (size_t i = 0; i < 20000; i++) 
		sources.emplace_back(uniform(gen), uniform(gen), 1.0);
    
    // Set a strict maximum radius so no particle spawns outside the window
    // double max_cluster_radius = 400.0;
    // double max_uniform_radius = 450.0;

    // // 1. Generate Clustered Particles (Gaussian)
    // while (sources.size() < 100000) {
    //     double x = cluster(gen);
    //     double y = cluster(gen);
        
    //     double dx = x - center.real();
    //     double dy = y - center.imag();
        
    //     // REJECTION SAMPLING: Only accept if it falls inside our maximum radius!
    //     if (std::sqrt(dx*dx + dy*dy) <= max_cluster_radius) {
    //         sources.emplace_back(x, y, 1.0);
    //     }
    // }

    // // 2. Generate Uniform Particles (Circle, not a square!)
    // while (sources.size() < 20000) { // 40,000 + 10,000 = 50,000 total
    //     double x = uniform(gen);
    //     double y = uniform(gen);
        
    //     double dx = x - center.real();
    //     double dy = y - center.imag();
        
    //     // REJECTION SAMPLING: Ensures the "background" particles form a perfect circle
    //     if (std::sqrt(dx*dx + dy*dy) <= max_uniform_radius) {
    //         sources.emplace_back(x, y, 1.0);
    //     }
    // }

	const double orbital_speed = 200.0;
	for (auto &s : sources) {
		Complex disp = s.position - center;
		double r = std::abs(disp);
		if (r > 1.0) {
			Complex tangent{-disp.imag(), disp.real()};
			tangent /= r;
			s.velocity = tangent * orbital_speed;
		}
	}
	
	fmm::FmmTree tree(sources, 100, 12);
	tree.buildTree();

	while (window.isOpen()) {
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) window.close();

			else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
				if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) window.close();
				if (keyPressed->scancode == sf::Keyboard::Scancode::Space) {
					#pragma omp parallel 
					{
						// Create a local generator just for this CPU core, seeded by the thread ID
						std::mt19937 local_gen(42 + omp_get_thread_num());
						
						#pragma omp for
						for (size_t i = 0; i < sources.size(); i++) {
							sources[i].velocity += Complex{uniform_center(local_gen), uniform_center(local_gen)};
						}
					}
				} 
			}
		}
		fpstimer.restart();
		window.clear(sf::Color::Black);

		#pragma omp parallel for
		for (size_t i = 0; i < sources.size(); i++) {
			sources[i].velocity += 0.5 * tree.forces[i] * dt;
			sources[i].position += sources[i].velocity * dt;
		}

		tree.buildTree();
		
		// auto boxes = tree.getBoxGeometries();
		// sf::VertexArray box_va(sf::PrimitiveType::Lines, boxes.size() * 8);
		// sf::Color box_color(66, 191, 245);

		// #pragma omp parallel for schedule(static)
		// for (size_t i = 0; i < (int)boxes.size(); i++) {
		// 	int id_b = i * 8;

		// 	float cx = static_cast<float>(boxes[i].first.real());
		// 	float cy = static_cast<float>(boxes[i].first.imag());
		// 	float len = static_cast<float>(boxes[i].second / 2.0);

		// 	sf::Vector2f tl(cx - len, cy - len);
		// 	sf::Vector2f tr(cx + len, cy - len);
		// 	sf::Vector2f br(cx + len, cy + len);
		// 	sf::Vector2f bl(cx - len, cy + len);

		// 	box_va[id_b    ] = sf::Vertex{tl, box_color};
		// 	box_va[id_b + 1] = sf::Vertex{tr, box_color};

		// 	box_va[id_b + 2] = sf::Vertex{tr, box_color};
		// 	box_va[id_b + 3] = sf::Vertex{br, box_color};

		// 	box_va[id_b + 4] = sf::Vertex{br, box_color};
		// 	box_va[id_b + 5] = sf::Vertex{bl, box_color};

		// 	box_va[id_b + 6] = sf::Vertex{bl, box_color};
		// 	box_va[id_b + 7] = sf::Vertex{tl, box_color};
		// }
		// window.draw(box_va);

		#pragma omp parallel for
		for (size_t i = 0; i < sources.size(); i++)
			sources[i].velocity += 0.5 * tree.forces[i] * dt;

		sf::VertexArray particle_va(sf::PrimitiveType::Points, sources.size());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)sources.size(); ++i) {
            float cx = static_cast<float>(sources[i].position.real());
            float cy = static_cast<float>(sources[i].position.imag());
            
            particle_va[i] = sf::Vertex{sf::Vector2f(cx, cy), p_color};
        }

		window.draw(particle_va); // 5ms

        float ms = 1.0 * fpstimer.getElapsedTime().asMicroseconds() / 1000;
        ms = 1.0 * fpstimer.getElapsedTime().asMicroseconds() / 1000;
        sf::Text number(font, std::to_string(ms) + "ms", 24);
        number.setFillColor(sf::Color::White);
        window.draw(number);

		window.display();
	}

	return 0;
}
