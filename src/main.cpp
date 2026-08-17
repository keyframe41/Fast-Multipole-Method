#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>

#include <omp.h>

#include "fmm_tree.hpp"
#include "barnes_hut_tree.hpp"
#include "diagnostics.hpp"

int main() {
	sf::Color p_color = sf::Color::Cyan;	
	const int screen_size = 1380;
	sf::RenderWindow window(sf::VideoMode({screen_size, screen_size}) , "Simulation");
	const int frame_rate = 60;
	window.setFramerateLimit(frame_rate);

	const double dt = 0.001;

	sf::Clock fpstimer;
	const sf::Font font("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");

	std::mt19937 gen(42);
	std::normal_distribution<double> cluster(1.0 * screen_size / 2, 1.0 * screen_size / 10);
	std::uniform_real_distribution<double> uniform(100.0, screen_size * 1.0 - 100.0);

	std::vector<fmm::Source> sources;
	Complex center{1.0 * screen_size / 2, 1.0 * screen_size / 2};
	for (size_t i = 0; i < 40000; i++) { 
		double x = cluster(gen), y = cluster(gen);
		while (x < 100 or x > screen_size - 100) x = cluster(gen);
		while (y < 100 or y > screen_size - 100) y = cluster(gen);
		sources.emplace_back(x, y, 10.0);
	}
	for (size_t i = 0; i < 10000; i++) 
		sources.emplace_back(uniform(gen), uniform(gen), 1.0);

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
	
	fmm::FmmTree tree(sources, 60, 10);
	// fmm::BhTree tree(sources, 1, 2.5);
	tree.buildTree();

	bool drawBoxes = false;

	int frame_limit = 1800, tot_frames = 0;
	float tot_time = 0, max_time = 0, min_time = 100;
	float times[frame_limit];

	double error_l2[frame_limit];
	double error_abs[frame_limit];
	double avg_error_l2 = 0, avg_error_abs = 0, max_error = 0;

	while (window.isOpen()) {
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) window.close();

			else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
				if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) window.close();
			}
		}
		fpstimer.restart();
		window.clear(sf::Color::Black);

		#pragma omp parallel for
		for (size_t i = 0; i < sources.size(); i++) {
			sources[i].velocity += 0.5 * tree.forces[i] * dt;
			sources[i].position += sources[i].velocity * dt;
		}

		// forces = fmm::computeExactForces(sources);
		tree.buildTree();
		// fmm::ErrorData error = fmm::evaluateSimulationError(sources, tree.forces, 1000);
		
		if (drawBoxes) {
			auto boxes = tree.getBoxGeometries();
			sf::VertexArray box_va(sf::PrimitiveType::Lines, boxes.size() * 8);
			sf::Color box_color(66, 191, 245);

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < (int)boxes.size(); i++) {
				int id_b = i * 8;

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

		window.draw(particle_va);

		float ms = 1.0 * fpstimer.getElapsedTime().asMicroseconds() / 1000;
		sf::Text number(font, std::to_string(ms) + "ms", 24);
		number.setFillColor(sf::Color::White);
		window.draw(number);
		window.display();

		// min_time = std::min(min_time, ms);
		// max_time = std::max(max_time, ms);
		// tot_time += ms;
		// times[tot_frames] = ms;
		
		tot_frames++;
		// if (tot_frames == frame_limit) break;
	}
	return 0;
}
