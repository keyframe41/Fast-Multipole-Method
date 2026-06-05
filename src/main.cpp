#include <SFML/Graphics.hpp>
#include <vector>
#include <random>

#include "fmm_tree.hpp"

int main() {
	sf::RenderWindow window(sf::VideoMode({1000, 1000}) , "Simulation");
	const int frame_rate = 60;
    window.setFramerateLimit(frame_rate);

	sf::Clock fpstimer;
	const sf::Font font("/Library/Fonts/JetBrainsMonoNL-Regular.ttf");

	std::mt19937 gen(42);
	std::normal_distribution<double> cluster(500.0, 100.0);
	std::uniform_real_distribution<double> uniform(50.0, 950.0);

	std::vector<fmm::PointSource> sources;
	for (std::size_t i = 0; i < 40000; i++) 
		sources.emplace_back(cluster(gen), cluster(gen), 1.0);
	for (std::size_t i = 0; i < 10000; i++) 
		sources.emplace_back(uniform(gen), uniform(gen), 1.0);
	
	fmm::FmmTree tree(50);

	while (window.isOpen()) {
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) window.close();

			else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>())
				if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) window.close();
		}
		fpstimer.restart();
		window.clear(sf::Color::Black);

		tree.buildTree(sources);
		auto boxes = tree.getBoxGeometries();

		sf::VertexArray box_va(sf::PrimitiveType::Lines, boxes.size() * 8);
		int id_b = 0;
		sf::Color box_color(66, 191, 245);
		for (const auto &box : boxes) {
			float cx = static_cast<float>(box.first.real());
			float cy = static_cast<float>(box.first.imag());
			float len = static_cast<float>(box.second / 2.0);

			sf::Vector2f tl(cx - len, cy - len);
			sf::Vector2f tr(cx + len, cy - len);
			sf::Vector2f br(cx + len, cy + len);
			sf::Vector2f bl(cx - len, cy + len);

			box_va[id_b++] = sf::Vertex{tl, box_color};
			box_va[id_b++] = sf::Vertex{tr, box_color};

			box_va[id_b++] = sf::Vertex{tr, box_color};
			box_va[id_b++] = sf::Vertex{br, box_color};

			box_va[id_b++] = sf::Vertex{br, box_color};
			box_va[id_b++] = sf::Vertex{bl, box_color};

			box_va[id_b++] = sf::Vertex{bl, box_color};
			box_va[id_b++] = sf::Vertex{tl, box_color};
		}
		
		sf::VertexArray particle_va(sf::PrimitiveType::Triangles, sources.size() * 6);
		int id_p = 0;
		sf::Color p_color = sf::Color::Cyan;
		float rad = 1.0f;
		
		for (const auto &s : sources) {
			float cx = static_cast<float>(s.position.real());
			float cy = static_cast<float>(s.position.imag());
			
			sf::Vector2f tl(cx - rad, cy - rad);
			sf::Vector2f tr(cx + rad, cy - rad);
			sf::Vector2f br(cx + rad, cy + rad);
			sf::Vector2f bl(cx - rad, cy + rad);
			
			particle_va[id_p++] = sf::Vertex{tl, p_color};
			particle_va[id_p++] = sf::Vertex{tr, p_color};
			particle_va[id_p++] = sf::Vertex{bl, p_color};
			
			particle_va[id_p++] = sf::Vertex{bl, p_color};
			particle_va[id_p++] = sf::Vertex{tr, p_color};
			particle_va[id_p++] = sf::Vertex{br, p_color};
		}

		window.draw(box_va);
		window.draw(particle_va);

        float ms = 1.0 * fpstimer.getElapsedTime().asMicroseconds() / 1000;
        ms = 1.0 * fpstimer.getElapsedTime().asMicroseconds() / 1000;
        sf::Text number(font, std::to_string(ms) + "ms", 24);
        number.setFillColor(sf::Color::White);
        window.draw(number);

		window.display();
	}

	return 0;
}
