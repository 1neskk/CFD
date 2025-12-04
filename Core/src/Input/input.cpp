#include "input.h"

#include "../application.h"

#include <GLFW/glfw3.h>

namespace input
{
	bool input::is_key_pressed(key_code keycode)
	{
		GLFWwindow* window = application::get().get_window();
		auto state = glfwGetKey(window, static_cast<int>(keycode));
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool input::is_mouse_button_pressed(mouse_button button)
	{
		GLFWwindow* window = application::get().get_window();
		auto state = glfwGetMouseButton(window, static_cast<int>(button));
		return state == GLFW_PRESS;
	}

	glm::vec2 input::get_mouse_position()
	{
		GLFWwindow* window = application::get().get_window();
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		return { static_cast<float>(x), static_cast<float>(y) };
	}

	void input::set_cursor_mode(cursor_mode mode)
	{
		GLFWwindow* window = application::get().get_window();
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL + static_cast<int>(mode));
	}
}