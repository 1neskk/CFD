#pragma once

#include "key_codes.h"

#include <glm/glm.hpp>

namespace input
{
	class input
	{
	public:
		static bool is_key_pressed(key_code key);
		static bool is_mouse_button_pressed(mouse_button button);

		static glm::vec2 get_mouse_position();

		static void set_cursor_mode(cursor_mode mode);
	};
}