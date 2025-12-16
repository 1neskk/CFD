#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

class MeshVoxelizer {
public:
    struct Triangle {
        glm::vec3 v0, v1, v2;
    };

    MeshVoxelizer();
    ~MeshVoxelizer();

    bool load_obj(const std::string& path);
    
    void voxelize(unsigned char* buffer, int width, int height, int depth, float scale);
    void rotate_y(float angle_degrees);
    
    glm::vec3 get_min_bound() const { return m_min_bound; }
    glm::vec3 get_max_bound() const { return m_max_bound; }

private:
    std::vector<Triangle> m_triangles;
    glm::vec3 m_min_bound;
    glm::vec3 m_max_bound;

    bool is_inside(const glm::vec3& point);
    bool ray_triangle_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, 
                                const Triangle& tri, float& t);
};
