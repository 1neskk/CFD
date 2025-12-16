#include "mesh_voxelizer.h"
#include "logger.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>
#include <algorithm>
#include <limits>

constexpr auto EPSILON = 0.0000001f;

MeshVoxelizer::MeshVoxelizer() : m_min_bound(std::numeric_limits<float>::max()), m_max_bound(std::numeric_limits<float>::lowest()) {
}

MeshVoxelizer::~MeshVoxelizer() {
}

bool MeshVoxelizer::load_obj(const std::string& path) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./assets/"; 

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(path, reader_config)) {
        if (!reader.Error().empty()) {
            LOG_ERROR("TinyObjReader: {}", reader.Error());
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        LOG_WARN("TinyObjReader: {}", reader.Warning());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    m_triangles.clear();
    m_min_bound = glm::vec3(std::numeric_limits<float>::max());
    m_max_bound = glm::vec3(std::numeric_limits<float>::lowest());

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            // We assume triangulated mesh (fv should be 3)
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            Triangle tri;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                
                float vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                float vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                float vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                glm::vec3 vert(vx, vy, vz);
                if (v == 0) tri.v0 = vert;
                if (v == 1) tri.v1 = vert;
                if (v == 2) tri.v2 = vert;

                m_min_bound = glm::min(m_min_bound, vert);
                m_max_bound = glm::max(m_max_bound, vert);
            }
            m_triangles.push_back(tri);

            index_offset += fv;
        }
    }

#ifdef _DEBUG
    LOG_INFO("Loaded mesh with {} triangles. Bounds: [{}, {}, {}] to [{}, {}, {}]", 
#endif
        m_triangles.size(), 
        m_min_bound.x, m_min_bound.y, m_min_bound.z,
        m_max_bound.x, m_max_bound.y, m_max_bound.z);

    return true;
}

// Moller-Trumbore intersection algorithm
bool MeshVoxelizer::ray_triangle_intersect(const glm::vec3& ray_origin, const glm::vec3& ray_dir, 
                                           const Triangle& tri, float& t) {
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;

    edge1 = tri.v1 - tri.v0;
    edge2 = tri.v2 - tri.v0;
    h = glm::cross(ray_dir, edge2);
    a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.

    f = 1.0f / a;
    s = ray_origin - tri.v0;
    u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    q = glm::cross(s, edge1);
    v = f * glm::dot(ray_dir, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * glm::dot(edge2, q);

    if (t > EPSILON)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool MeshVoxelizer::is_inside(const glm::vec3& point) {
    // Simple AABB check first
    if (point.x < m_min_bound.x || point.x > m_max_bound.x ||
        point.y < m_min_bound.y || point.y > m_max_bound.y ||
        point.z < m_min_bound.z || point.z > m_max_bound.z) {
        return false;
    }

    // Ray casting: Cast a ray in +X direction
    glm::vec3 ray_dir(1.0f, 0.0f, 0.0f);
    int intersections = 0;

    for (const auto& tri : m_triangles) {
        float t;
        if (ray_triangle_intersect(point, ray_dir, tri, t)) {
            intersections++;
        }
    }

    // Odd number of intersections means inside
    return (intersections % 2) != 0;
}

void MeshVoxelizer::voxelize(unsigned char* buffer, int width, int height, int depth, float scale) {
    glm::vec3 grid_center(width / 2.0f, height / 2.0f, depth / 2.0f);

    glm::vec3 mesh_center;
    mesh_center.x = (m_min_bound.x + m_max_bound.x) / 2;
    mesh_center.y = m_min_bound.y + grid_center.y / scale;
    mesh_center.z = (m_min_bound.z + m_max_bound.z) / 2;
    
    LOG_INFO("Voxelizing mesh. Mesh Center: [{}, {}, {}] -> Grid Center: [{}, {}, {}], Scale: {}", 
        mesh_center.x, mesh_center.y, mesh_center.z,
        grid_center.x, grid_center.y, grid_center.z, scale);

    // Precompute triangle bounds for fast culling
    struct TriBounds {
        float min_y, max_y, min_z, max_z;
    };
    std::vector<TriBounds> tri_bounds(m_triangles.size());
    for (size_t i = 0; i < m_triangles.size(); ++i) {
        const auto& tri = m_triangles[i];
        tri_bounds[i].min_y = std::min({tri.v0.y, tri.v1.y, tri.v2.y});
        tri_bounds[i].max_y = std::max({tri.v0.y, tri.v1.y, tri.v2.y});
        tri_bounds[i].min_z = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
        tri_bounds[i].max_z = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
    }

    #pragma omp parallel for collapse(2)
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            glm::vec3 grid_start(0, y, z);
            glm::vec3 ray_origin_mesh = (grid_start - grid_center) / scale + mesh_center;
            glm::vec3 ray_dir(1.0f, 0.0f, 0.0f);

            std::vector<float> intersections;
            intersections.reserve(16); // Heuristic reserve

            for (size_t i = 0; i < m_triangles.size(); ++i) {
                if (ray_origin_mesh.y < tri_bounds[i].min_y || ray_origin_mesh.y > tri_bounds[i].max_y ||
                    ray_origin_mesh.z < tri_bounds[i].min_z || ray_origin_mesh.z > tri_bounds[i].max_z) {
                    continue;
                }

                float t;
                if (ray_triangle_intersect(ray_origin_mesh, ray_dir, m_triangles[i], t)) {
                    intersections.push_back(t);
                }
            }

            std::sort(intersections.begin(), intersections.end());

            for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
                float t_enter = intersections[i];
                float t_exit = intersections[i+1];

                int x_start = std::max(0, (int)ceil(t_enter * scale));
                int x_end = std::min(width, (int)floor(t_exit * scale));

                for (int x = x_start; x < x_end; x++) {
                     int idx = z * width * height + y * width + x;
                     buffer[idx] = 255;
                }
            }
        }
    }
}

void MeshVoxelizer::rotate_y(float angle_degrees) {
    float rad = glm::radians(angle_degrees);
    float c = cos(rad);
    float s = sin(rad);

    m_min_bound = glm::vec3(std::numeric_limits<float>::max());
    m_max_bound = glm::vec3(std::numeric_limits<float>::lowest());

    for (auto& tri : m_triangles) {
        auto rotate = [&](glm::vec3& v) {
            float x = v.x * c + v.z * s;
            float z = -v.x * s + v.z * c;
            v.x = x;
            v.z = z;
        };

        rotate(tri.v0);
        rotate(tri.v1);
        rotate(tri.v2);

        m_min_bound = glm::min(m_min_bound, tri.v0);
        m_min_bound = glm::min(m_min_bound, tri.v1);
        m_min_bound = glm::min(m_min_bound, tri.v2);

        m_max_bound = glm::max(m_max_bound, tri.v0);
        m_max_bound = glm::max(m_max_bound, tri.v1);
        m_max_bound = glm::max(m_max_bound, tri.v2);
    }
}
