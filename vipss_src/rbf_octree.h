#pragma once
#include <string>
//#include "Octree/octree.h"

#include "CGAL/Simple_cartesian.h"
#include "CGAL/Octree.h"
#include "CGAL/Point_set_3.h"
#include "CGAL/Point_set_3/IO.h"
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/Orthtree.h>
#include <CGAL/Orthtree_traits_3.h>

#include "Octree.hpp"
#include <cstdlib>
#include <time.h>

//using namespace OrthoTree;

using std::array;
using std::vector;

//class RBF_Octree {
//    public:
//        RBF_Octree(){};
//        ~RBF_Octree(){};
//        void LoadOctreePts(const std::string& path);
//        std::vector<Point3D> GetPts(size_t level);
//
//    public:
//        OctreePointC octree_;
//        
//
//};

namespace CGAL_OCTREE {


    typedef CGAL::Simple_cartesian<double> Kernel;

    typedef Kernel::Point_3 Point;
    typedef Kernel::Vector_3 Vec3;

    typedef CGAL::Sphere_3<Kernel> Sphere_3;
    typedef CGAL::Point_set_3<Point, Vec3> Point_set;
    typedef Point_set::Point_map Point_map;
    typedef CGAL::Octree<Kernel, Point_set, Point_map> Octree;
    typedef CGAL::Orthtrees::Preorder_traversal Preorder_traversal;
    typedef CGAL::Orthtree_traits_3<Kernel> Traits;
    typedef CGAL::Orthtree<Traits, Point_set> Orthtree;
    typedef Orthtree::Sphere QSphere;
    typedef struct {
        Point pt;
        Kernel dist;
    }PtD;


    class CGAL_OCTree
    {
    public:
        /*CGAL_OCTree(){};
        ~CGAL_OCTree(){};*/
        int LoadPts(const std::string& data_path);
        void BuildOCTree();
        void FindNearestNeighbor();
        void SearchNeighborWithinRadius(double radius);

    public:
        std::vector<double> in_pts_;
        /*std::vector<Point> in_pts_cgal_;*/
        //Octree octree_;
        std::shared_ptr<Octree> tree_ptr_;

        Point_set in_points_;
        size_t leaf_box_pt_num_ = 4;
        size_t max_level_ = 10;
        size_t radius_search_max_num_ = 20;

    private:
        

    };

    int test_CGAL(const std::string& path);
}

template<typename T>
class CustomPoint
{
public:
    CustomPoint(T x, T y, T z) : x(x), y(y), z(z)
    {
    }

    

    T getX() const
    {
        return x;
    }
    T getY() const
    {
        return y;
    }
    T getZ() const
    {
        return z;
    }

    void setX(const T x_val)
    {
        x = x_val;
    }

    void setY(const T y_val)
    {
        y = y_val;
    }

    void setZ(const T z_val)
    {
        z = z_val;
    }

public:
    T x, y, z;
    //size_t id;
};

//typedef CustomPoint<double> Pt3d;
typedef CustomPoint<float> Pt3f;

class CustomOctree
{
public:
    void InitPts(const std::vector<double>& in_pts);
    void RadiusSearch(const Pt3f& q, float radius, std::vector<uint32_t>& results);

public:
    std::vector<Pt3f> pts_;
    unibn::Octree<Pt3f> octree_;

};