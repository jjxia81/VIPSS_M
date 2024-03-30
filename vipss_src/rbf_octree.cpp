#include "rbf_octree.h"
#include "readers.h"
#include <array>
#include <iostream>
#include <CGAL/nearest_neighbor_delaunay_2.h>
#include <boost/iterator/function_output_iterator.hpp>

//
//
//void RBF_Octree::LoadOctreePts(const std::string& path)
//{
//    std::vector<double> pts;
//    std::vector<double> normals;
//    readPLYFile(path, pts, normals);
//    
//    const size_t n_pt = pts.size() / 3;
//    std::vector<Point3D> oct_points;
//    for(size_t i = 0; i < n_pt; ++i)
//    {
//        Point3D cur_p{ pts[3 * i], pts[3 * i + 1], pts[3 * i + 2] };
//        oct_points.push_back(cur_p);
//        
//    }
//    size_t max_npt = oct_points.size();
//    bool parallel_create = true;
//    /*auto const bbox = BoundingBox3D{ {0, 0, 0}, {1.0, 1.0, 1.0} };*/
//    auto const bbox = BoundingBox3D{ {-1, -1, -1}, {1.0, 1.0, 1.0} };
//    /*octree_ = OctreePointC(oct_points, 8, bbox, max_npt, parallel_create); */
//
//}
//
//std::vector<Point3D> RBF_Octree::GetPts(size_t level)
//{
//    std::vector<Point3D> out_pts;
//    
//    auto ids = octree_.CollectAllIdInDFS();
//    size_t max_n = 70;
//    std::vector<double> pts;
//    for (size_t i =0; i <= 100; ++i)
//    {
//        auto p = octree_.Get(ids[i]);
//        pts.push_back(p[0]);
//        pts.push_back(p[1]);
//        pts.push_back(p[2]);
//    }
//    std::string out_path = "octree_pts.ply";
//    writePLYFile(out_path, pts);
//    
//}

namespace CGAL_OCTREE {

    // Type Declarations
    
    int CGAL_OCTree::LoadPts(const std::string& data_path)
    {
        std::ifstream stream(CGAL::data_file_path(data_path));
        stream >> in_points_;
        if (0 == in_points_.number_of_points()) {
            std::cerr << "Error: cannot read file" << std::endl;
            return EXIT_FAILURE;
        }
        
        /*in_points_.add_property_map<size_t>("id");*/

        /*auto normals = in_points_.normal_map();
        auto points = in_points_.point_map();
        for (size_t i = 0; i < 10; ++i)
        {
            cout << points[i] << " ";
            cout << normals[i] << endl;

        }*/
      
        return EXIT_SUCCESS;
    }

    void CGAL_OCTree::BuildOCTree()
    {
        //tree_ptr_ = std::make_shared<Octree>(in_points_, in_points_.point_map());
        Octree new_tree(in_points_, in_points_.point_map());
        
        new_tree.refine(10, 20);
        auto points = in_points_.point_map();
        /*std::vector<Point> points_to_find = {
         {0, 0, 0},
         {1, 1, 1},
         {-1, -1, -1},
         {-0.46026, -0.25353, 0.32051},
         {-0.460261, -0.253533, 0.320513}
        };*/
        
        for (const auto& pt : points)
        {
            cout << "search point : " << pt << endl;
            QSphere sphere(pt, 0.01);
            std::vector<Point> output;
            std::cout << "squared radius " << sphere.squared_radius() << std::endl;
            new_tree.nearest_neighbors(sphere, std::back_inserter(output));
          
            for (auto pt : output)
            {
                std::cout << "pt " << pt << std::endl;
            }

            new_tree.nearest_neighbors
            (pt, 2, // k=1 to find the single closest point
                boost::make_function_output_iterator
                ([&](const Point& nearest)
                    {
                        std::cout << "the nearest point to (" << pt <<
                            ") is (" << nearest << ")" << std::endl;
                    }));

           
            break;
        }

        
    }


    int test_CGAL(const std::string& path)
    {
        CustomOctree octree;
        std::vector<double> pts;
        std::vector<double> nts;
        readPLYFile(path, pts, nts);
        octree.InitPts(pts);
        float radius = 0.03;
        std::vector<uint32_t> results;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            Pt3f pt((float)pts[3 * i], (float)pts[3 * i + 1], float(pts[3*i + 2]));

            std::cout << " query pt " <<  pt.getX() << " " << pt.getY() << " " << pt.getZ() << std::endl;
            octree.RadiusSearch(pt, radius, results);
            
            for (auto id : results)
            {
                std::cout << octree.pts_[id].getX() << " " << octree.pts_[id].getY() 
                    << " " << octree.pts_[id].getZ() << std::endl;
            }
            break;
        }
        
        //CGAL_OCTree octree;
        /*octree.LoadPts(path);
        octree.BuildOCTree();
        */
        // Point set will be used to hold our points
        //Point_set points;
        // Load points from a file.
        
    
        //if (0 == points.number_of_points()) {
        //    std::cerr << "Error: cannot read file" << std::endl;
        //    return EXIT_FAILURE;
        //}
        //std::cout << points.has_normal_map() << std::endl;
        //std::cout << "loaded " << points.number_of_points() << " points\n" << std::endl;
        //    //Create an octree from the points
        //Octree octree(points, points.point_map());
        //// Build the octree using the default arguments
        //size_t leaf_box_pt_num = 20;
        //octree.refine(10, leaf_box_pt_num);
        //// Print out a few nodes
        //std::cout << "Navigation relative to the root node" << std::endl;
        //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        //std::cout << "the root node: " << std::endl;
        //std::cout << octree.root() << std::endl;
        //std::cout << "the first child of the root node: " << std::endl;
        //std::cout << octree.root()[0].local_coordinates() << std::endl;
        //std::cout << "the fifth child: " << std::endl;
        //std::cout << octree.root()[4] << std::endl;
        //std::cout << "the fifth child, accessed without the root keyword: " << std::endl;
        //std::cout << octree[4] << std::endl;
        //std::cout << "the second child of the fourth child: " << std::endl;
        //std::cout << octree.root()[4][1] << std::endl;
        //std::cout << "the second child of the fourth child, accessed without the root keyword: " << std::endl;
        //std::cout << octree[4][1] << std::endl;
        //std::cout << std::endl;
        //// Retrieve one of the deeper children
        //Octree::Node cur = octree[3][2];
        //std::cout << "Navigation relative to a child node" << std::endl;
        //std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        //std::cout << "the third child of the fourth child: " << std::endl;
        //std::cout << cur << std::endl;
        //std::cout << "the third child: " << std::endl;
        //std::cout << cur.parent() << std::endl;
        //std::cout << "the next sibling of the third child of the fourth child: " << std::endl;
        //std::cout << cur.parent()[cur.local_coordinates().to_ulong() + 1] << std::endl;
        //
        //std::vector<double> outpts;
        //auto node = octree.root();
        //size_t sum = 0;
        //size_t sumup = 0;
        //std::stack<Octree::Node> search_nodes;
        //search_nodes.push(node);
        //std::set< Octree::Node> valid_nodes;
        //while (!search_nodes.empty())
        //{
        //    auto s_node = search_nodes.top();
        //    search_nodes.pop();
        //   
        //    if ( s_node.size() <= leaf_box_pt_num)
        //    {
        //        valid_nodes.insert(s_node);
        //    }
        //    else {
        //        for (size_t i = 0; i < 8; ++i)
        //        {
        //            if (s_node[i].is_null()) continue;
        //          
        //            auto new_node = s_node[i];
        //            if (new_node.size() == 0) continue;
        //            search_nodes.push(new_node);
        //        }
        //    }
        //
        //}

        //size_t nid = 0;
        //for (auto& node : valid_nodes)
        //{
        //    size_t p_count = 0;
        //    double cx = 0;
        //    double cy = 0;
        //    double cz = 0;
        //    std::vector<double> cur_out_pts;
        //    cout << "node size " << node.size() << endl;
        //    for (auto itor = node.begin(); itor != node.end(); ++itor)
        //    {
        //        p_count++;
        //        auto& cur_p = points.point(*itor);
        //        cx += cur_p.x();
        //        cy += cur_p.y();
        //        cz += cur_p.z();
        //        cur_out_pts.push_back(cur_p.x());
        //        cur_out_pts.push_back(cur_p.y());
        //        cur_out_pts.push_back(cur_p.z());
        //       
        //    }
        //    std::string c_out_path = std::to_string(nid) + "octree_pts";
        //    writePLYFile(c_out_path, cur_out_pts);
        //    nid++;

        //    cx /= double(p_count);
        //    cy /= double(p_count);
        //    cz /= double(p_count);
        //    outpts.push_back(cx);
        //    outpts.push_back(cy);
        //    outpts.push_back(cz);
        //}
        //

        ///*for (size_t i = 0; i < 8; ++i)
        //{
        //    sumup += node[i].size();
        //    for (size_t j = 0; j < 8; ++j)
        //    {
        //        sum += node[i][j].size();
        //        cout << "child size " << node[i][j].size() << endl;
        //    }
        //}*/
        //cout << "sum up " << sumup << endl;
        //cout << "sum " << sum << endl;
 
     
        //// Print out the octree using preorder traversal
        //for (Octree::Node node : octree.traverse<Preorder_traversal>()) {
        //    break;
        //    int count = 0;
        //    for (auto itor = node.begin(); itor != node.end(); ++itor)
        //    {
        //        count++;
        //        auto p_it = points.begin() + (*itor);
        //        auto& cur_p = points.point(*p_it);
        //        std::cout << cur_p << std::endl;
        //        outpts.push_back(cur_p.x());
        //        outpts.push_back(cur_p.y());
        //        outpts.push_back(cur_p.z());

        //        if (count > 70) break;
        //    }
        //    break;
        //}
        //std::string out_path = "octree_pts.ply";
        //writePLYFile(out_path, outpts);

       
        return EXIT_SUCCESS;

    }
}


void CustomOctree::InitPts(const std::vector<double>& in_pts)
{
    pts_.clear();
    for (size_t i = 0; i < in_pts.size() / 3; ++i)
    {
        Pt3f pt((float)in_pts[3 * i], (float)in_pts[3 * i + 1], (float)in_pts[3 * i + 2]);
        pts_.push_back(pt);
    }
    octree_.initialize(pts_);
}

void CustomOctree::RadiusSearch(const Pt3f& q, float radius, std::vector<uint32_t>& results)
{
    octree_.radiusNeighbors<unibn::L2Distance<Pt3f>>(q, radius, results);
}