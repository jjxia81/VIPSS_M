#include "rbfcore.h"
#include "utility.h"
#include "Solver.h"
#include <armadillo>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <queue>
#include "readers.h"
#include "sample.h"
//#include "mymesh/UnionFind.h"
//#include "mymesh/tinyply.h"

typedef std::chrono::high_resolution_clock Clock;
double randomdouble() {return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);}
double randomdouble(double be,double ed) {return be + randomdouble()*(ed-be);	}

void RBF_Core::AssignInitNormals(const std::vector<double>& in_normals)
{
    initnormals = in_normals;
}

void RBF_Core::NormalRecification(double maxlen, vector<double>&nors){


    double maxlen_r = -1;
    auto p_vn = nors.data();
    int  np = nors.size()/3;
    if(1){
        for(int i=0;i<np;++i){
            maxlen_r = max(maxlen_r,MyUtility::normVec(p_vn+i*3));
        }

        cout<<"maxlen_r: "<<maxlen_r<<endl;
        double ratio = maxlen / maxlen_r;
        for(auto &a:nors)a*=ratio;
    }else{
        for(int i=0;i<np;++i){
            MyUtility::normalize(p_vn+i*3);
        }

    }




}

bool RBF_Core::Write_Hermite_NormalPrediction(string fname, int mode){


//    vector<uchar>labelcolor(npt*4);
//    vector<uint>f2v;
//    uchar red[] = {255,0,0, 255};
//    uchar green[] = {0,255,0, 255};
//    uchar blue[] = {0,0,255, 255};
//    for(int i=0;i<labels.size();++i){
//        uchar *pcolor;
//        if(labels[i]==0)pcolor = green;
//        else if(labels[i]==-1)pcolor = blue;
//        else if(labels[i]==1)pcolor = red;
//        for(int j=0;j<4;++j)labelcolor[i*4+j] = pcolor[j];
//    }
    //fname += mp_RBF_METHOD[curMethod];

//    for(int i=0;i<npt;++i){
//        uchar *pcolor = green;
//        for(int j=0;j<4;++j)labelcolor[i*4+j] = pcolor[j];
//    }

    vector<double>nors;
    if(mode ==0)nors=initnormals;
    else if(mode == 1)nors=newnormals;
    else if(mode == 2)nors = initnormals_uninorm;
    NormalRecification(1.,nors);

    //for(int i=0;i<npt;++i)if(randomdouble()<0.5)MyUtility::negVec(nors.data()+i*3);
    //cout<<pts.size()<<' '<<f2v.size()<<' '<<nors.size()<<' '<<labelcolor.size()<<endl;
    //writePLYFile(fname,pts,f2v,nors,labelcolor);

    //writeObjFile_vn(fname,pts,nors);
    writePLYFile_VN(fname,pts, initnormals);

    return 1;
}



void RBF_Core::Set_RBF(vector<double>&pts){
    isHermite = false;
    a.set_size(npt);
    M.set_size(npt,npt);
    double *p_pts = pts.data();
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            {
                M(i,j) = M(j,i) = Kernal_Function_2p(p_pts+i*3, p_pts+j*3);
            }
        }
    }
}

void RBF_Core::Set_RBFSparse(vector<double>&pts, double kernel_radius){
    isHermite = false;
    a.set_size(npt);
    M_s.set_size(npt,npt);

    std::vector<std::vector<uint32_t>> pt_pair_ids;
    // arma::mat dist_simbol = arma::ones(npt, npt) * -1.0;
    for (int i = 0; i < npt; ++i) {
        Pt3f new_p(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        std::vector<uint32_t> results;
        octree.RadiusSearch(new_p, (float)kernel_radius, results);
        pt_pair_ids.push_back(results);
    }

    double *p_pts = pts.data();
    for(int i=0;i<npt;++i){
        for (auto j : pt_pair_ids[i])
        {
            if (M_s(i, j) != 0) continue;
            M_s(i, j) = M_s(j, i) = Kernal_Function_2p(p_pts + i * 3, p_pts + j * 3);  
        }
    }
}



void RBF_Core::Set_HermiteRBF(vector<double>&pts){

    cout<<"Set_HermiteRBF"<<endl;
    //for(auto a:pts)cout<<a<<' ';cout<<endl;
    isHermite = true;
    if(apply_sample && auxi_npt > 0)
    {

        As_.set_size(auxi_npt,npt*4 + 4);
        
        double *p_pts = pts.data();
        double *auxi_p_pts = auxi_pts.data();
        double Gs[3];
        std::cout << " use sample points " << std::endl;
        std::cout << " auxi points number " << auxi_npt << std::endl;
        for(int i=0;i<auxi_npt;++i){
            for(int j=0;j<npt;++j){
                As_(i,j) = Kernal_Function_2p(auxi_p_pts+i*3, p_pts+j*3);
                Kernal_Gradient_Function_2p(auxi_p_pts+i*3, p_pts+j*3, Gs);
                for(int k=0;k<3;++k)As_(i,npt+j+k*npt) = Gs[k];
            }
            As_(i,4*npt) = 1;
            for(int j=0;j<3;++j)As_(i, 4*npt + j+1) = auxi_p_pts[i*3+j];
        }
    }

    a.set_size(npt*4);
    M.set_size(npt*4,npt*4);
    double *p_pts = pts.data();
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){
            M(i,j) = M(j,i) = Kernal_Function_2p(p_pts+i*3, p_pts+j*3);
        }
    }

    double G[3];
    for(int i=0;i<npt;++i){
        for(int j=0;j<npt;++j){

            Kernal_Gradient_Function_2p(p_pts+i*3, p_pts+j*3, G);
            //            int jind = j*3+npt;
            //            for(int k=0;k<3;++k)M(i,jind+k) = -G[k];
            //            for(int k=0;k<3;++k)M(jind+k,i) = G[k];

            for(int k=0;k<3;++k)M(i,npt+j+k*npt) = G[k];
            for(int k=0;k<3;++k)M(npt+j+k*npt,i) = G[k];

        }
    }

    double H[9];
    for(int i=0;i<npt;++i){
        for(int j=i;j<npt;++j){

            Kernal_Hessian_Function_2p(p_pts+i*3, p_pts+j*3, H);
            //            int iind = i*3+npt;
            //            int jind = j*3+npt;
            //            for(int k=0;k<3;++k)
            //                for(int l=0;l<3;++l)
            //                    M(jind+l,iind+k) = M(iind+k,jind+l) = -H[k*3+l];

            for(int k=0;k<3;++k)
                for(int l=0;l<3;++l)
                    M(npt+j+l*npt,npt+i+k*npt) = M(npt+i+k*npt,npt+j+l*npt) = -H[k*3+l];
        }
    }

    //cout<<std::setprecision(5)<<std::fixed<<M<<endl;

    bsize= 4;
    N.zeros(npt*4,4);
    b.set_size(4);

    for(int i=0;i<npt;++i){
        N(i,0) = 1;
        for(int j=0;j<3;++j)N(i,j+1) = pts[i*3+j];
    }
    for(int i=0;i<npt;++i){
        //        int ind = i*3+npt;
        //        for(int j=0;j<3;++j)N(ind+j,j+1) = 1;

        for(int j=0;j<3;++j)N(npt+i+j*npt,j+1) = -1;
    }

    //cout<<N<<endl;
    //arma::vec eigval = eig_sym( M ) ;
    //cout<<eigval.t()<<endl;
    if(only_build_M)
    {
        return;
    }


    if(!isnewformula){
        cout<<"start solve M: "<<endl;
        auto t1 = Clock::now();
        if(isinv)Minv = inv(M);
        else {
            arma::mat Eye;
            Eye.eye(npt*4,npt*4);
            Minv = solve(M,Eye);
        }
        cout<<"solved M: "<<(invM_time = std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;

        t1 = Clock::now();
        if(isinv)bprey = inv_sympd(N.t() * Minv * N) * N.t() * Minv;
        else {
            arma::mat Eye2;
            Eye2.eye(bsize,bsize);
            bprey = solve(N.t() * Minv * N, Eye2) * N.t() * Minv;
        }
        cout<<"solved bprey "<<std::chrono::nanoseconds(Clock::now() - t1).count()/1e9<<endl;
    }else{

    }
}



void RBF_Core::Set_HermiteRBFSparse(vector<double>& pts, double kernel_dist) {

    cout << "Set_HermiteRBF sparse" << endl;
    //for(auto a:pts)cout<<a<<' ';cout<<endl;
    isHermite = true;
   

    auto t3 = Clock::now();
    std::vector<std::vector<uint32_t>> pt_pair_ids;
    arma::mat dist_simbol = arma::ones(npt, npt) * -1.0;
    for (int i = 0; i < npt; ++i) {
        Pt3f new_p(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        std::vector<uint32_t> results;
        octree.RadiusSearch(new_p, (float)kernel_dist, results);
        pt_pair_ids.push_back(results);
    }
    auto t4 = Clock::now();
    cout << "Octree radius search time: " << (std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;

    std::cout << " use eigen sparse " << use_eigen_sparse << std::endl;

    typedef Eigen::Triplet<double> Tri;
    std::vector<Tri> s_tripletList;
    std::vector<Tri> g_tripletList;
    std::vector<Tri> m_tripletList;

    //a.set_size(npt * 4);
    if (use_eigen_sparse)
    {
        M_es.resize(npt * 4 + 4, npt * 4 + 4);
        F_s_sp.resize(npt, npt* 4 + 4);
        F_g_sp.resize(3 * npt, npt * 4 + 4);
    }
    else {
        M_s.set_size(npt * 4, npt * 4);
    }
    
    double* p_pts = pts.data();
    for (int i = 0; i < npt; ++i) {
        for (auto j : pt_pair_ids[i])
        {
            if (use_eigen_sparse)
            {
                //if (M_es.insert(i, j) != 0) continue;
                double val = Kernal_Function_2p(p_pts + i * 3, p_pts + j * 3);
                Tri ele(i, j, val);
                m_tripletList.push_back(ele);
                s_tripletList.push_back(ele);
               /* M_es.insert(i, j) = val;
                M_es.insert(j, i) = val;
                F_s_sp.insert(i, j) = val;
                F_s_sp.insert(j, i) = val;*/
            }
            else {
                if (M_s(i, j) != 0) continue;
                M_s(i, j) = M_s(j, i) = Kernal_Function_2p(p_pts + i * 3, p_pts + j * 3);
            }
            
        }
    }

    std::cout << " Finish Matrix kernel value " << std::endl;
    double G[3];
    for (int i = 0; i < npt; ++i) {
        for (auto j : pt_pair_ids[i])
        {
            Kernal_Gradient_Function_2p(p_pts + i * 3, p_pts + j * 3, G);
            if (use_eigen_sparse)
            {
                for (int k = 0; k < 3; ++k)
                {
                    Tri ele(i, npt + j + k * npt, G[k]);
                    m_tripletList.push_back(ele);
                    s_tripletList.push_back(ele);

                    Tri ele1(npt + j + k * npt, i, G[k]);
                    m_tripletList.push_back(ele1);

                    Tri ele2(j + k * npt, i, G[k]);
                    g_tripletList.push_back(ele);
                }
                
            }
            else {
                for (int k = 0; k < 3; ++k)M_s(i, npt + j + k * npt) = G[k];
                for (int k = 0; k < 3; ++k)M_s(npt + j + k * npt, i) = G[k];
            }

        }
    }
    std::cout << " Finish Matrix kernel gradient value " << std::endl;

    double H[9];
    for (int i = 0; i < npt; ++i) {
        for (auto j : pt_pair_ids[i])
        {
            /*if (M_s(npt + j, npt + i) != 0) continue;*/
            
            if (use_eigen_sparse)
            {
                Kernal_Hessian_Function_2p(p_pts + i * 3, p_pts + j * 3, H);
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                    {
                        Tri ele(npt + j + l * npt, npt + i + k * npt, -H[k * 3 + l]);
                        m_tripletList.push_back(ele);
                        /*M_es.insert(npt + j + l * npt, npt + i + k * npt) = -H[k * 3 + l];
                        M_es.insert(npt + i + k * npt, npt + j + l * npt) = -H[k * 3 + l];*/
                        Tri ele1(j + l * npt, npt + i + k * npt, -H[k * 3 + l]);
                        g_tripletList.push_back(ele1);
                        /*F_g_sp.insert(j + l * npt, npt + i + k * npt) = -H[k * 3 + l];
                        F_g_sp.insert(i + k * npt, npt + j + l * npt) = -H[k * 3 + l];*/
                    }    
            }
            else {
                if (M_s(npt + j, npt + i) != 0) continue;
                Kernal_Hessian_Function_2p(p_pts + i * 3, p_pts + j * 3, H);
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        M_s(npt + j + l * npt, npt + i + k * npt) =
                        M_s(npt + i + k * npt, npt + j + l * npt) = -H[k * 3 + l];
            }
            
        
        }
        
    }
    std::cout << " Finish Matrix kernel Hessian gradient value " << std::endl;

    //cout<<std::setprecision(5)<<std::fixed<<M<<endl;
    if (use_eigen_sparse)
    {
        /*bsize = 4;
        N.zeros(npt * 4, 4);
        b.set_size(4);*/
        //N_es.resize(npt * 4, 4);
        for (int i = 0; i < npt; ++i) {
            Tri ele(i, npt * 4, 1);
            s_tripletList.push_back(ele);

            for (int j = 0; j < 3; ++j)
            {
                Tri ele1(i, npt * 4 + j, pts[i * 3 + j]);
                s_tripletList.push_back(ele1);
            }

            for (int j = 0; j < 3; ++j)
            {
                Tri ele1(i + j * npt, j + npt * 4 + 1, -1);
                g_tripletList.push_back(ele1);
            }

        }

        
        
    }
    else {
        bsize = 4;
        N.zeros(npt * 4, 4);
        b.set_size(4);

        for (int i = 0; i < npt; ++i) {
            N(i, 0) = 1;
            for (int j = 0; j < 3; ++j)N(i, j + 1) = pts[i * 3 + j];
        }
        for (int i = 0; i < npt; ++i) {
            //        int ind = i*3+npt;
            //        for(int j=0;j<3;++j)N(ind+j,j+1) = 1;

            for (int j = 0; j < 3; ++j)N(npt + i + j * npt, j + 1) = -1;
        }
    }

    if (use_eigen_sparse)
    {
        cout << "S mat size " << s_tripletList.size();
        F_s_sp.setFromTriplets(s_tripletList.begin(), s_tripletList.end());
        cout << " " << F_s_sp.nonZeros() << endl;

        F_g_sp.setFromTriplets(g_tripletList.begin(), g_tripletList.end());
        M_es.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    }
    
    return;
}


double Gaussian_2p(const double *p1, const double *p2, double sigma){

    return exp(-MyUtility::vecSquareDist(p1,p2)/(2*sigma*sigma));
}



void RBF_Core::Set_Actual_User_LSCoef(double user_ls){

    User_Lamnbda = User_Lamnbda_inject = user_ls > 0 ?  user_ls : 0;

}

void RBF_Core::Set_Actual_Hermite_LSCoef(double hermite_ls){

    ls_coef = Hermite_ls_weight_inject = hermite_ls > 0?hermite_ls:0;
}

void RBF_Core::Set_SparsePara(double spa){
    sparse_para = spa;
}

void RBF_Core::Set_User_Lamnda_ToMatrix(double user_ls){
    {
        Set_Actual_User_LSCoef(user_ls);
        auto t1 = Clock::now();
        cout<<"setting K, Hermite Lamnda matrix "<<endl;
        cout<<"User_Lamnbda " << User_Lamnbda <<endl;
        if(User_Lamnbda>0){
            arma::sp_mat eye;
            eye.eye(npt,npt);
            if(apply_sample)
            {
                dI = inv(eye + K00);
                saveK_finalH = K = K11 - (K01.t()*dI*K01);
            } else
            {
                dI = inv(eye + User_Lamnbda*K00);
                saveK_finalH = K = K11 - (User_Lamnbda)*(K01.t()*dI*K01);
            }
            
        }else 
        {
            saveK_finalH = K = K11;
        }
        cout<<"solved: "<<(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;
    }
    finalH = saveK_finalH;
}

void RBF_Core::Set_HermiteApprox_Lamnda(double hermite_ls){


    {
        Set_Actual_Hermite_LSCoef(hermite_ls);
        auto t1 = Clock::now();
        cout<<"setting K, HermiteApprox_Lamnda"<<endl;
        if(ls_coef>0){
            arma::sp_mat eye;
            eye.eye(npt,npt);

            if(ls_coef > 0){
                // if(apply_sample)
                // {
                //     std::cout << "ws size " << Ws.n_rows << " " << Ws.n_cols << std::endl;
                //     if(Ws.n_rows > 0)
                //     {
                //         K = (ls_coef+User_Lamnbda) * Minv + user_beta * Ws;
                //     } else {
                //         K = (ls_coef+User_Lamnbda) * Minv;
                //     }

                //     K00 = K.submat(0,0,npt-1,npt-1);
                //     K01 = K.submat(0,npt,npt-1,npt*4-1);
                //     K11 = K.submat( npt, npt, npt*4-1, npt*4-1 );
                //     arma:: mat tmpdI = inv(eye + K00);
                //     K = K11 - (K01.t()*tmpdI*K01);
                // } else
                {
                    arma:: mat tmpdI = inv(eye + (ls_coef+User_Lamnbda)*K00);
                    K = K11 - (ls_coef+User_Lamnbda)*(K01.t()*tmpdI*K01);
                }
                
            }else{
                K = saveK_finalH;
            }
        }
        cout<<"Approx solved: "<<(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;    
    }
}

void RBF_Core::Set_Hermite_PredictNormal(vector<double>&pts){

    Set_HermiteRBF(pts);
    auto t1 = Clock::now();
    cout<<"setting K"<<endl;

    if(!isnewformula){
        arma::mat D = N.t()*Minv;
        K = Minv - D.t()*inv(D*N)*D;
        K = K.submat( npt, npt, npt*4-1, npt*4-1 );
        finalH = saveK_finalH = K;

    }else{
        cout<<"using new formula"<<endl;
        bigM.zeros((npt+1)*4,(npt+1)*4);
        bigM.submat(0,0,npt*4-1,npt*4-1) = M;
        bigM.submat(0,npt*4,(npt)*4-1, (npt+1)*4-1) = N;
        bigM.submat(npt*4,0,(npt+1)*4-1, (npt)*4-1) = N.t();

        //for(int i=0;i<4;++i)bigM(i+(npt)*4,i+(npt)*4) = 1;

        auto t2 = Clock::now();
        bigMinv = inv(bigM);
        cout<<"bigMinv: "<<(setK_time= std::chrono::nanoseconds(Clock::now() - t2).count()/1e9)<<endl;
		bigM.clear();
        Minv = bigMinv.submat(0,0,npt*4-1,npt*4-1);
        Ninv = bigMinv.submat(0,npt*4,(npt)*4-1, (npt+1)*4-1);

        std::cout << "------------- User lamnbda "<< User_Lamnbda << std::endl;
        if(apply_sample)
        {
            
            if(Ws.n_rows > 0)
            {
                arma::mat Jk = bigMinv.submat(0,0,npt*4 + 3,npt*4-1);
                Ws = As_ * Jk;
                Ws = Ws.t() * Ws; 
                if(auxi_npt > 20 * npt)
                {
                    // user_beta = 2 * double(npt) / double(auxi_npt);
                    user_beta = 1.0;
                }
                if(User_Lamnbda> 0)
                {
                    K = User_Lamnbda * Minv + user_beta * Ws;
                } else {
                    K = Minv + user_beta * Ws;
                }
                
            } else 
            {
                if(User_Lamnbda > 0)
                {
                    K = User_Lamnbda * Minv;
                } else {
                    K =  Minv;
                }
                
            }

            

            // std::cout << " beta " << user_beta << std::endl;
            
            // K = User_Lamnbda * Minv ;
            // K = Minv;
            std::cout << "*************apply_sample K size " << K.n_rows << " " << K.n_cols << std::endl;
        } else {
            K = Minv;
        }

        bigMinv.clear();
        //K = Minv - Ninv *(N.t()*Minv);
        // K = Minv;
        K00 = K.submat(0,0,npt-1,npt-1);
        K01 = K.submat(0,npt,npt-1,npt*4-1);
        K11 = K.submat( npt, npt, npt*4-1, npt*4-1 );

        if(apply_sample && User_Lamnbda <= 1e-12)
        {
            arma::sp_mat eye;
            eye.eye(npt,npt);
            dI = inv(eye + K00);
        }
        

        M.clear();N.clear();
        cout<<"K11: "<<K11.n_cols<<endl;


        //Set_Hermite_DesignedCurve();

        Set_User_Lamnda_ToMatrix(User_Lamnbda_inject);

		
//		arma::vec eigval, ny;
//		arma::mat eigvec;
//		ny = eig_sym( eigval, eigvec, K);
//		cout<<ny<<endl;
        cout<<"K: "<<K.n_cols<<endl;
    }

    //K = ( K.t() + K )/2;
    cout<<"solve K total: "<<(setK_time= std::chrono::nanoseconds(Clock::now() - t1).count()/1e9)<<endl;
    return;

}



void RBF_Core::SetInitnormal_Uninorm(){

    initnormals_uninorm = initnormals;
    for(int i=0;i<npt;++i)MyUtility::normalize(initnormals_uninorm.data()+i*3);

}

int RBF_Core::Solve_Hermite_PredictNormal_UnitNorm(){

    arma::vec eigval, ny;
    arma::mat eigvec;

    if(!isuse_sparse){
        ny = eig_sym( eigval, eigvec, K);
    }else{
//		cout<<"use sparse eigen"<<endl;
//        int k = 4;
//        do{
//            ny = eigs_sym( eigval, eigvec, sp_K, k, "sa" );
//            k+=4;
//        }while(ny(0)==0);
    }


    cout<<"eigval(0): "<<eigval(0)<<endl;

    int smalleig = 0;

    initnormals.resize(npt*3);
    arma::vec y(npt*4);
    for(int i=0;i<npt;++i)y(i) = 0;
    for(int i=0;i<npt*3;++i)y(i+npt) = eigvec(i,smalleig);
    for(int i=0;i<npt;++i){
        initnormals[i*3]   = y(npt+i);
        initnormals[i*3+1] = y(npt+i+npt);
        initnormals[i*3+2] = y(npt+i+npt*2);
        //MyUtility::normalize(normals.data()+i*3);
    }


    SetInitnormal_Uninorm();
    cout<<"Solve_Hermite_PredictNormal_UnitNorm finish"<<endl;
    return 1;
}



/***************************************************************************************************/
/***************************************************************************************************/
double acc_time;

static int countopt = 0;
double optfunc_Hermite(const vector<double>&x, vector<double>&grad, void *fdata){

    auto t1 = Clock::now();
    RBF_Core *drbf = reinterpret_cast<RBF_Core*>(fdata);
    int n = drbf->npt;
    arma::vec arma_x(n*3);

    // std::cout << " .............optfunc_Hermite  000 " << std::endl;

    //(  sin(a)cos(b), sin(a)sin(b), cos(a)  )  a =>[0, pi], b => [-pi, pi];
    vector<double>sina_cosa_sinb_cosb(n * 4);
    for(int i=0;i<n;++i){
        int ind = i*4;
        sina_cosa_sinb_cosb[ind] = sin(x[i*2]);
        sina_cosa_sinb_cosb[ind+1] = cos(x[i*2]);
        sina_cosa_sinb_cosb[ind+2] = sin(x[i*2+1]);
        sina_cosa_sinb_cosb[ind+3] = cos(x[i*2+1]);
    }

    // std::cout << " .............optfunc_Hermite  001 " << std::endl;
    for(int i=0;i<n;++i){
        auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;
        //        int ind = i*3;
        //        arma_x(ind) = p_scsc[0] * p_scsc[3];
        //        arma_x(ind+1) = p_scsc[0] * p_scsc[2];
        //        arma_x(ind+2) = p_scsc[1];
        arma_x(i) = p_scsc[0] * p_scsc[3];
        arma_x(i+n) = p_scsc[0] * p_scsc[2];
        arma_x(i+n*2) = p_scsc[1];
    }

    // std::cout << " .............optfunc_Hermite  002 " << std::endl;
    arma::vec a2;
    //if(drbf->isuse_sparse)a2 = drbf->sp_H * arma_x;
    //else
    //std::cout << " .............opt_incre  "<< drbf->opt_incre << std::endl;
    // if(drbf->apply_sample && drbf->opt_incre)
    // {
    //     a2 = drbf->saveK_finalH_incre * arma_x;
    //     // std::cout << " a2  "<< a2.n_rows << std::endl;
    // } else {
    //     a2 = drbf->finalH * arma_x;
    // }

    a2 = drbf->finalH * arma_x;
    
    //std::cout << " .............optfunc_Hermite  003 " << std::endl;
    if (!grad.empty()) {

        grad.resize(n*2);

        for(int i=0;i<n;++i){
            auto p_scsc = sina_cosa_sinb_cosb.data()+i*4;

            //            int ind = i*3;
            //            grad[i*2] = a2(ind) * p_scsc[1] * p_scsc[3] + a2(ind+1) * p_scsc[1] * p_scsc[2] - a2(ind+2) * p_scsc[0];
            //            grad[i*2+1] = -a2(ind) * p_scsc[0] * p_scsc[2] + a2(ind+1) * p_scsc[0] * p_scsc[3];

            grad[i*2] = a2(i) * p_scsc[1] * p_scsc[3] + a2(i+n) * p_scsc[1] * p_scsc[2] - a2(i+n*2) * p_scsc[0];
            grad[i*2+1] = -a2(i) * p_scsc[0] * p_scsc[2] + a2(i+n) * p_scsc[0] * p_scsc[3];

        }
    }
    //std::cout << " .............optfunc_Hermite  004 " << std::endl;
    double re = arma::dot( arma_x, a2 );
    // std::cout << " residual value :  " << re << std::endl;
    countopt++;

    acc_time+=(std::chrono::nanoseconds(Clock::now() - t1).count()/1e9);

    //cout<<countopt++<<' '<<re<<endl;
    return re;

}



int RBF_Core::Opt_Hermite_PredictNormal_UnitNormal(){


    sol.solveval.resize(npt * 2);

    for(int i=0;i<npt;++i){
        double *veccc = initnormals.data()+i*3;
        {
            //MyUtility::normalize(veccc);
            sol.solveval[i*2] = atan2(sqrt(veccc[0]*veccc[0]+veccc[1]*veccc[1]),veccc[2] );
            sol.solveval[i*2 + 1] = atan2( veccc[1], veccc[0]   );
        }

    }
    //cout<<"smallvec: "<<smallvec<<endl;

    if(1){
        vector<double>upper(npt*2);
        vector<double>lower(npt*2);
        for(int i=0;i<npt;++i){
            upper[i*2] = 2 * my_PI;
            upper[i*2 + 1] = 2 * my_PI;

            lower[i*2] = -2 * my_PI;
            lower[i*2 + 1] = -2 * my_PI;
        }

        countopt = 0;
        acc_time = 0;

        //LocalIterativeSolver(sol,kk==0?normals:newnormals,300,1e-7);
        Solver::nloptwrapper(lower,upper,optfunc_Hermite,this,1e-7,3000,sol);
        cout<<"number of call: "<<countopt<<" t: "<<acc_time<<" ave: "<<acc_time/countopt<<endl;
        callfunc_time = acc_time;
        solve_time = sol.time;
        //for(int i=0;i<npt;++i)cout<< sol.solveval[i]<<' ';cout<<endl;

    }
    newnormals.resize(npt*3);
    arma::vec y(npt*4);
    for(int i=0;i<npt;++i)y(i) = 0;
    for(int i=0;i<npt;++i){

        double a = sol.solveval[i*2], b = sol.solveval[i*2+1];
        newnormals[i*3]   = y(npt+i) = sin(a) * cos(b);
        newnormals[i*3+1] = y(npt+i+npt) = sin(a) * sin(b);
        newnormals[i*3+2] = y(npt+i+npt*2) = cos(a);
        MyUtility::normalize(newnormals.data()+i*3);
    }
    Set_RBFCoef(y);
    //sol.energy = arma::dot(a,M*a);
    cout<<"Opt_Hermite_PredictNormal_UnitNormal"<<endl;
    return 1;
}

void RBF_Core::Opt_Hermite_With_InputNormal()
{
    std::cout << "start to init rbf y " << endl;
    arma::vec y(npt * 4);
    for (int i = 0; i < npt; ++i)y(i) = 0;
    for (int i = 0; i < npt; ++i) {
        y(npt + i) = initnormals[i * 3];
        y(npt + i + npt) = initnormals[i * 3 + 1];
        y(npt + i + npt * 2) = initnormals[i * 3 + 2];
    }
    std::cout << "start to set rbf y " << endl;
    Set_RBFCoef(y);
    std::cout << "finish set RBF coef " << endl;
}

void RBF_Core::CalculateAuxiDistanceVal(const std::string& color_file, bool save_color)
{
    arma::mat func_para =  arma::join_cols(a,b);
    if(As_.n_rows == 0)
    {
        cout << "There is no auxi points. " << endl;
        return;
    }
    std::cout << "func_para mat " << func_para.n_rows << " " << func_para.n_cols << std::endl;
    std::cout << "As mat " << As_.n_rows << " " << As_.n_cols << std::endl;
    auxi_dist_mat = As_ * func_para;
    // std::cout << "dist mat " << dist_mat.n_cols << " " << dist_mat.n_rows << std::endl;
    std::vector<uint8_t> colors;
    auxi_dist_mat = arma::abs(auxi_dist_mat);
   
    // PTSample::FurthestSamplePointCloud(new_pts, incre_num, incre_key_pts, new_auxi_pts);
    // for(const auto val : incre_key_pts)
    // {
    //     pts.push_back(val);
    // }
    // npt = pts.size();
    // auxi_pts = new_auxi_pts;
    // auxi_npt = new_auxi_pts.size();

    double max_dist = auxi_dist_mat.max();
    std::cout << " ----- max dist val 000 : " << max_dist << std::endl;
    if(max_dist < sample_threshold)
    {
        std::cout << " ----- max dist val : " << max_dist << std::endl;
        sample_iter = false;
    }

    max_dist = std::max(1e-8, max_dist);
    
    for(size_t i = 0; i < auxi_npt; ++i)
    {
        double c_val = auxi_dist_mat(i, 0);
        double g_rat =std::max(0.0, max_dist/2.0 - c_val) / (max_dist / 2.0);
        double r_rat =std::max(0.0, c_val - max_dist/2.0) / (max_dist / 2.0);
        uint g_val = uint(g_rat* 255);
        uint r_val = uint(r_rat * 255);

        colors.push_back(r_val);
        colors.push_back(g_val);
        colors.push_back(0);
    }
    // std::string out_color_file = "auxi_color";
    if(save_color)
    {
        writePLYFile_CO(color_file, auxi_pts, colors);
    }
    
}

void RBF_Core::Set_RBFCoef(arma::vec &y){
    cout<<"Set_RBFCoef"<<endl;
    if(curMethod==HandCraft){
        cout<<"HandCraft, not RBF"<<endl;
        return;
    }
    if(!isnewformula){
        b = bprey * y;
        a = Minv * (y - N*b);
    }else{
        // if (apply_sample)
        // {
        //     y.subvec(0, npt - 1) = - dI * K01 * y.subvec(npt, npt * 4 - 1);
        // }
        // else 
        {
            if (User_Lamnbda > 0)y.subvec(0, npt - 1) = -User_Lamnbda * dI * K01 * y.subvec(npt, npt * 4 - 1);
        }
        a = Minv*y;
        b = Ninv.t()*y;
    }

    cout << " b : " << b << endl; 

}



int RBF_Core::Lamnbda_Search_GlobalEigen(){

    vector<double>lamnbda_list({0, 0.001, 0.01, 0.1, 1});
    // vector<double>lamnbda_list({0.001, 0.01, 0.1, 1});
    //vector<double>lamnbda_list({  0.5,0.6,0.7,0.8,0.9,1,1.1,1.5,2,3});
    //lamnbda_list.clear();
    //for(double i=1.5;i<2.5;i+=0.1)lamnbda_list.push_back(i);
    //vector<double>lamnbda_list({0});
    vector<double>initen_list(lamnbda_list.size());
    vector<double>finalen_list(lamnbda_list.size());
    vector<vector<double>>init_normallist;
    vector<vector<double>>opt_normallist;

    lamnbda_list_sa = lamnbda_list;
    for(int i=0;i<lamnbda_list.size();++i){
        cout << "try lambda value " << lamnbda_list[i] << endl;

        Set_HermiteApprox_Lamnda(lamnbda_list[i]);

        cout << "try curMethod " << curMethod << endl;
        if(curMethod==Hermite_UnitNormal){
            Solve_Hermite_PredictNormal_UnitNorm();
        }
        cout << " start  OptNormal " << endl;
        //Solve_Hermite_PredictNormal_UnitNorm();
        OptNormal(1);

        initen_list[i] = sol.init_energy;
        finalen_list[i] = sol.energy;

        init_normallist.emplace_back(initnormals);
        opt_normallist.emplace_back(newnormals);
    }

    lamnbdaGlobal_Be.emplace_back(initen_list);
    lamnbdaGlobal_Ed.emplace_back(finalen_list);

    cout<<std::setprecision(8);
    for(int i=0;i<initen_list.size();++i){
        cout<<lamnbda_list[i]<<": "<<initen_list[i]<<" -> "<<finalen_list[i]<<endl;
    }

    int minind = min_element(finalen_list.begin(),finalen_list.end()) - finalen_list.begin();
    cout<<"min energy: "<<endl;
    cout<<lamnbda_list[minind]<<": "<<initen_list[minind]<<" -> "<<finalen_list[minind]<<endl;


    initnormals = init_normallist[minind];
    SetInitnormal_Uninorm();
    newnormals = opt_normallist[minind];
	return 1;
}




void RBF_Core::Print_LamnbdaSearchTest(string fname){


    cout<<setprecision(7);
    cout<<"Print_LamnbdaSearchTest"<<endl;
    for(int i=0;i<lamnbda_list_sa.size();++i)cout<<lamnbda_list_sa[i]<<' ';cout<<endl;
    cout<<lamnbdaGlobal_Be.size()<<endl;
    for(int i=0;i<lamnbdaGlobal_Be.size();++i){
        for(int j=0;j<lamnbdaGlobal_Be[i].size();++j){
            cout<<lamnbdaGlobal_Be[i][j]<<"\t"<<lamnbdaGlobal_Ed[i][j]<<"\t";
        }
        cout<<gtBe[i]<<"\t"<<gtEd[i]<<endl;
    }

    ofstream fout(fname);
    fout<<setprecision(7);
    if(!fout.fail()){
        for(int i=0;i<lamnbda_list_sa.size();++i)fout<<lamnbda_list_sa[i]<<' ';fout<<endl;
        fout<<lamnbdaGlobal_Be.size()<<endl;
        for(int i=0;i<lamnbdaGlobal_Be.size();++i){
            for(int j=0;j<lamnbdaGlobal_Be[i].size();++j){
                fout<<lamnbdaGlobal_Be[i][j]<<"\t"<<lamnbdaGlobal_Ed[i][j]<<"\t";
            }
            fout<<gtBe[i]<<"\t"<<gtEd[i]<<endl;
        }
    }
    fout.close();

}



void RBF_Core::clearMemory()
{
    K.clear();
    As_.clear();
    Ws.clear();
    finalH.clear();
    saveK_finalH.clear();
    saveK_finalH_incre.clear();
    K_incre_.clear();
    Minv.clear();
    dI.clear();
    M.clear();
    N.clear();

    K00.clear();
    K01.clear();
    K11.clear();

    Ninv.clear();
    Minv.clear();
    
}