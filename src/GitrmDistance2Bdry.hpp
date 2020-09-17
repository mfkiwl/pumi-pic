#ifndef GITRM_DISTANCE2BDRY_HPP
#define GITRM_DISTANCE2BDRY_HPP

#include "GitrmParticles.hpp"

/** @brief Calculate distance of particles to domain boundary.
 * Not yet clear if a pre-determined depth can be used
*/


inline void gitrm_findDistanceToBdry(GitrmParticles& gp,
   GitrmMesh& gm, int debug=0) {
  if(debug)
    printf("gitrm_findDistanceToBdry \n");
  int tstep = iTimePlusOne;
  int rank = gp.getCommRank();
  auto* ptcls = gp.ptcls;
  auto* b = gm.getBdryPtr();
  const auto bdryFaceVertCoordsPic = b->getBdryFaceVertCoordsPic();
  const auto bdryFaceVertsPic = b->getBdryFaceVertsPic();
  const auto bdryFaceIdsPic = b->getBdryFaceIdsPic();
  const auto bdryFaceIdPtrsPic = b->getBdryFaceIdPtrsPic();
  const auto origGlobalIds = b->getOrigGlobalIds();

  //if true bdryFaceIdsPic has all material bdry face ids to use for each elem
  const int allMatBdryData = b->isAllMatBdryData();
  //valid if fullMesh, since this array contains all of those selected boundary
  //faceIds in all picparts. Used to compare fid in GITR mesh
  const auto bdryFaceOrderedIds = gm.getBdryFaceOrderedIds();
  auto* fullMesh = gm.getFullMesh();
  const auto full_f2rPtr = fullMesh->ask_up(o::FACE, o::REGION).a2ab;
  const auto full_f2rElem = fullMesh->ask_up(o::FACE, o::REGION).ab2b;

  const auto psCapacity = ptcls->capacity();
  o::Write<o::Real> closestPoints(psCapacity*3, 0, "closest_points");
  o::Write<o::LO> closestBdryFaceIds(psCapacity, -1, "closest_fids");
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto vel_ps = ptcls->get<PTCL_VEL>();

  if(debug)
    MPI_Barrier(MPI_COMM_WORLD);
  if(debug > 1)
    std::cout << " nbdryFaceIdsPic " << bdryFaceIdsPic.size() << " nbdryFaceIdPtrsPic "
      << bdryFaceIdPtrsPic.size() << " nbdryFaceVertCoordsPic " <<
      bdryFaceVertCoordsPic.size() << " nbdryFaceVertsPic " << bdryFaceVertsPic.size() << "\n";

  //The mesh data used may not be from a full mesh, specifically in the case
  // where the mesh data stored only to support dist-to-bdry calculation.
  // This means only limited omega_h functions can be called with the data.
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if (mask > 0) {
      auto ptcl = pid_ps(pid);
      o::LO beg = 0;
      o::LO nFaces = 0;
      o::LO end = 0;

      //bdry data read from file, or calculated in this run, were selected and stored
      //only for the elements of this picpart
      beg = bdryFaceIdPtrsPic[elem];
      end = bdryFaceIdPtrsPic[elem+1];
      //set -1's for brute-force calculation
      beg = (beg == -1) ? 0 : beg;
      end = (end == -1) ? bdryFaceIdsPic.size(): end;
      nFaces = end - beg;

      if(!nFaces || debug > 3)
        printf("%d: ptcl %d tstep %d elem %d gid %ld nFaces %d beg %d end %d\n",
         rank, ptcl, tstep, elem, origGlobalIds[elem],  nFaces, beg, end);
      OMEGA_H_CHECK(nFaces);

      double dist = 0;
      double min = 1.0e+30;
      o::LO bfid = -1;
      o::LO fid = -1;
      auto ref = p::makeVector3(pid, pos_ps);
      auto point = o::zero_vector<3>();
      auto pt = o::zero_vector<3>();
      o::Matrix<3,3> face;
      for(o::LO ii = 0; ii < nFaces; ++ii) {
        o::LO ind = beg + ii;
        bfid = bdryFaceIdsPic[ind];
        face = p::get_face_coords_of_tet(bdryFaceVertsPic, bdryFaceVertCoordsPic, bfid);
        int region = -1;
        pt = p::closest_point_on_triangle(face, ref, &region);
        dist = o::norm(pt - ref);

        //use old method of full mesh and read data to verify
        if(debug > 3) {
          //valid only if this local bfid was used while creating this, which is
          //true in the case of fullMesh
          auto ordId = bdryFaceOrderedIds[bfid];
          printf("p %d e %d ii %d bfid %d dist %g ordId %d pt %g %g %g\n",
              ptcl, elem, ii, bfid, dist, ordId,  pt[0], pt[1], pt[2]);
          printf("p %d ii %d face %g %g %g ; %g %g %g ; %g %g %g\n",
              ptcl, ii, face[0][0], face[0][1], face[0][2],
              face[1][0], face[1][1], face[1][2],face[2][0], face[2][1], face[2][2]);
        } //end of debug

        if(dist < min) {
          min = dist;
          fid = bfid;
          for(int i=0; i<3; ++i)
            point[i] = pt[i];
        }
        OMEGA_H_CHECK(fid >= 0);
      } //for nFaces

      closestBdryFaceIds[pid] = fid;
      for(o::LO j=0; j<3; ++j)
        closestPoints[pid*3+j] = point[j];

      if(debug >2) {
        auto f = p::get_face_coords_of_tet(bdryFaceVertsPic, bdryFaceVertCoordsPic, fid);
        auto bel = -1;
        //fullMesh data used (no selective bdry data stored) in this brute-force case
        if(allMatBdryData)
          bel = p::elem_id_of_bdry_face_of_tet(fid, full_f2rPtr, full_f2rElem); 
        printf("%d: dist: ptcl %d tstep %d el %d MINdist %.15e nf %d bfid %d"
          " ordId %d bel %d\n", rank, ptcl, tstep, elem, min, nFaces, fid,
          bdryFaceOrderedIds[fid], bel);
        printf("%d: dist: ptcl %d tstep %d closest-pt %.15f %.15f %.15f \n",
          rank, ptcl, tstep, point[0], point[1],point[2]);
        printf("%d: dist: ptcl %d tstep %d pos %.15e %.15e %.15e \n face %g %g %g :"
          " %g %g %g : %g %g %g\n", rank, ptcl, tstep, ref[0], ref[1], ref[2],
          f[0][0], f[0][1], f[0][2], f[1][0],f[1][1], f[1][2],f[2][0],f[2][1],f[2][2]);
      } //debug

      auto vel = p::makeVector3(pid, vel_ps);
      if(debug > 1 || isnan(ref[0]) || isnan(ref[1]) || isnan(ref[2])
        || isnan(vel[0]) || isnan(vel[1]) || isnan(vel[2])) {
        printf("%d: ptcl %d tstep %d pos %g %g %g vel %g %g %g\n", rank, ptcl, tstep,
          ref[0], ref[1], ref[2], vel[0], vel[1], vel[2]);
      }
    } //mask
  };

  p::parallel_for(ptcls, lambda, "dist2bdry_kernel");
  gp.closestPoints = o::Reals(closestPoints);
  gp.closestBdryFaceIds = o::LOs(closestBdryFaceIds);
}


inline void GITR_findDistanceToBdry(GitrmParticles& gp, GitrmMesh& gm, int debug=0) {

  if(debug)
    printf("GITR_findDistanceToBdry \n");
  int tstep = iTimePlusOne;
  int rank = gp.getCommRank();
  auto* ptcls = gp.ptcls;
  auto* b = gm.getBdryPtr();
  const auto bdryFaceVertCoordsPic = b->getBdryFaceVertCoordsPic();
  const auto bdryFaceVertsPic = b->getBdryFaceVertsPic();
  const auto bdryFaceIdsPic = b->getBdryFaceIdsPic();
  const auto bdryFaceIdPtrsPic = b->getBdryFaceIdPtrsPic();
  const auto origGlobalIds = b->getOrigGlobalIds();
  const auto bdryFaceOrderedIds = gm.getBdryFaceOrderedIds();
  auto* fullMesh = gm.getFullMesh();
  const auto full_f2rPtr = fullMesh->ask_up(o::FACE, o::REGION).a2ab;
  const auto full_f2rElem = fullMesh->ask_up(o::FACE, o::REGION).ab2b;

  const auto psCapacity = ptcls->capacity();
  o::Write<o::Real> closestPoints(psCapacity*3, 0, "closest_points");
  o::Write<o::LO> closestBdryFaceIds(psCapacity, -1, "closest_fids");
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto vel_ps = ptcls->get<PTCL_VEL>();
  double dMin = 1e12;

  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if (mask > 0) {
      auto ptcl = pid_ps(pid);
      o::LO beg = 0;
      o::LO nFaces = 0;
      o::LO end = 0;
      beg = bdryFaceIdPtrsPic[elem];
      end = bdryFaceIdPtrsPic[elem+1];
      //set -1's for brute-force calculation
      beg = (beg == -1) ? 0 : beg;
      end = (end == -1) ? bdryFaceIdsPic.size(): end;
      nFaces = end - beg;

      if(!nFaces || debug > 3)
        printf("%d: ptcl %d tstep %d elem %d gid %ld nFaces %d beg %d end %d\n",
         rank, ptcl, tstep, elem, origGlobalIds[elem],  nFaces, beg, end);
      OMEGA_H_CHECK(nFaces);

      double minDistance = dMin;
      int minIndex = -1;
      int minFid = -1;
      auto p0 = p::makeVector3(pid, pos_ps);
      auto dirUnitVec = o::zero_vector<3>();
      auto point = o::zero_vector<3>();
      auto pt = o::zero_vector<3>();

      for(o::LO fi = 0; fi < nFaces; ++fi) {
        //if(matZ <= 0)
        //  continue;
        o::LO ind = beg + fi;
        auto bfid = bdryFaceIdsPic[ind];
        auto ord = bdryFaceOrderedIds[bfid];
        auto face = p::get_face_coords_of_tet(bdryFaceVertsPic, bdryFaceVertCoordsPic, bfid);
        auto normalVec = o::cross((face[1]-face[0]), (face[2]-face[0]));
        auto plane_d = -(o::inner_product(normalVec, face[0]));
        auto plane_norm = o::norm(normalVec);

        auto pt2PlaneDist = (normalVec*p0 + plane_d)/plane_norm;
        normalVec = normalVec/plane_norm;
        pt = p0 - pt2PlaneDist*normalVec;
        auto A = face[0];
        auto B = face[1];
        auto C = face[2];
        auto AB = B - A;
        auto AC = C - A;
        auto BC = C - B;
        auto CA = A - C;
        auto AP = pt - A;
        auto BP = pt - B;
        auto CP = pt - C;
        normalVec = o::cross(AB, AC);
        auto ABXAP = o::cross(AB, AP);
        auto BCXBP = o::cross(BC, BP);
        auto CAXCP = o::cross(CA, CP);

        auto signDot0 = copysign(1.0, o::inner_product(ABXAP, normalVec));
        auto signDot1 = copysign(1.0, o::inner_product(BCXBP, normalVec));
        auto signDot2 = copysign(1.0, o::inner_product(CAXCP, normalVec));

        auto totalSigns = abs(signDot0 + signDot1 + signDot2);
        auto p0A = A - p0;
        auto p0B = B - p0;
        auto p0C = C - p0;
        auto p0Anorm = o::norm(p0A);   
        auto p0Bnorm = o::norm(p0B);   
        auto p0Cnorm = o::norm(p0C);

        auto distances = o::zero_vector<7>();
        auto normals = o::zero_vector<21>();
        auto closestAll = o::zero_vector<21>();
        distances[1] = p0Anorm;
        distances[2] = p0Bnorm;   
        distances[3] = p0Cnorm;

        for(int i=3, j=0; i<6; ++i)
          closestAll[i] = A[j++]; 
        for(int i=6, j=0; i<9; ++i)
          closestAll[i] = B[j++];
        for(int i=9, j=0; i<12; ++i)
          closestAll[i] = B[j++];
        for(int i=3, j=0; i<6; ++i)
          normals[i] = p0A[j++]/p0Anorm;
        for(int i=6, j=0; i<9; ++i)
          normals[i] = p0B[j++]/p0Bnorm;
        for(int i=9, j=0; i<12; ++i)
          normals[9] = p0C[j++]/p0Cnorm;

        auto normAB = o::norm(AB);
        auto normBC = o::norm(BC);
        auto normCA = o::norm(CA);
        auto ABhat = AB/normAB;
        auto BChat = BC/normBC; 
        auto CAhat = CA/normCA;
             
        auto tAB =  o::inner_product(p0A,ABhat);
        auto tBC = o::inner_product(p0B,BChat);
        auto tCA = o::inner_product(p0C,CAhat);
        tAB = -1*tAB;
        tBC = -1*tBC;
        tCA = -1*tCA;

        if(tAB > 0 && tAB < normAB) {
          auto projP0AB = tAB*ABhat;
          projP0AB = A + projP0AB;
          auto p0AB = projP0AB - p0;
          auto p0ABdist = o::norm(p0AB);
          distances[4] = p0ABdist;   
          for(int i=12, j=0; i<15; ++i)
            normals[i] = p0AB[j++]/p0ABdist;

          if(debug > 3) 
           printf("d2bdry: ptcl %d timestep %d ord %d (tAB > 0.0) && (tAB < normAB) distances[4] %g\n", 
             ptcl, tstep, ord, distances[4]);
        } else {
          distances[4] = dMin;   
        } 
     
        if(tBC > 0 && tBC < normBC) {
          auto projP0BC = tBC*ABhat;
          projP0BC = B + projP0BC;
          auto p0BC = projP0BC - p0;
          auto p0BCdist = o::norm(p0BC);
          distances[5] = p0BCdist;   
          for(int i=15, j=0; i<18; ++i)
            normals[i] = p0BC[j++]/p0BCdist;
          if(debug > 3) {
              printf("d2bdry: ptcl %d tstep %d  ord %d (tBC > 0.0) && (tBC < normBC) distances[5] %g\n", 
                ptcl, tstep, ord, distances[5]);
              printf("d2bdry: ptcl %d tstep %d ord %d tBC %g ABhat %g %g %g projP0BC %g %g %g  p0BC %g %g %g \n",
                  ptcl, tstep, ord, tBC, ABhat[0], ABhat[1], ABhat[2], projP0BC[0],
                  projP0BC[1], projP0BC[2], p0BC[0], p0BC[1], p0BC[2]);
            }
        } else {
           distances[5] = dMin;   
        } 
          
        if((tCA > 0.0) && (tCA < normCA)) {
           auto projp0CA = tCA*CAhat;
           projp0CA = C + projp0CA;
           auto p0CA = projp0CA - p0;
           auto p0CAdist = o::norm(p0CA);
           distances[6] = p0CAdist;
           for(int i=18, j=0; i<21 ; ++i) 
             normals[i] = p0CA[j++]/p0CAdist;
           if(debug > 3)  
             printf("d2bdry: ptcl %d timestep %d ord %d (tCA > 0.0) && (tCA < normCA)  distances[6] %g\n", 
                 ptcl, tstep, ord, distances[6]);
        } else {
           distances[6] = dMin;  
        } 

        if (totalSigns == 3.0) {
            auto perpDist = abs(pt2PlaneDist); 
            normalVec = pt - p0;
            auto normVec = o::normalize(normalVec);
            distances[0] = perpDist;   
            for(int i=0; i<3; ++i)
              normals[i] = normVec[i];
            if(debug > 3)
              printf("d2bdry: ptcl %d tstep %d ord %d signs3  distances[0] %g\n",
                  ptcl, tstep, ord, distances[0]);
        } else {
            distances[0] = dMin;  
        }
        int index = 0;
        for(int j = 0; j < 7; j++) {
          if(distances[j] < distances[index]) {
             index = j;
           }
        }

        if(distances[index] < minDistance) {
          minDistance = distances[index];
          for(int i=0; i<3; ++i)
            dirUnitVec[i] = normals[index*3+i];
          minIndex = ind;
          minFid = bfid;
        }

        point = p0 - minDistance*dirUnitVec;
        if(debug>3) {
          printf("d2bdry ptcl %d tstep %d  ind %d ordfid %d dist %.15f minDis %.15f minIndex %d minFid %d\n" 
             " planePt %.15f %.15f %.15f \n", ptcl, tstep, ind, ord, distances[index],
            minDistance, minIndex, minFid, pt[0], pt[1], pt[2]);
          printf(" d2bdry ptcl %d tstep %d face %g %g %g : %g %g %g : %g %g %g\n closest %.15f %.15f %.15f\n",
            ptcl, tstep, face[0][0], face[0][1], face[0][2], face[1][0], face[1][1], face[1][2],
            face[2][0], face[2][1], face[2][2], point[0], point[1], point[2]);
        }
      } //nfaces

      closestBdryFaceIds[pid] = minFid;
      for(o::LO j=0; j<3; ++j)
        closestPoints[pid*3+j] = point[j];

      if(debug > 2) { 
        auto face = p::get_face_coords_of_tet(bdryFaceVertsPic, bdryFaceVertCoordsPic, minIndex);
        auto ordId = bdryFaceOrderedIds[minIndex];
        printf("\nd2bdry::minDist %.15f ptcl %d tstep %d ordId %d minind %d minFid %d"
          " face %g %g %g : %g %g %g : %g %g %g\n", minDistance, ptcl, tstep,
          ordId, minIndex, minFid, face[0][0], face[0][1], face[0][2], face[1][0], face[1][1],
          face[1][2], face[2][0], face[2][1], face[2][2]);
     
        printf("d2bdry::ptcl %d tstep %d pos %.15f %.15f %.15f closest-pt"
          " %.15f %.15f %.15f\n", ptcl, tstep, p0[0], p0[1], p0[2], point[0], point[1], point[2]);
      }
    }//mask
  };
  p::parallel_for(ptcls, lambda, "dist2bdry_kernel");
  gp.closestPoints = o::Reals(closestPoints);
  gp.closestBdryFaceIds = o::LOs(closestBdryFaceIds);
}

#endif
