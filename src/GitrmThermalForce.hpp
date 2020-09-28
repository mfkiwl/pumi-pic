#ifndef THERMAL_FORCE_H
#define THERMAL_FORCE_H
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"

inline void gitrm_thermal_force(PS* ptcls, int *iteration, const GitrmMesh& gm,
const GitrmParticles& gp, double dt, const o::LOs& elm_ids, int debug=0) {
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();
  if(debug)
      printf("Thermal Force\n");
  
  //Mesh data regarding the gradient of temperatures
  const auto& gradTi_d = gm.getGradTi();
  const auto& gradTiGridX = gm.getIonTempGrad2dGrid(1);
  const auto& gradTiGridZ = gm.getIonTempGrad2dGrid(2);
  //Setting up of 2D magnetic field data 
  const auto& BField_2d = gm.getBfield2d();
  const auto& bGridX = gm.getBfield2dGrid(1);
  const auto& bGridZ = gm.getBfield2dGrid(2);
  auto& mesh = gm.mesh;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto gradTionVtx = mesh.get_array<o::Real>(o::VERT, "gradTiVtx");
  const auto BField = o::Reals(); //o::Reals(mesh.get_array<o::Real>(o::VERT, "BField"));

  auto useConstantBField = gm.isUsingConstBField();
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto use3dField = USE3D_BFIELD;
  bool cylSymm = true;

  const double amu = 184; 
  const double background_amu = 4;
  const auto MI     = 1.6737236e-27;

  const auto iTimeStep = iTimePlusOne - 1;
  auto& xfaces =gp.wallCollisionFaceIds;

  auto update_thermal = PS_LAMBDA(const int& e, const int& pid, const bool& mask) { 
    if(mask > 0 && elm_ids[pid] >= 0) {	
      o::LO el = elm_ids[pid];
      auto posit          = p::makeVector3(pid, x_ps_d);
      auto ptcl           = pid_ps(pid);
      auto charge         = charge_ps_d(pid);
      auto fid            = xfaces[pid];
      if(!charge || fid >=0)
          return;

      auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
      auto eField         = p::makeVector3(pid, efield_ps_d);
      auto vel            = p::makeVector3(pid, vel_ps_d);
      auto bField         = o::zero_vector<3>();

      if (use2dInputFields || useConstantBField){
        p::interp2dVector_wgrid(BField_2d, bGridX, bGridZ, posit_next, bField, cylSymm);
      } else if (use3dField){
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
        p::interpolate3dFieldTet(mesh2verts, BField, el, bcc, bField); 
      }
      auto b_mag  = Omega_h::norm(bField);
      Omega_h::Vector<3> b_unit = bField/b_mag;

      auto pos2D          = o::zero_vector<3>();
      pos2D[0]            = sqrt(posit_next[0]*posit_next[0] + posit_next[1]*posit_next[1]);
      pos2D[1]            = 0;
      pos2D[2]            = posit_next[2];

      auto gradti         = o::zero_vector<3>();

      //find out the gradients of the electron and ion temperatures at that particle poin
      if (use2dInputFields) {
        p::interp2dVector_wgrid(gradTi_d, gradTiGridX, gradTiGridZ, posit_next, gradti, true);
      } else if (use3dField){
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
        p::interpolate3dFieldTet(mesh2verts, gradTionVtx, el, bcc, gradti);
      }
      
      o::Real mu = amu /(background_amu+amu);
      o::Real beta = 3 * (mu + 5*sqrt(2.0)*charge*charge*(1.1*pow(mu, (5 / 2))-0.35*pow(mu,(3/2)))-1)/
                     (2.6 - 2*mu + 5.4*pow(mu, 2));
      auto dv_ITG = o::zero_vector<3>();
      for(int i=0; i<3; ++i)
        dv_ITG[i] = 1.602e-19*dt/(amu*MI)*beta* gradti[i]*b_unit[i];
      
      if (debug > 2) {
        printf("ptcl %d t %d B %g %g %g \n", ptcl, iTimeStep, bField[0], bField[1], bField[2]);
        printf("ptcl %d t %d GradTi %g %g %g \n", ptcl, iTimeStep, gradti[0], gradti[1], gradti[2]);
        printf("ptcl  %d timestep %d ITG %.16e %.16e %.16e \n",
          ptcl, iTimeStep, dv_ITG[0], dv_ITG[1], dv_ITG[2]);
        printf(" ptcl %d timestep %d charge %d amu %g background_amu %g \n", 
          ptcl,iTimeStep, charge, amu, background_amu);
        printf("Position partcle %d timestep %d is %.15e %.15e %.15e \n", ptcl,
          iTimeStep, posit_next[0],posit_next[1],posit_next[2]);
      }
      auto vf = vel + dv_ITG;
      for(int i=0; i<3; ++i)
        vel_ps_d(pid,i) = vf[i];
      
      if(debug > 1) {
        printf("thermalforce: ptcl %d timestep %d vel %.15f %.15f %.15f  => "
          "  %.15f %.15f %.15f \n", ptcl, iTimeStep, vel[0],vel[1],vel[2], 
          vf[0],vf[1],vf[2]);
      } 
    }
  };
  p::parallel_for(ptcls, update_thermal, "thermal_force_kernel");
    
}
#endif
