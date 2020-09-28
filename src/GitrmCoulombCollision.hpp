#ifndef GITRM_COULOMB_COLLISION_H
#define GITRM_COULOMB_COLLISION_H
#include "GitrmParticles.hpp"

OMEGA_H_DEVICE void getSlowDownFrequencies(const Omega_h::Vector<3> &relVel,
double& nu_friction, double& nu_deflection, double& nu_parallel, double& nu_energy,
const double temp_i, const double temp_el, const double dens, const double charge,
const int background_Z, const double background_amu, const double amu,
int ptcl, int tstep, int debug) {
  const double Q    = 1.60217662e-19;
  const double EPS0 = 8.854187e-12;
  const double pi   = 3.14159265;
  const auto MI     = 1.6737236e-27;
  const auto ME     = 9.10938356e-31;

  auto velNorm = Omega_h::norm(relVel);
  if(debug >2)
    printf("relVel %g %g %g\n", relVel[0], relVel[1], relVel[2]);
  auto lam_d = sqrt(EPS0*temp_el/(dens*pow(background_Z,2)*Q));
  auto lam = 12*pi*dens*pow(lam_d,3)/charge;
  auto gam_electron_background = 0.238762895*pow(charge,2)*log(lam)/(amu*amu);
  if(gam_electron_background < 0.0){
      gam_electron_background=0.0;
  }

  auto gam_ion_background = 0.238762895*pow(charge,2)*pow(background_Z,2)*log(lam)/(amu*amu);
  if(gam_ion_background < 0.0){
    gam_ion_background=0.0;
  }
  if(debug >2)
    printf("lam_d %g lam %g gam_el_bg %g gam_ion_bg %g \n", lam_d, lam,
        gam_electron_background, gam_ion_background);
  auto xx_i= pow(velNorm,2)*background_amu*MI/(2*temp_i*Q);
  auto xx_e= pow(velNorm,2)*ME/(2*temp_el*Q);

  auto psi_prime_i = 2*sqrt(xx_i/pi)*exp(-xx_i);
  auto psi_psiprime_i = erf(sqrt(xx_i));
  auto psi_i = psi_psiprime_i - psi_prime_i;
  auto psi_e = 0.75225278*pow(xx_e,1.5);
  auto psi_prime_e = 1.128379*sqrt(xx_e);
  //auto psi_psiprime_e = psi_e + psi_prime_e;
  auto psi_psiprime_psi2x_e = 1.128379*sqrt(xx_e)*exp(-xx_e);

  auto nu_0_i = gam_electron_background*dens/pow(velNorm,3);
  auto nu_0_e = gam_ion_background*dens/pow(velNorm,3);

  auto nu_friction_i = (1+amu/background_amu)*psi_i*nu_0_i;
  auto nu_deflection_i = 2*(psi_psiprime_i - psi_i/(2*xx_i))*nu_0_i;

  auto nu_parallel_i = psi_i/xx_i*nu_0_i;
  auto nu_energy_i = 2*(amu/background_amu*psi_i - psi_prime_i)*nu_0_i;
  nu_friction = nu_friction_i; // + nu_friction_e;
  nu_deflection = nu_deflection_i; // + nu_deflection_e;
  nu_parallel = nu_parallel_i; // + nu_parallel_e;
  nu_energy = nu_energy_i; // + nu_energy_e;

  if (temp_i <= 0 || temp_el<=0){
    nu_friction = 0;
    nu_deflection = 0;
    nu_parallel = 0;
    nu_energy = 0;
  }
  if (dens <= 0){
    nu_friction = 0;
    nu_deflection = 0;
    nu_parallel = 0;
    nu_energy = 0;
  }

  if(debug > 2){
    printf("ptcl %d tstep %d NU_friction %.15f NU_deflection %.15f\n",ptcl,
      tstep, nu_friction, nu_deflection);
    printf("ptcl %d tstep %d NU_parallel %.15f NU_energy %.15f\n", ptcl,
      tstep, nu_parallel, nu_energy);
    printf("ptcl %d tstep %d Ion-temp %.15f el-temp %.15f ion-density %g \n",ptcl,
      tstep, temp_i, temp_el,dens);
  }
}

OMEGA_H_DEVICE void getSlowDownDirections2(const Omega_h::Vector<3> &relvel,
 Omega_h::Vector<3> &parallel_dir, Omega_h::Vector<3> &perp1_dir,
 Omega_h::Vector<3> &perp2_dir, int ptcl, int tstep, int debug){

  auto rvel = relvel;
  o::Real velNorm   = Omega_h::norm(relvel);
  if(p::almost_equal(velNorm, 0)) {
    velNorm = 1;
    rvel[2] = 1;
    rvel[0] = 0;
    rvel[1] = 0;
  }
  parallel_dir = rvel/velNorm;
  if(debug > 2)
    printf(" rvel %g %g %g velNorm %g \n", rvel[0], rvel[1], rvel[2], velNorm);

  //perpendicular velocity unit vectors, from (ez1,ez2,ez3)x(0,0,1)
  auto zdir = o::zero_vector<3>();
  zdir[2] = 1;
  perp1_dir = o::cross(parallel_dir, zdir);
  // The above cross product will be zero for particles with a pure 
  // z-directed (ez3) velocity. Find those particles and get the perpendicular 
  // unit vectors by taking the cross product (ez1,ez2,ez3)x(0,1,0) instead
  double exnorm = o::norm(perp1_dir);
  if(abs(exnorm) < 1.0e-12) {
    auto ydir = o::zero_vector<3>();
    ydir[1] = 1;
    perp1_dir = o::cross(parallel_dir, ydir);
  }
  // Ensure all the perpendicular direction vectors are unit
  perp1_dir = o::normalize(perp1_dir);

  if(isnan(perp1_dir[0]) || isnan(perp1_dir[1]) || isnan(perp1_dir[2]))
    printf("perp_dir nan: parallel %f %f %f parallel_norm %f\n", parallel_dir[0],
      parallel_dir[1], parallel_dir[2], velNorm);

  // Find the second perpendicular direction by (ez1,ez2,ez3)x(ex1,ex2,ex3)
  perp2_dir = o::cross(parallel_dir, perp1_dir);

  if(debug > 2){
    printf("ptcl %d tstep %d parallel_dir %.15f %.15f %.15f\n",ptcl,tstep,
      parallel_dir[0], parallel_dir[1], parallel_dir[2]);
    printf("ptcl %d tstep %d perp1_dir %.15f %.15f %.15f\n",ptcl,tstep,
      perp1_dir[0], perp1_dir[1], perp1_dir[2]);
    printf("ptcl %d tstep %d perp2_dir %.15f %.15f %.15f\n",ptcl,tstep,
      perp2_dir[0], perp2_dir[1], perp2_dir[2]);
  }
}

    
inline void gitrm_coulomb_collision(PS* ptcls, int *iteration, const GitrmMesh& gm,
   const GitrmParticles& gp, double dt, const o::LOs& elm_ids, int debug=0) {
  
  const double pi = 3.14159265;

  auto pid_ps_global=ptcls->get<PTCL_ID_GLOBAL>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto x_ps_d = ptcls->get<PTCL_POS>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto efield_ps_d  = ptcls->get<PTCL_EFIELD>();
  auto vel_ps_d = ptcls->get<PTCL_VEL>();
  auto charge_ps_d = ptcls->get<PTCL_CHARGE>();

  auto useConstantFlowVelocity = gm.isUsingConstFlowVel();
  auto useConstantBField = gm.isUsingConstBField();
  auto use2dInputFields = USE2D_INPUTFIELDS;
  auto use3dField = USE3D_BFIELD;
  bool cylSymm = true;

  const double amu = gm.getImpurityAmu();
  const double background_amu = gm.getBackgroundAmu();
  const int background_Z = gm.getBackgroundZ();
  if(debug)
    std::cout << "COULOMB collision ConstFlowVel " << useConstantFlowVelocity << "\n";
  o::Reals constFlowVelocity(3);
  if(useConstantFlowVelocity) {
    constFlowVelocity = o::Reals(o::Write<o::Real>(o::HostWrite<o::Real>(
      {CONSTANT_FLOW_VELOCITY0, CONSTANT_FLOW_VELOCITY1, CONSTANT_FLOW_VELOCITY2})));
  }
  //flow velocity
  const auto flowVel_2d = gm.getFlowVel2d();
  const auto& velGridX = gm.getFlowVel2dGrid(1);
  const auto& velGridZ = gm.getFlowVel2dGrid(2);
 
  const auto flowVX0 = gm.getFlowVelX0();
  const auto flowVZ0 = gm.getFlowVelZ0();
  const auto flowVNx = gm.getFlowVelNx();
  const auto flowVNz = gm.getFlowVelNz();
  const auto flowVDx = gm.getFlowVelDz();
  const auto flowVDz = gm.getFlowVelDz();

  //Setting up of 2D magnetic field data
  const auto BField_2d = gm.getBfield2d();
  const auto& bGridX = gm.getBfield2dGrid(1);
  const auto& bGridZ = gm.getBfield2dGrid(2);
  //Setting up of 2D ion temperature data
  const auto temIon_d = gm.getTemIon();
  const auto& ionTempGridX = gm.getIonTemp2dGrid(1);
  const auto& ionTempGridZ = gm.getIonTemp2dGrid(2);
  //Setting up of 2D ion density data
  //const auto densIon_d = gm.getDensIon();
  const auto densEl_d = gm.getDensEl();
  const auto& densGridX = gm.getElDens2dGrid(1);
  const auto& densGridZ = gm.getElDens2dGrid(2);

  //Setting up of 2D electron temperature data
  const auto temEl_d  = gm.getTemEl();
  const auto& elTempGridX = gm.getElTemp2dGrid(1);
  const auto& elTempGridZ = gm.getElTemp2dGrid(2);
  //Data for 3D interpolation
  auto& mesh = gm.mesh;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto tIonVtx = mesh.get_array<o::Real>(o::VERT, "IonTempVtx");
  const auto densVtx = mesh.get_array<o::Real>(o::VERT, "IonDensityVtx");
  const auto tElVtx  = mesh.get_array<o::Real>(o::VERT, "ElTempVtx");
  const auto BField = o::Reals(); //o::Reals(mesh.get_array<o::Real>(o::VERT, "BField"));

  //Use of GITR generated random numbers
  const auto& testGitrPtclStepData = gp.testGitrPtclStepData;
  const auto testGDof = gp.testGitrStepDataDof;
  const auto testGNT = gp.testGitrStepDataNumTsteps;
  //const auto testGIind = gp.testGitrDataIoniRandInd;
  const auto tstep = iTimePlusOne - 1;
  const auto collisionIndex1 = gp.testGitrCollisionRndn1Ind;
  const auto collisionIndex2 = gp.testGitrCollisionRndn2Ind;
  const auto collisionIndex3 = gp.testGitrCollisionRndxsiInd;
  auto& xfaces =gp.wallCollisionFaceIds;

  const int useGitrRnd = gp.useGitrRndNums;
  auto& rpool = gp.rand_pool;

  const bool useCudaRnd = gp.useCudaRnd;
  auto* cuStates =  gp.cudaRndStates;

  auto lambda = PS_LAMBDA(const int& e, const int& pid, const bool& mask) {
    if(mask > 0 && elm_ids[pid] >= 0) {
      o::LO el = elm_ids[pid];
      auto ptcl           = pid_ps(pid);
      auto ptcl_global    = pid_ps_global(pid);
      auto charge         = charge_ps_d(pid);
      auto fid            = xfaces[pid];
      if(!charge || fid >=0)
        return;

      auto posit          = p::makeVector3(pid, x_ps_d);
      auto posit_next     = p::makeVector3(pid, xtgt_ps_d);
      auto eField         = p::makeVector3(pid, efield_ps_d);
      auto vel            = p::makeVector3(pid, vel_ps_d);
      auto parallel_dir   = o::zero_vector<3>();
      auto perp1_dir      = o::zero_vector<3>();
      auto perp2_dir      = o::zero_vector<3>();
      auto pos2D          = o::zero_vector<3>();
      pos2D[0]            = sqrt(posit_next[0]*posit_next[0] + posit_next[1]*posit_next[1]);
      pos2D[1]            = 0;
      pos2D[2]            = posit_next[2];
      Omega_h::Vector<3> flowVelocity_radial;
      Omega_h::Vector<3> flowVelocity;
      Omega_h::Vector<3> b_field;

      double dens = 0;
      double temp = 0;
      double temp_el = 0;
      auto velIn = vel;

      //TODO merge with intrp, simialr to bfield case
      if (useConstantFlowVelocity) {
        for(auto i=0; i<3; ++i)
        flowVelocity_radial[i] = constFlowVelocity[i];

        o::Real theta = atan2(posit_next[1], posit_next[0]);

        flowVelocity[0] = cos(theta)*flowVelocity_radial[0] - sin(theta)*flowVelocity_radial[1];
        flowVelocity[1] = sin(theta)*flowVelocity_radial[0] + cos(theta)*flowVelocity_radial[1];
        flowVelocity[2] = flowVelocity_radial[2];
      } else if (use2dInputFields) {
        p::interp2dVector_wgrid(flowVel_2d, velGridX, velGridZ, posit_next, flowVelocity, cylSymm);
      }

      if (use2dInputFields || useConstantBField) {
        p::interp2dVector_wgrid(BField_2d, bGridX, bGridZ, posit_next, b_field, cylSymm);
      }
      else if (use3dField) {
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
        p::interpolate3dFieldTet(mesh2verts, BField, el, bcc, b_field);
      }

      if (use2dInputFields){
        dens = p::interpolate2d_wgrid(densEl_d, densGridX, densGridZ, pos2D,
            true, 1, 0, false);
        temp = p::interpolate2d_wgrid(temIon_d, ionTempGridX, ionTempGridZ, pos2D,
            true, 1, 0, false);
        temp_el= p::interpolate2d_wgrid(temEl_d, elTempGridX, elTempGridZ, pos2D,
            true, 1, 0, false);
      }
      else if (use3dField){
        auto bcc = o::zero_vector<4>();
        p::findBCCoordsInTet(coords, mesh2verts, posit_next, el, bcc);
        dens    = p::interpolateTetVtx(mesh2verts, densVtx, el, bcc, 1);
        temp    = p::interpolateTetVtx(mesh2verts, tIonVtx, el, bcc, 1);
        temp_el = p::interpolateTetVtx(mesh2verts, tElVtx, el, bcc, 1);
      }
      if(debug > 2)
        printf("ptcl %d tstep %d density %g ion_temp %.15f el_temp3D %.15f\n",
          ptcl, tstep, dens, temp, temp_el);
      double nu_friction   = 0;
      double nu_deflection = 0;
      double nu_parallel   = 0;
      double nu_energy     = 0;

      auto relvel = vel - flowVelocity;
      if(debug >2)
        printf("Collision: relVel %g %g %g vel %g %g %g flowVel %g %g %g\n", relvel[0],
          relvel[1], relvel[2], vel[0], vel[1],vel[2], flowVelocity[0], flowVelocity[1],
          flowVelocity[2]);
      getSlowDownFrequencies(relvel, nu_friction, nu_deflection, nu_parallel, nu_energy, temp,temp_el,
              dens,charge, background_Z, background_amu, amu, ptcl, tstep, debug);

      getSlowDownDirections2(relvel, parallel_dir, perp1_dir, perp2_dir, ptcl, tstep, debug);

      auto velNorm  = Omega_h::norm(relvel);

      double n1  = 0;
      double n2  = 0;
      double xsi = 0;
      if(useGitrRnd) {
        n1  = testGitrPtclStepData[ptcl_global*testGNT*testGDof + tstep*testGDof + collisionIndex1];
        n2  = testGitrPtclStepData[ptcl_global*testGNT*testGDof + tstep*testGDof + collisionIndex2];
        xsi = testGitrPtclStepData[ptcl_global*testGNT*testGDof + tstep*testGDof + collisionIndex3];
      } else if (useCudaRnd) {
        auto localState = cuStates[ptcl_global];
        n1 = curand_normal(&localState);
        n2 = curand_normal(&localState);
        xsi = curand_uniform(&localState);
        cuStates[ptcl_global] = localState;
      } else {
        auto rnd = rpool.get_state();
        n1 = rnd.normal();
        n2 = rnd.normal();
        xsi = rnd.drand();
        rpool.free_state(rnd);
      }
      if(nu_parallel <=0)
        nu_parallel = 0;
      auto coeff_par = n1 * sqrt(2.0*nu_parallel * dt);
      auto cosXsi = cos(2.0 * pi * xsi) - 0.0028;
      if(cosXsi > 1)
        cosXsi = 1;
      auto sinXsi = sin(2.0 * pi * xsi);
      if(nu_deflection <=0)
        nu_deflection = 0;
      
      double coeff_perp1 = cosXsi * sqrt(nu_deflection * dt * 0.5);
      double coeff_perp2 = sinXsi * sqrt(nu_deflection * dt * 0.5);

      double nuEdt = nu_energy*dt;
      if (nuEdt < -1.0)
        nuEdt = -1.0;

      auto vNormEt = velNorm*(1-0.5*nuEdt);
      auto vPar = (1+coeff_par)*parallel_dir;
      auto vPerp0 = coeff_perp1 * perp1_dir + coeff_perp2 * perp2_dir;
      auto vPerp = abs(n2) * vPerp0;
      auto vPar2 = velNorm * dt * nu_friction* parallel_dir;
      vel = vNormEt*(vPar + vPerp) - vPar2 + flowVelocity;

      if(debug > 2){
        printf("RndNums-coulomb ptcl %d tstep %d n1 %g n2 %g xsi %g nu_parallel %g\n",
         ptcl, tstep, n1, n2, xsi, nu_parallel);

        printf("Collision ptcl %d timestep %d cosXsi %g sinXsi %g vPartNorm %g charge %d\n",
          ptcl, tstep, cosXsi, sinXsi, velNorm, charge);
        printf("Collision ptcl %d timestep %d nuEdt %g coeff_par %g cf_perp1,2 %g %g \n",
          ptcl, tstep, nuEdt, coeff_par, coeff_perp1, coeff_perp2);

        printf("Collision: ptcl %d timestep %d par-dir %g %g %g \n", ptcl, tstep,
          parallel_dir[0], parallel_dir[1], parallel_dir[2]);
        printf("Collision: ptcl %d timestep %d perpdir1 %g %g %g perpdir2 %g %g %g \n",
          ptcl, tstep, perp1_dir[0], perp1_dir[1], perp1_dir[2],
          perp2_dir[0], perp2_dir[1], perp2_dir[2]);
      }

      if(debug > 1)
        printf("Collision ptcl %d timestep %d pos %.15f %.15f %.15f vel_in %.15f %.15f %.15f"
          " => %.15f %.15f %.15f\n", ptcl, tstep, posit_next[0],
          posit_next[1],posit_next[2], velIn[0], velIn[1], velIn[2], vel[0],vel[1],vel[2]);

      vel_ps_d(pid,0) = vel[0];
      vel_ps_d(pid,1) = vel[1];
      vel_ps_d(pid,2) = vel[2];
    }
  };
  p::parallel_for(ptcls, lambda, "coulomb_collision_kernel");
}

#endif
