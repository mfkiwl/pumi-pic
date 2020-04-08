#ifndef GITRM_SPECTROSCOPY_HPP
#define GITRM_SPECTROSCOPY_HPP

#include "GitrmParticles.hpp" 

class GitrmSpectroscopy {
public:
  GitrmSpectroscopy();

  void initSpectroscopy();
  void writeSpectroscopyFile(const std::string& file="spec.nc");
  double netX0 = 0;
  double netXn = 0;
  double netY0 = 0;
  double netYn = 0;
  double netZ0 = 0;
  double netZn = 0;
  int nX = 0;
  int nY = 0;
  int nZ = 0;
  int nBins = 0;
  int nSpec = 0;
  
  o::Write<o::Real> netBins;
  o::Real gridX0 = 0;
  o::Real dX = 0;
  o::Real gridXn = 0;
  o::Real gridY0 = 0;
  o::Real dY = 0;
  o::Real gridYn = 0;
  o::Real gridZ0 = 0;
  o::Real dZ = 0;
  o::Real gridZn = 0;
};

inline void gitrm_spectroscopy(PS* ptcls, GitrmSpectroscopy& sp, o::LOs& elm_ids,
    bool debug = false) {
  const auto istep = iTimePlusOne - 1;
  const auto nX = sp.nX;
  const auto nY = sp.nY;
  const auto nZ = sp.nZ;
  const auto gridX0 = sp.gridX0;
  const auto gridY0 = sp.gridY0;
  const auto gridZ0 = sp.gridZ0;
  const auto dx = sp.dX;
  const auto dy = sp.dY;
  const auto dz = sp.dZ;
  const auto gridXn = sp.gridXn;
  const auto gridYn = sp.gridYn;
  const auto gridZn = sp.gridZn;
  const auto nBins = sp.nBins;
  auto& bins = sp.netBins;
  const auto pid_ps = ptcls->get<PTCL_ID>();
  const auto next_pos = ptcls->get<PTCL_NEXT_POS>();
  const auto charge_ps = ptcls->get<PTCL_CHARGE>();
  const auto weight_ps = ptcls->get<PTCL_WEIGHT>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0 && elm_ids[pid] >=0) {
      auto ptcl = pid_ps(pid);
      auto pos = p::makeVector3(pid, next_pos);
      auto x = pos[0];
      auto y = pos[1];
      auto z = pos[2];
      if(z > gridZ0 && z < gridZn && x > gridX0 && x < gridXn &&
	y > gridY0 && y < gridYn) {
	int indX = floor((x - gridX0)/dx);
	int indZ = floor((z - gridZ0)/dz);
	int indY = floor((y - gridY0)/dy);
	indX = (indX < 0 || indX >= nX) ? 0: indX; 
	indY = (indY < 0 || indY >= nY) ? 0: indY; 
	indZ = (indZ < 0 || indZ >= nZ) ? 0: indZ; 
	auto charge = charge_ps(pid);
	auto weight = weight_ps(pid);
	auto index = nBins*nX*nY*nZ + indZ*nX*nY +indY*nX+ indX;
	auto old = Kokkos::atomic_fetch_add(&(bins[index]), weight);
        if(debug)
          printf( "spec ptcl %d step %d bins %g index %d weight %g charge %d "
           "nBins %d pos %g %g %g ind %d \n", ptcl, istep, old+weight, index, weight, 
            charge, nBins, x, y, z, charge*nX*nY*nZ + indZ*nX*nY + indY*nX+ indX);
        if(charge < nBins) {
	  auto ind = charge*nX*nY*nZ + indZ*nX*nY + indY*nX+ indX;
	  OMEGA_H_CHECK(ind >=0);
	  auto old = Kokkos::atomic_fetch_add(&bins[ind], weight);
          if(debug)
            printf("spec: ptcl %d step %d bins %g \n", ptcl, istep, old+weight);
	}
      } // z
    } //mask
  };
  p::parallel_for(ptcls, lambda, "spectroscopy");
}

#endif

