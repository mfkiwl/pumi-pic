#!/bin/bash

dd=~/GITRmdev

./GITRm --kokkos-threads=1  $dd/pisces_v3.osh  \
 owners_file_ignored $dd/particleSourceHighFlux.nc   \
 $dd/profilesHighFlux.nc $dd/ADAS_Rates_W.nc \
 $dd/ftridynSelf.nc $dd/profiles.nc  $1 $2 $3 $4
