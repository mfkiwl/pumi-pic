#!/bin/bash

# input file should contain ordered time steps of particle(s)
[[ $# -lt 1 ]] && echo "argv[0] <infile> [<pattern1> <pattern2> <pidColumn> <xcol> <nP>]" && exit;
infile=$1
[[ ! -f $infile ]] && echo "input file invalid" && exit;

patt="ptclID" # gitr
[[ $# -gt 1 ]] && patt=$2

pat0="iter 0"  #"t 0"
[[ $# -gt 2 ]] && pat0=$3

pidcol=2
[[ $# -gt 3 ]] && pidcol=$4

xcol=6
[[ $# -gt 4 ]] && xcol=$5

np=x
[[ $# -gt 5 ]] && np=$6

nptcl=0
re='^[0-9]+$'
[[ ! $np =~ $re ]] && echo "No particle limit provided"
[[ $np =~ $re ]] && nptcl=$np

echo "Using pattern \"${patt}\" and \"${pat0}\" pid_col $pidcol numP $nptcl"

ptcls=($(awk -v pn="$pidcol" -v p1="$patt" -v p2="$pat0" '$0 ~ p1 && $0 ~ p2 { print $pn}' "$infile"))
#ptcls=($(grep -w --color=no $patt  "${infile}" | grep "iter 0" | cut -d\  -f2)) 
[[ $nptcl -lt 1 ]] && let "nptcl = ${#ptcls[@]}"  && echo -e "\t Updated nptcl = $nptcl"

name=$(basename "$infile")
fname=${name}_ptcls.vtk
[[ -f $fname ]] && mv $fname  ${fname}_old

# get non-zero number of points of particles
pts=()
for (( i=0; i<nptcl ; i++ )); do
  pid=${ptcls[$i]}
  [[ -z $pid ]] && continue
  # total entries for this particle
  # np=$(grep -w --color=no "${patt} ${pid}" "$infile" | grep iter | wc -l) 
  np=$( awk -v pid=$pid -v pn=$pidcol -v p1="$patt" '$0 ~ p1 && $pn == pid { print $pn}' "$infile"| wc -l)
  [[ $np -eq  0 ]] && continue 
  pts+=( $np )
done

#total points for all ptcls
points=0
for i in ${pts[@]}; do
  let "points += i"
done

echo "Total points $points of $nptcl particles"

echo -e "# vtk DataFile Version 2.0\nparticle paths\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS $points float" > $fname 

#collect points of particles. 
for (( i=0; i<nptcl ; i++ )); do
  pid=${ptcls[$i]}
  [[ -z $pid ]] && continue
  #echo "points of ip $i pid $pid"
  awk -v pid=$pid -v pn="$pidcol" -v p1="$patt" -v xc=$xcol \
  '$0 ~ p1 && $pn == pid { print $xc, $(xc+1), $(xc+2) }' "$infile" >> $fname
  #grep -w --color=no "${patt} ${pid}"  "$infile" |cut -d\  -f6-8 | sed -e 's/^[[:space:]]*//' >> $fname
done

#connect; # pids and pts are in same order
cells=0
for (( i=0; i<nptcl ; i++ )); do
  np=${pts[i]}
  [[ -z $np ]] && continue
  let "cells = cells + $np-1"
  #echo "$i $np $cells"
done

let "cellnums = $cells *3"
echo "CELLS $cells $cellnums" >> $fname

cumul=0
for (( i=0; i<nptcl ; i++ )); do
  np=${pts[i]}
  [[ -z $np ]] && continue
  ((upto = cumul + np - 1))
  ((lower = upto - 1))
  for n in $(seq $cumul $lower); do
    ((nplus = n + 1))
    echo "2 $n $nplus" >> $fname
  done;
  let cumul=$cumul+$np
done

echo "CELL_TYPES $cells" >> $fname

for n in $(seq 1 $cells); do
  echo 3 >> $fname
done

 

