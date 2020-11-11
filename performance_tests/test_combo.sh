#!/bin/bash
# Bash script to run a series of tests of ps_pseudopush

for e in 38000
do
  for distribution in 3
  do
    for struct in 0 
    do
      for team_size in 16 32
      do
        for slicing in 64 256 512 1024 2048
        do
          ./ps_combo $e $((e * 1000)) $distribution -n $struct -s $team_size -v $slicing -p 10 &> out.txt
          python3 combo_format.py 1 1 
        done
      done
    done
  done
done
