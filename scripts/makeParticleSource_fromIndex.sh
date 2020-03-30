#!/bin/bash

[[ $# -lt 2 ]] && echo "$0 <ptclSrc-Ncdump-file> <indexFile> [<index_start1=0/1(1)> <script_get_num.sh>]" && exit 1
src=$1
ind=$2
ind_start1=1 # gitr positions.m start with ptcl id 1
[[ $# -gt 2 ]] && ind_start1=$3

script_dir=$(dirname $0)

script="${script_dir}/script_get_num.sh"
[[ $# -gt 3 ]] && script=$4

echo "script ${script_dir} $script"

indices=( $(awk '{print $1}' $ind))
out=particleSourcesSelected
[[ -f $out ]] && mv $out ${out}_old
nP=${#indices[@]}
echo " nP = $nP ;" > $out


variables=( "x" "y" "z" "vx" "vy" "vz" )
for p in "${variables[@]}"; do \
  i=1
  patt="$p"
  echo -n " $patt = " >> $out
  echo "collecting $p"
  for n in "${indices[@]}"; do \
    [[ ! -z $ind_start1 ]] && (( n = n-1))
      val=$( "$script" "$src" $n "$patt" |sed 's/:.*//' | sed 's/.*=//') 
      # awk '{ match($1, /=/, arr); if(arr[1] != "") print arr[1]  print substr($1, 0, length($1) - 3)}')
    [[ -z $val ]] && echo "$patt empty" && exit
   "$script" "$src" $n "$patt"

    echo -n "$val" >> $out
    let "this= $i % 5"
    [[ $i -lt ${#indices[@]} ]] &&  echo -n ", " >> $out
    [[ $this -eq 0 && $i -gt 0 ]] && echo >> $out
    let i=i+1
  done
  echo  ";" >> $out
done

