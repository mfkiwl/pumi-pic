#! /bin/bash -f

# input file is positions.m

[[ $# -lt 1 ]] && echo "$0 <file> <result =1 | printmatch = 2| printall=3> " && exit

file=$1
option=1
[[ $# -eq 2 ]] && option=$2

result=0
[[ $option -eq 1 ]] && result=1

printmatch=0
[[ $option -eq 2 ]] && printmatch=1

if [ $option -eq 3 ]; then \
  awk '{print substr($2, 0, length($2) - 3), " pos "  $5, $6, $7;}' $file;
  exit;
fi

[[ $option -eq 1 ]] && echo "#Bead  Counts z_min  z_max"

for n in $( seq 0 13); do
awk -v n=$n -v result=$result -v printmatch=$printmatch '
BEGIN {
 h1=0;
 h2=0.01275;
 h=0.01;
 c=0.0446;
 r=0.005;
 sum=0;

 if(n==1) { h1=h2; h2+=h;}
 else if(n>1) {h1=h2+(n-1)*h; h2+=n*h;}
} 
{
  if(sqrt(($5-c)*($5-c)+($6*$6)) <= r && $7>h1 && $7<=h2) {
    sum+=1;
    #if(n==13) 
    if(printmatch)
      print substr($2, 0, length($2)-3),  $5, $6, $7, " bead " n, sqrt(($5-c)*($5-c)+($6*$6)); 
  }
}
 END {
  if(result)
    print n "  " sum "\t\t" h1 "\t" h2 ; 
 }' $file
 
 done


