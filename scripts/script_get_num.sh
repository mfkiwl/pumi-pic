#!/bin/bash

# extract number @ $2 of data $3 from $1
# $1 filename
# $2 sequential number=index starting 0
# $3 pattern
[[ $# -eq 0 ]] && echo argv[0] " <dat-file> <seq-num-start0> <pattern> " && exit
file=$1
pos=$2
patt="$3"
end=";"
# NF-1 since last comma in each line
awk -F '[,;]' -v pos=$pos \
  -v beg="$patt" -v end="$end" \
  'BEGIN { 
    pre=0;num=0;ind=pos; 
  }

 $1 ~ beg {
   split($1, arr,"=") ; 
   key=arr[1]; 
   gsub (" ", "", key);
   if(key==beg) pat=1; 
 }

 pat{
   pre=num;
   num+=NF-1;
   if(pos<=num-1) {
     if(ind>=pre) 
       ind=pos-pre ;
     print $(ind+1) " :" pre "-" num-1 ":" $0;
     exit; 
   }
   if($0 ~ end){ exit;}
 }' $file

