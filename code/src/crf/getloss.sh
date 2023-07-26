a=$(grep -n 'Fri Jun 30 19:39:32 2023' ctb.out |awk 'NR==1 {gsub(/:/,"",$1); print $1}')
grep ctb.out | head -n ${a,\033[1;33m,""}
