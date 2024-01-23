for((i=0;;));
do
	res=$(nvidia-smi | \
		grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
		sed "s/^|//" | \
		awk '{print ($8" "$10)}' | \
		sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
		awk '{  print $2 - $1} ')

	if [ $res > 18000 ];then
		echo $res
		nohup python main.py >> jw_3-4.out &
		break
	
	fi
	sleep 2s
done
