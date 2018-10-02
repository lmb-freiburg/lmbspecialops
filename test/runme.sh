export TF_CPP_MIN_LOG_LEVEL=2
log_file="results.log"
> $log_file
tail -f results.log & 
for f in *.py; do
	echo "$f is starting"
	python3 $f >> $log_file 2>&1
	echo "$f is done"
	sleep 1
done 
