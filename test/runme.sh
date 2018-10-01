for f in *.py; do
	echo "$f is starting"
	python $f
	echo "$f is done"
	sleep 1
done 
