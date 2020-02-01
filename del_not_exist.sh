#!/bin/sh

for file in $1
do
	if [ -d "$(dirname $file$2)" ] && ! ls $file$2 >/dev/null 2>&1; then
		echo $file
		rm -r $file
	fi
done

