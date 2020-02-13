#!/bin/bash -e

classlist=$1
from=$2
to=$3
count=$4

if [ ! -d $from ]; then
    echo "$from does not exist"
    exit 1
fi

function copy()
{
    # define function to avoid limitation of xargs replacement up to 255 bytes
    class=$1
    from=$2
    to=$3
    count=$4
    find $from -maxdepth 3 -name "${class}@*" | shuf | head -n $count | xargs -I{} cp {} $to/$class/
}

export -f copy

cat $classlist | xargs -I{} bash -c "mkdir -p $to/{}"
cat $classlist | xargs -n1 -I{} -P8 bash -c "copy {} $from $to $count"
#head -n $count | xargs -I@ -P8 echo @ $to/{}"
