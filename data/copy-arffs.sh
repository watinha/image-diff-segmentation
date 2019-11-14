for i in `find "/Users/watinha/Dropbox/artigos/xbi/mobile/201901/new/log" -name *.arff`; do
    j=`echo $i | sed "s/.*\/log\///" | sed "s/\//-/g"`;
    echo "$i -> $j";
    cp $i $j;
done
