#!/bin/bash
echo "!!!!!---single_1---" > temp.txt

for((i=1;i<=20;i++));
do 
    ./single_1 1 >> temp.txt;
done
echo "!!!!!---single_2---" >> temp.txt
for((i=1;i<=20;i++));
do 
    ./single_2 1 >> temp.txt;
done
./shared_sleep 1