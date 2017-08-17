#!/usr/bin/env bash

mkdir results
cp ../models/models0/* .
./bin/cli test ../data2/test0/
mv results results0
mv results.txt results0.txt

mkdir results
cp ../models/models1/* .
./bin/cli test ../data2/test1/
mv results results1
mv results.txt results1.txt

mkdir results
cp ../models/models2/* .
./bin/cli test ../data2/test2/
mv results results2
mv results.txt results2.txt

mkdir results
cp ../models/models3/* .
./bin/cli test ../data2/test3/
mv results results3
mv results.txt results3.txt

mkdir results
cp ../models/models4/* .
./bin/cli test ../data2/test4/
mv results results4
mv results.txt results4.txt
