#!/usr/bin/env bash

mkdir results
cp ../models/models0/* .
./bin/cli test ../data2/test0/ > result0.txt

mkdir results
cp ../models/models1/* .
./bin/cli test ../data2/test1/ > result1.txt

mkdir results
cp ../models/models2/* .
./bin/cli test ../data2/test2/ > result2.txt

mkdir results
cp ../models/models3/* .
./bin/cli test ../data2/test3/ > result3.txt

mkdir results
cp ../models/models4/* .
./bin/cli test ../data2/test4/ > result4.txt
