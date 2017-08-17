import os
import shutil

probSVM = 0

inPath = "/home/tom/Downloads/cabinet-parser-master"
binPath = "/build/bin/"
dataPath = "/data/depth/118/crossValidation"
scriptsPath = inPath+"/scripts"
resultsPath = scriptsPath+"/results"

#print("copying files... ")

#shutil.copy(inPath+"/build/edge_model.bin", scriptsPath+"/edge_model.bin")
#shutil.copy(inPath+"/build/appearance_01.csv", scriptsPath+"/appearance_01.csv")
#shutil.copy(inPath+"/build/sizes.csv", scriptsPath+"/sizes.csv")
#shutil.copy(inPath+"/build/codebook.dat", scriptsPath+"/codebook.dat")

#if probSVM:
#    shutil.copy(inPath+"/build/class0vsAllSVM.xml", scriptsPath+"/class0vsAllSVM.xml")
#    shutil.copy(inPath+"/build/class1vsAllSVM.xml", scriptsPath+"/class1vsAllSVM.xml")
#    shutil.copy(inPath+"/build/class2vsAllSVM.xml", scriptsPath+"/class2vsAllSVM.xml")
#    shutil.copy(inPath+"/build/trainParameters.yml", scriptsPath+"/trainParameters.yml")
#else:
#    shutil.copy(inPath+"/build/depthPrior.yml", scriptsPath+"/depthPrior.yml")
#    shutil.copy(inPath+"/build/shapePrior.yml", scriptsPath+"/shapePrior.yml")
#    shutil.copy(inPath+"/build/maxDepth.txt", scriptsPath+"/maxDepth.txt")#
    






if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

from distutils.dir_util import copy_tree
fromDirectory = resultsPath

open(scriptsPath+"/results.txt", 'w')



set = "set1"
print("cross Validation: "+set)
if not os.path.exists(scriptsPath+"/"+set):
    os.makedirs(scriptsPath+"/"+set)

print("training")
os.system(inPath+binPath+"cli train "+inPath+dataPath+"/"+set+"/train/")

run = "run1"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run2"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run3"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

print("copying "+set+" files")
shutil.copy(scriptsPath+"/edge_model.bin", scriptsPath+"/"+set+"/edge_model.bin")
shutil.copy(scriptsPath+"/appearance_01.csv", scriptsPath+"/"+set+"/appearance_01.csv")
shutil.copy(scriptsPath+"/sizes.csv", scriptsPath+"/"+set+"/sizes.csv")
shutil.copy(scriptsPath+"/codebook.dat", scriptsPath+"/"+set+"/codebook.dat")

if probSVM:
    shutil.copy(scriptsPath+"/class0vsAllSVM.xml", scriptsPath+"/"+set+"/class0vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class1vsAllSVM.xml", scriptsPath+"/"+set+"/class1vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class2vsAllSVM.xml", scriptsPath+"/"+set+"/class2vsAllSVM.xml")
    shutil.copy(scriptsPath+"/trainParameters.yml", scriptsPath+"/"+set+"/trainParameters.yml")
else:
    shutil.copy(scriptsPath+"/depthPrior.yml", scriptsPath+"/"+set+"/depthPrior.yml")
    shutil.copy(scriptsPath+"/shapePrior.yml", scriptsPath+"/"+set+"/shapePrior.yml")
    shutil.copy(scriptsPath+"/maxDepth.txt", scriptsPath+"/"+set+"/maxDepth.txt")





set = "set2"
print("cross Validation: "+set)
if not os.path.exists(scriptsPath+"/"+set):
    os.makedirs(scriptsPath+"/"+set)

print("training")
os.system(inPath+binPath+"cli train "+inPath+dataPath+"/"+set+"/train/")

run = "run1"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run2"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run3"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

print("copying "+set+" files")
shutil.copy(scriptsPath+"/edge_model.bin", scriptsPath+"/"+set+"/edge_model.bin")
shutil.copy(scriptsPath+"/appearance_01.csv", scriptsPath+"/"+set+"/appearance_01.csv")
shutil.copy(scriptsPath+"/sizes.csv", scriptsPath+"/"+set+"/sizes.csv")
shutil.copy(scriptsPath+"/codebook.dat", scriptsPath+"/"+set+"/codebook.dat")

if probSVM:
    shutil.copy(scriptsPath+"/class0vsAllSVM.xml", scriptsPath+"/"+set+"/class0vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class1vsAllSVM.xml", scriptsPath+"/"+set+"/class1vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class2vsAllSVM.xml", scriptsPath+"/"+set+"/class2vsAllSVM.xml")
    shutil.copy(scriptsPath+"/trainParameters.yml", scriptsPath+"/"+set+"/trainParameters.yml")
else:
    shutil.copy(scriptsPath+"/depthPrior.yml", scriptsPath+"/"+set+"/depthPrior.yml")
    shutil.copy(scriptsPath+"/shapePrior.yml", scriptsPath+"/"+set+"/shapePrior.yml")
    shutil.copy(scriptsPath+"/maxDepth.txt", scriptsPath+"/"+set+"/maxDepth.txt")



set = "set3"
print("cross Validation: "+set)

if not os.path.exists(scriptsPath+"/"+set):
    os.makedirs(scriptsPath+"/"+set)

print("training")
os.system(inPath+binPath+"cli train "+inPath+dataPath+"/"+set+"/train/")

run = "run1"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run2"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

run = "run3"
print("testing "+run)
os.system(inPath+binPath+"cli test "+inPath+dataPath+"/"+set+"/test/")
toDirectory   = scriptsPath+"/"+set+"/"+run+"/results"
if not os.path.exists(toDirectory):
    os.makedirs(toDirectory)
copy_tree(fromDirectory, toDirectory)
shutil.copy(scriptsPath+"/results.txt", scriptsPath+"/"+set+"/"+run+"/results.txt")

print("copying "+set+" files")
shutil.copy(scriptsPath+"/edge_model.bin", scriptsPath+"/"+set+"/edge_model.bin")
shutil.copy(scriptsPath+"/appearance_01.csv", scriptsPath+"/"+set+"/appearance_01.csv")
shutil.copy(scriptsPath+"/sizes.csv", scriptsPath+"/"+set+"/sizes.csv")
shutil.copy(scriptsPath+"/codebook.dat", scriptsPath+"/"+set+"/codebook.dat")

if probSVM:
    shutil.copy(scriptsPath+"/class0vsAllSVM.xml", scriptsPath+"/"+set+"/class0vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class1vsAllSVM.xml", scriptsPath+"/"+set+"/class1vsAllSVM.xml")
    shutil.copy(scriptsPath+"/class2vsAllSVM.xml", scriptsPath+"/"+set+"/class2vsAllSVM.xml")
    shutil.copy(scriptsPath+"/trainParameters.yml", scriptsPath+"/"+set+"/trainParameters.yml")
else:
    shutil.copy(scriptsPath+"/depthPrior.yml", scriptsPath+"/"+set+"/depthPrior.yml")
    shutil.copy(scriptsPath+"/shapePrior.yml", scriptsPath+"/"+set+"/shapePrior.yml")
    shutil.copy(scriptsPath+"/maxDepth.txt", scriptsPath+"/"+set+"/maxDepth.txt")

print("Cross Validation Completed")
