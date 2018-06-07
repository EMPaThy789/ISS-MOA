javac -encoding utf8 -cp moa.jar moa/core/Utils.java
jar -uf moa.jar  moa/core/Utils.class 
#rm moa/core/Utils.class 

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.java
jar -uf moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/iss/ranking/*.java
jar -uf moa.jar moa/classifiers/iss/ranking/*.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/iss/knn/*.java
jar -uf moa.jar moa/classifiers/iss/knn/*.class



javac -encoding utf8 -cp  moa.jar moa/classifiers/iss/*.java
jar -uf  moa.jar moa/classifiers/iss/*.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/kNNISSTemp.java
jar -uf moa.jar moa/classifiers/lazy/kNNISSTemp.class


javac -encoding utf8 -cp moa.jar moa/streams/generators/ConditionalGenerator.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG2.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG3.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/SEAFD.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/AssetNegotiationGenerator.java
jar -uf moa.jar moa/streams/generators/*.class
#rm moa/streams/generators/*.class

#javac -encoding utf8 -cp  moa.jar moa/ISSExperiments.java
#jar -uf  moa.jar moa/ISSExperiments.class
#rm moa/ISSExperiments.class

#javac -encoding utf8 -cp  moa.jar moa/Testing.java
#jar -uf  moa.jar moa/Testing.class
#rm moa/Testing.class

read -rsp $'Press enter to continue...\n'