javac -encoding utf8 -cp moa.jar moa/core/Utils.java
jar -uf  moa.jar  moa/core/Utils.class 
rm moa/core/Utils.class 

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.class
rm moa/classifiers/lazy/neighboursearch/EuclideanDistance.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/neighboursearch/CumulativeLinearNNSearch.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/*.class
rm moa/classifiers/lazy/neighboursearch/*.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/rankingfunctions/*.java
jar -uf  moa.jar moa/classifiers/lazy/rankingfunctions/*.class
rm moa/classifiers/lazy/rankingfunctions/*.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/kNNISS.java
jar -uf  moa.jar moa/classifiers/lazy/kNNISS.class
rm moa/classifiers/lazy/*.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/bayes/NaiveBayesISS.java
jar -uf  moa.jar moa/classifiers/bayes/NaiveBayesISS.class
rm moa/classifiers/bayes/NaiveBayesISS.class

javac -encoding utf8 -cp  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.class
rm moa/classifiers/lazy/neighboursearch/EuclideanDistance.class

javac -encoding utf8 -cp moa.jar moa/streams/generators/ConditionalGenerator.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG2.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/BG3.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/SEAFD.java
javac -encoding utf8 -cp moa.jar moa/streams/generators/AssetNegotiationGenerator.java
jar -uf  moa.jar moa/streams/generators/*.class
rm moa/streams/generators/*.class

javac -encoding utf8 -cp  moa.jar moa/ISSExperiments.java
jar -uf  moa.jar moa/ISSExperiments.class
rm moa/ISSExperiments.class

read -rsp $'Press enter to continue...\n'