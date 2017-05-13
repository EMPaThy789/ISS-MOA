javac -cp moa.jar moa/core/Utils.java
jar -uf  moa.jar  moa/core/Utils.class 
rm moa/core/Utils.class 

javac -cp  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.class
rm moa/classifiers/lazy/neighboursearch/EuclideanDistance.class

javac -cp  moa.jar moa/classifiers/lazy/neighboursearch/CumulativeLinearNNSearch.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/*.class
rm moa/classifiers/lazy/neighboursearch/*.class


javac -cp  moa.jar moa/classifiers/lazy/rankingfunctions/*.java
jar -uf  moa.jar moa/classifiers/lazy/rankingfunctions/*.class
rm moa/classifiers/lazy/rankingfunctions/*.class

javac -cp  moa.jar moa/classifiers/bayes/NaiveBayesISS.java
jar -uf  moa.jar moa/classifiers/bayes/NaiveBayesISS.class
rm moa/classifiers/bayes/NaiveBayesISS.class

javac -cp  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.java
jar -uf  moa.jar moa/classifiers/lazy/neighboursearch/EuclideanDistance.class
rm moa/classifiers/lazy/neighboursearch/EuclideanDistance.class

javac -cp moa.jar moa/streams/generators/ConditionalGenerator.java
jar -uf  moa.jar moa/streams/generators/*.class
rm moa/streams/generators/*.class


read -rsp $'Press enter to continue...\n'