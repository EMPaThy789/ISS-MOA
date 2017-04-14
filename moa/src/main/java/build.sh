javac -cp moa.jar moa\core\Utils.java
jar -uf  moa.jar  moa\core\Utils.class 
rm moa\core\Utils.class 

javac -cp  moa.jar moa\classifiers\lazy\neighboursearch\EuclideanDistance.java
jar -uf  moa.jar moa\classifiers\lazy\neighboursearch\EuclideanDistance.class
rm moa\classifiers\lazy\neighboursearch\EuclideanDistance.class

javac -cp  moa.jar moa\classifiers\lazy\neighboursearch\CumulativeLinearNNSearch.java
jar -uf  moa.jar moa\classifiers\lazy\neighboursearch\CumulativeLinearNNSearch.class
rm moa\classifiers\lazy\neighboursearch\CumulativeLinearNNSearch.class

javac -cp  moa.jar moa\classifiers\lazy\neighboursearch\OptimisedCumulativeLinearNNSearch.java
jar -uf  moa.jar moa\classifiers\lazy\neighboursearch\OptimisedCumulativeLinearNNSearch.class
rm moa\classifiers\lazy\neighboursearch\OptimisedCumulativeLinearNNSearch.class

javac -cp  moa.jar moa\classifiers\lazy\rankingfunctions\*.java
jar -uf  moa.jar moa\classifiers\lazy\rankingfunctions\*.class
rm moa\classifiers\lazy\rankingfunctions\*.class

javac -cp  moa.jar moa\classifiers\lazy\kNNISS.java
jar -uf  moa.jar moa\classifiers\lazy\kNNISS.class
rm moa\classifiers\lazy\kNNISS.class

javac -cp moa.jar moa\streams\generators\*.java
jar -uf  moa.jar moa\streams\generators\*.class
rm moa.jar moa\streams\generators\*.class

