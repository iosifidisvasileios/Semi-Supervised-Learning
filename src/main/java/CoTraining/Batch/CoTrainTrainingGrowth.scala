package CoTraining.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object CoTrainTrainingGrowth {



  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Using CoTrain Technique")

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")

    val htf_F0 = new HashingTF(1500000)
    val htf_F1 = new HashingTF(3500000)

    val data = sc.textFile(args(0))
    val evaluationData = sc.textFile(args(1))
    val unlabeled = sc.textFile(args(2))

    //    println("unlabeled size " + unlabeled.count())

    val testingSet = evaluationData.map { line =>
      val parts = line.split('\t')
      val preprocessed = parts(3).split(" ")

      var bigrams = ""
      var index = 0

      while (index < preprocessed.length - 1) {
        bigrams += preprocessed(index) + "_" + preprocessed(index + 1) + " "
        index += 1
      }

      bigrams = bigrams.substring(0, bigrams.length - 1)
      (parts(1).toDouble, htf_F0.transform(parts(3).split(' ')),htf_F1.transform(bigrams.split(' ')))
    }.cache()

    var trainingSet_L0 = data.map { line =>
      val parts = line.split('\t')
      LabeledPoint(parts(1).toDouble, htf_F0.transform(parts(3).split(' ')))
    }.cache()

    var trainingSet_L1 = data.map { line =>
      val parts = line.split('\t')
      val preprocessed = parts(3).split(" ")
      var bigrams = ""
      var index = 0

      while (index < preprocessed.length - 1) {
        bigrams += preprocessed(index) + "_" + preprocessed(index + 1) + " "
        index += 1
      }
      bigrams = bigrams.substring(0, bigrams.length - 1)
      LabeledPoint(parts(1).toDouble, htf_F1.transform(bigrams.split(' ')))
    }.cache()


    var unlabeled_F0 = unlabeled.map { line =>
      val parts = line.split('\t')

      val preprocessed = parts(3).split(" ")

      var bigrams = ""
      var index = 0

      while (index < preprocessed.length - 1) {
        bigrams += preprocessed(index) + "_" + preprocessed(index + 1) + " "
        index += 1
      }

      bigrams = bigrams.substring(0, bigrams.length - 1)

      (line, htf_F0.transform(parts(3).split(' ')), htf_F1.transform(bigrams.split(' ') ))
    }.cache()

    var unlabeled_F1 = unlabeled.map { line =>
      val parts = line.split('\t')
      val preprocessed = parts(3).split(" ")

      var bigrams = ""
      var index = 0

      while (index < preprocessed.length - 1) {
        bigrams += preprocessed(index) + "_" + preprocessed(index + 1) + " "
        index += 1
      }

      bigrams = bigrams.substring(0, bigrams.length - 1)
      (line, htf_F0.transform(parts(3).split(' ')), htf_F1.transform(bigrams.split(' ') ))
    }.cache()

    val testingSize = testingSet.count()
    //train classifiers

    var model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
    var model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

    var resultsHigh_L0 = unlabeled_F0.map{ line=>
      (model_F0.predict(line._2), line._3)
    }.cache()

    var resultsHigh_L1 = unlabeled_F1.map{ line=>
      (model_F1.predict(line._3), line._2)
    }.cache()

    trainingSet_L0 = trainingSet_L0.union(resultsHigh_L1.map{ line => LabeledPoint(line._1, line._2)}).cache()
    trainingSet_L1 = trainingSet_L1.union(resultsHigh_L0.map{ line => LabeledPoint(line._1, line._2)}).cache()

    model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
    model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

    var predictionAndLabel = testingSet.map(p => (model_F0.predict(p._2), p._1))
    var accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSize
    println(args(0))
    println("Accuracy for Model_0 (Unigrams): " + accuracy)
    println("Positives - bigrams: " + resultsHigh_L1.filter( _._1 == 4).count)
    println("Negatives - bigrams: " + resultsHigh_L1.filter( _._1 == 0).count)

    predictionAndLabel = testingSet.map(p => (model_F1.predict(p._3), p._1))
    accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSize
    println("Accuracy for Model_1 (Bigrams ): " + accuracy)
    println("Positives - bigrams: " + resultsHigh_L0.filter( _._1 == 4).count)
    println("Negatives - bigrams: " + resultsHigh_L0.filter( _._1 == 0).count)

  }
}

