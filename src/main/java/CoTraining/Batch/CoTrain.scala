package CoTraining.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object CoTrain {



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
    val unlabeled = sc.textFile(args(2)).cache()
    val confidenceLevel = args(3).toDouble

    val testingSetL0 = evaluationData.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf_F0.transform(parts(7).split(' ')))
    }.cache()

    val testingSetL1 = evaluationData.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf_F1.transform(parts(9).split(' ')))
    }.cache()

    var trainingSet_L0 = data.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf_F0.transform(parts(7).split(' ')))
    }.cache()

    var trainingSet_L1 = data.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf_F1.transform(parts(9).split(' ')))
    }.cache()


    var unlabeled_F0 = unlabeled.map { line =>
      val parts = line.split(',')
      (line, htf_F0.transform(parts(7).split(' ')), htf_F1.transform(parts(9).split(' ') ))
    }

    var unlabeled_F1 = unlabeled.map { line =>
      val parts = line.split(',')
      (line, htf_F0.transform(parts(7).split(' ')), htf_F1.transform(parts(9).split(' ') ))
    }
//    unlabeled.unpersist()
    var model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
    var model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

    for(step <- 0 to 5) {
      //train classifiers
      if (step != 0) {
        model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
        model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

      var predictionAndLabel = testingSetL0.map(p => (model_F0.predict(p.features), p.label))
      var accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSetL0.count()
      println("~~~~~~~~~~~~~~~~~~~~~~~~~this is step : "+ (step) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

      println("Accuracy for Model_0 (preprocessed text): ~~> " + accuracy)

      predictionAndLabel = testingSetL1.map(p => (model_F1.predict(p.features), p.label))
      accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSetL1.count()
      println("Accuracy for Model_1 (special features ): ~~> " + accuracy)

      println("size of training set for model_0 :  ~~~~~~~~> " + trainingSet_L0.count())
      println("size of training set for model_1 :  ~~~~~~~~> " + trainingSet_L1.count())

      println("size of unlabeled set for model_0 :  ~~~~~~~~> " + unlabeled_F0.count())
      println("size of unlabeled set for model_1 :  ~~~~~~~~> " + unlabeled_F1.count())
      println("*********************************************************")

      }

      // put htf (8) to vector so dont compute them when added to trainingsetL1
      var resultsHigh_L0 = unlabeled_F0.filter { item =>
        val output = model_F0.predictProbabilities(item._2)
        output(0) >= confidenceLevel || output(1) >= confidenceLevel
      }.map{ line=>
        (model_F0.predict(line._2), line._3)
      }.cache()

      unlabeled_F0 = unlabeled_F0.filter { item =>
        val output = model_F0.predictProbabilities(item._2)
        output(0) < confidenceLevel && output(1) < confidenceLevel
      }.cache()

      // put htf (7) to vector so dont compute them when added to trainingsetL0
      var resultsHigh_L1 = unlabeled_F1.filter { item =>
        val output = model_F1.predictProbabilities(item._3)
        output(0) >= confidenceLevel || output(1) >= confidenceLevel
      }.map{ line=>
        (model_F1.predict(line._3), line._2)
      }.cache()

      unlabeled_F1 = unlabeled_F1.filter { item =>
        val output = model_F1.predictProbabilities(item._3)
        output(0) < confidenceLevel && output(1) < confidenceLevel
      }.cache()

      trainingSet_L0 = trainingSet_L0.union(resultsHigh_L1.map{ line => LabeledPoint(line._1, line._2)}).cache()
      trainingSet_L1 = trainingSet_L1.union(resultsHigh_L0.map{ line => LabeledPoint(line._1, line._2)}).cache()
    }
  }
}

