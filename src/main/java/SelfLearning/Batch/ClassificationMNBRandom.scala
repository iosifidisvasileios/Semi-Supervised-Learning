package SelfLearning.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object ClassificationMNBRandom {

  //   def conf = new SparkConf().setAppName("App_Name").setMaster("spark://localhost:6066").set("spark.ui.port","8080");
  //   var sc: SparkContext = _


  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Random Holdout MNB")

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")

    val htf = new HashingTF(1500000)
    val data = sc.textFile(args(0))
    val evaluationData = sc.textFile(args(1))
    val unlabeled = sc.textFile(args(2))

    val testingSet = evaluationData.map { line =>
      val parts = line.split(',')
      var id = 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf.transform(parts(7).split(' ')))
    }.cache()

    val trainingSet = data.map { line =>
      val parts = line.split(',')
      var id= 0
      val rand = Random.nextInt(10)
      if (rand  >= 5 ){
        id = 1
      }
      LabeledPoint(id.toDouble, htf.transform(parts(7).split(' ')))
    }.cache()

    val model = NaiveBayes.train(trainingSet, lambda = 1.0,  modelType = "multinomial")

    val test_set = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(7).split(' ')
      htf.transform(text)
    }

    val labeled = test_set.map{ item =>
      LabeledPoint( model.predict(item), item)
    }.cache()
    // train based on new labeled corpus
    val newTraining = labeled.union(trainingSet).cache()
    val evaluator_model = NaiveBayes.train(newTraining, lambda = 1.0,  modelType = "multinomial")
    //evaluate it
    val predictionAndLabel = testingSet.map(p => (evaluator_model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()

    println(accuracy)
    }

}

