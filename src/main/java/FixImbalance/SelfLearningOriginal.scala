package FixImbalance

/**
  * Created by iosifidis on 25.10.16.
  */
/**
  * Created by iosifidis on 10.10.16.
  */

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object SelfLearningOriginal {

  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName("Self Learning Imbalance Fix")
    val argLen = args.length

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val data = sc.textFile(args(0))

    val prediction = args(argLen - 1).toDouble
    val unlabeled = sc.textFile(args(argLen - 4))
    val htf = new HashingTF(1500000)

    var trainingSet = data.map { line =>
      val parts = line.split(',')
      var par = 0

      if(parts(3).equals("positive")){
        par = 1
      }else if(parts(3).equals("negative")){
        par = 0
      }else {
        par = parts(parts.length - 1).toInt
       }
      (par.toDouble, line)
    }.cache()

    for (cnt <- 1 until argLen - 4){
      println("fuse ground and labeled " + cnt)
      trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
        val parts = line.split(',')
        val par = parts(parts.length - 1).toDouble
        (par, line)

      }).cache()
    }

    val train = trainingSet.map{ line=>
      LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
    }

    println("final positives = " + train.filter(_.label == 1).count() )
    println("final negatives = " + train.filter(_.label == 0).count() )

    val model = NaiveBayes.train(train , lambda = 1.0, modelType = "multinomial")

    val forPrediction = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(5).split(' ')
      (line, htf.transform(text))
    }.cache()

    val unlabeldCountCorpus = forPrediction.count

    println("TRAINING SET SIZE     :------------------------------------------------: "  + trainingSet.count())
    println("UnLabel  SET SIZE     :------------------------------------------------: "  + unlabeldCountCorpus)

    val lowProbabilitiesSet = forPrediction.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) < prediction && model.predictProbabilities(item._2)(1) < prediction) {
        List(item._1)
      } else {
        None
      }
    }.cache()


    val highProbabilitiesSet = forPrediction.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) >= prediction || model.predictProbabilities(item._2)(1) >= prediction) {
        List(item._1 +","+ model.predict(item._2).toInt)
      } else {
        None
      }
    }

    var remainingUnlabeled = lowProbabilitiesSet.count()
    println("Unlabeled SET SIZE     :------------------------------------------------: " + remainingUnlabeled )
    println("Labeled SET SIZE     :------------------------------------------------: "  + (unlabeldCountCorpus - remainingUnlabeled))

    highProbabilitiesSet.saveAsTextFile(args(argLen - 3))
    lowProbabilitiesSet.saveAsTextFile(args(argLen - 2))
  }

}
