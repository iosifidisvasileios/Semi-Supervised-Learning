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


object JustPredict {

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Self Learning MNB")//.setMaster("yarn-client")
    val argLen = args.length

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val data = sc.textFile(args(0))

    val prediction = args(argLen - 1).toDouble

    val unlabeled = sc.textFile(args(argLen - 4))

    val htf = new HashingTF(1500000)

    var trainingSet = data.map { line =>
      val parts = line.split(',')
      var par = 0.0

      if(parts(3).equals("positive")){
        par = 1.0
      }else if(parts(3).equals("negative")){
        par = 0.0
      } else if (parts(3).equals("neutral")){
        try {
          par = parts(parts.length - 1).replace(")", "").toDouble
        }catch {
          case ex:Throwable=> par = -1
        }
      }
      LabeledPoint(par, htf.transform(parts(5).split(' ')))
    }.cache()

    for (cnt <- 1 until argLen - 4){
      println("fuse ground and labeled " + cnt)
      trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>

        val parts = line.split(',')
        var par = 0
        if (parts(3).equals("positive")) {
          par = 1
        } else if (parts(3).equals("negative")) {
          par = 0
        } else if (parts(3).equals("neutral")){
          try {
            par = parts(parts.length - 1).replace(")", "").toInt
          }catch {
             case ex:Throwable=> par = -1
            }
        }
        LabeledPoint(par.toDouble, htf.transform(parts(5).split(' ')))
      }).cache()
    }

    trainingSet = trainingSet.filter(_.label != -1)

    val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")

    val testing = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(5).split(' ')
      (line, htf.transform(text))
    }.cache()

    println("TRAINING SET Positive     :------------------------------------------------: "  + trainingSet.filter(_.label == 1).count())
    println("TRAINING SET Negative     :------------------------------------------------: "  + trainingSet.filter(_.label == 0).count())
//    println("TESTING SET SIZE      :------------------------------------------------: "  + testing.count())

    val lowProbabilitiesSet = testing.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) < prediction && model.predictProbabilities(item._2)(1) < prediction) {
        List(item._1)
      } else {
        None
      }
    }.cache()

    println("LOW  PROBABILITIES SET SIZE     :------------------------------------------------: " + lowProbabilitiesSet.count())

    val highProbabilitiesSet = testing.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) >= prediction || model.predictProbabilities(item._2)(1) >= prediction) {
        List(item._1 +","+ model.predict(item._2).toInt)
      } else {
        None
      }
    }
    println("HIGH PROBABILITIES SET SIZE     :------------------------------------------------: "  + highProbabilitiesSet.count())

    highProbabilitiesSet.saveAsTextFile(args(argLen - 3))
    lowProbabilitiesSet.saveAsTextFile(args(argLen - 2))
  }

}
