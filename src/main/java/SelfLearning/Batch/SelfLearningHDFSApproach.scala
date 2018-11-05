package SelfLearning.Batch

/**
  * Created by iosifidis on 25.10.16.
  */
/**
  * Created by iosifidis on 10.10.16.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint


object SelfLearningHDFSApproach {

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Self Learning MNB File Approach")//.setMaster("yarn-client")
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
      LabeledPoint(par.toDouble, htf.transform(parts(5).split(' ')))
    }.cache()

    for (cnt <- 1 until argLen - 4){
      println("fuse ground and labeled " + cnt)
      trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
        val parts = line.split(',')
        val par = parts(parts.length - 1).toDouble
        LabeledPoint(par, htf.transform(parts(5).split(' ')))
      }).cache()
    }

    val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")
    val testing = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(5).split(' ')
      (line, htf.transform(text))
    }.cache()

    println("TRAINING SET SIZE     :------------------------------------------------: "  + trainingSet.count())
    println("TESTING SET SIZE      :------------------------------------------------: "  + testing.count())

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
