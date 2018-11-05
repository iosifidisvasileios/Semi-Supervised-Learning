package SelfLearning.Batch

/**
  * Created by iosifidis on 25.10.16.
  */
/**
  * Created by iosifidis on 10.10.16.
  */

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}


object SelfLearningHDFSApproachNoRetweets {

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Self Learning MNB File Approach") //.setMaster("yarn-client")

    val sc = new SparkContext(conf)
    val htf = new HashingTF(1500000)

    sc.setLogLevel("ERROR")
    val data = sc.textFile(args(0))
    val noRetweets = data.filter{line =>
    val flag = line.split(",")(4).toInt
      flag == 0
    }.map(item => item).cache()

    println("this is the no retweet corpus: " + noRetweets.count())
//
    var trainingSet = noRetweets.map { line =>
      val parts = line.split(',')
      var par = 0

      if (parts(5).equals("positive")) {
        par = 1
      } else if (parts(5).equals("negative")) {
        par = 0
      } else {
        par = parts(8).toInt
      }
      LabeledPoint(par.toDouble, htf.transform(parts(7).split(' ')))
    }.cache()
    println("this is the no retweet corpus: " + trainingSet.count())



  }

}
