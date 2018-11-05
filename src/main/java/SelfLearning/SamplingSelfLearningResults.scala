package SelfLearning

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by iosifidis on 26.10.16.
  */
object SamplingSelfLearningResults {
  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName("Split Dataset")
    val sc = new SparkContext(conf)
    val data = sc.textFile(args(0))
    val outputDir = args(1)

    sc.parallelize(data.filter{ line =>
      line.split(',')(7).toDouble == 1.0
    }.take(100)).saveAsTextFile(outputDir + "/positive")

    sc.parallelize(data.filter{ line =>
      line.split(',')(7).toDouble == 0.0
    }.take(100)).saveAsTextFile(outputDir + "/negative")



  }
}
