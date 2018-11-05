package SelfLearning.Batch

import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

object CalculatePosNegBatch {

  //   def conf = new SparkConf().setAppName("App_Name").setMaster("spark://localhost:6066").set("spark.ui.port","8080");
  //   var sc: SparkContext = _


  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName(getClass().getName())

    val sc = new SparkContext(conf)
    println("Hello, world!")
    sc.setLogLevel("ERROR")

    val data = sc.textFile(args(0))
    val pos = data.filter{ line =>
      val index = line.split(",")
      val label = index(index.length - 1).toDouble
      label == 1.0
    }.count()
    val neg = data.filter{ line =>
      val index = line.split(",")
      val label = index(index.length - 1).toDouble
      label == 0.0
    }.count()

    println("File is -> " + args(0))
    println("positive number: " + pos)
    println("negative number: " + neg)

  }
}