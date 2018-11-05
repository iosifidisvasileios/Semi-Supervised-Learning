package SelfLearning.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object ClassificationMNBFlushtoHDFS {

  //   def conf = new SparkConf().setAppName("App_Name").setMaster("spark://localhost:6066").set("spark.ui.port","8080");
  //   var sc: SparkContext = _


  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Self Learning Flush to HDFS")

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")

    val htf = new HashingTF(1500000)
    val data = sc.textFile(args(0))
    val unlabeled = sc.textFile(args(1))


    val trainingSet = data.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(5).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf.transform(parts(7).split(' ')))
    }.cache()

    val model = NaiveBayes.train(trainingSet, lambda = 1.0,  modelType = "multinomial")

    val test_set = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(7).split(' ')
      (line, htf.transform(text))
    }

    val labeled = test_set.map{ item =>
      (item._1, model.predict(item._2))
    }.cache()

    labeled.saveAsTextFile("SelfLearningPredictions")

    }

}

