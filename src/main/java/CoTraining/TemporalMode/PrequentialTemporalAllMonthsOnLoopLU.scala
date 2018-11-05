package CoTraining.TemporalMode

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object PrequentialTemporalAllMonthsOnLoopLU {

  //   def conf = new SparkConf().setAppName("App_Name").setMaster("spark://localhost:6066").set("spark.ui.port","8080");
  //   var sc: SparkContext = _


  private[this] val logger = Logger.getLogger(getClass().getName())

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName(getClass.getName)

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")
    val format = new java.text.SimpleDateFormat("EEE MMM dd HH:mm:ss Z yyyy")

    val htf_F0 = new HashingTF(1500000)
    val htf_F1 = new HashingTF(3500000)

    var data = sc.textFile(args(0))
    val unlabeled = sc.textFile(args(1)).cache()
    val month = args(2).toInt

    //    train till november

    println("this month is: " + month )
    var trainingSetF0 = data.filter { line =>
      val parts = line.split(',')
      var temp = format.parse(parts(1))
      temp.getMonth <= month
    }.map{
      line =>
        val parts = line.split(',')
        var id= 0
        if (parts(3).equals("positive")){
          id = 1
        }
        LabeledPoint(id.toDouble, htf_F0.transform(parts(5).split(' ')))
    }.cache()

    var trainingSetF1 = data.filter { line =>
      val parts = line.split(',')
      var temp = format.parse(parts(1))
      temp.getMonth <= month
    }.map{
      line =>
        val parts = line.split(',')
        var id= 0
        if (parts(3).equals("positive")){
          id = 1
        }
        LabeledPoint(id.toDouble, htf_F1.transform(parts(6).split(' ')))
    }.cache()


    val testingSet = data.filter { line =>
      val parts = line.split(',')
      var temp = format.parse(parts(1))
      temp.getMonth == month + 1
    }.map{
      line =>
        val parts = line.split(',')
        var id= 0
        if (parts(3).equals("positive")){
          id = 1
        }
        (id.toDouble, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(parts(6).split(' ')))
    }.cache()

    var unlabeledF0 = unlabeled.filter { line =>
      val parts = line.split(',')
      var temp = format.parse(parts(1))
      temp.getMonth <= month
    }.map{
      line =>
        val parts = line.split(',')
        (line, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(parts(6).split(' ') ))
    }.cache

    val sizeOfTest = testingSet.count()

    var resultsHigh_L0: RDD[(Double, Vector)] = null
    var resultsHigh_L1: RDD[(Double, Vector)] = null
    for(step <- 1 to 2) {

      println("step: " + step)
      val model_F0 = NaiveBayes.train(trainingSetF0, lambda = 1.0, modelType = "multinomial")
      val model_F1 = NaiveBayes.train(trainingSetF1, lambda = 1.0, modelType = "multinomial")

      var predictionAndLabel = testingSet.map(p => (model_F0.predict(p._2), p._1))
      var accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / sizeOfTest
      println("Unigrams > " + accuracy)

      predictionAndLabel = testingSet.map(p => (model_F1.predict(p._3), p._1))
      accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / sizeOfTest
      println("Bigrams > " + accuracy)

      println("size of training Size: " + trainingSetF0.count)
      println("size of testing  Size: " + sizeOfTest)

      if (step == 1) {

//        println("size of unlabeld Size: " + unlabeledF0.count)
        resultsHigh_L0 = unlabeledF0.map { item =>
          (model_F0.predict(item._2), item._3)
        }.cache()

        resultsHigh_L1 = unlabeledF0.map { item =>
          (model_F1.predict(item._3), item._2)
        }.cache()

        trainingSetF0 = trainingSetF0.union(resultsHigh_L1.map { line => LabeledPoint(line._1, line._2) })
        trainingSetF1 = trainingSetF1.union(resultsHigh_L0.map { line => LabeledPoint(line._1, line._2) })

      }else{



        val pos = resultsHigh_L0.filter { point =>
          point._1 == 1.0
        }.count()

        val neg = resultsHigh_L0.filter { point =>
          point._1 == 0.0
        }.count()

        println("Positive labeled produced by bigrams: "+ pos )
        println("Negative labeled produced by bigrams: "+ neg )

      }
    }

//    temp.getMonth == month || temp.getMonth == month - 1 || temp.getMonth == month - 2


  }
}

