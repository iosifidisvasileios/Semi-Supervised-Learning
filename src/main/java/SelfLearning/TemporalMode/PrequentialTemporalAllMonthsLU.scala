package SelfLearning.TemporalMode

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object PrequentialTemporalAllMonthsLU {

  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Temporal LU Holdout MNB")

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")
    val format = new java.text.SimpleDateFormat("EEE MMM dd HH:mm:ss Z yyyy")

    val htf = new HashingTF(1500000)

    var data = sc.textFile(args(0))
    val unlabeled = sc.textFile(args(1))


    var counter = 0
    //    train till november
    for (counter <- 0 to 10) {

      val trainingSet = data.filter { line =>
        val parts = line.split(',')
        var temp = format.parse(parts(1))
        temp.getMonth <= counter
      }.map{
        line =>
          val parts = line.split(',')
          var id= 0
          if (parts(5).equals("positive")){
            id = 1
          }
          LabeledPoint(id.toDouble, htf.transform(parts(7).split(' ')))
      }.cache()


      val testingSet = data.filter { line =>
        val parts = line.split(',')
        var temp = format.parse(parts(1))
        temp.getMonth == counter + 1
      }.map{
        line =>
          val parts = line.split(',')
          var id= 0
          if (parts(5).equals("positive")){
            id = 1
          }
          LabeledPoint(id.toDouble, htf.transform(parts(7).split(' ')))
      }.cache()

      val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")

      val test_set = unlabeled.filter { line =>
        val parts = line.split(',')
        var temp = format.parse(parts(1))
        temp.getMonth <= counter
      }.map{
        line =>
          val parts = line.split(',')
          val text = parts(7).split(' ')
          htf.transform(text)
      }

      val labeled = test_set.map { item =>
        LabeledPoint(model.predict(item), item)
      }.cache()
      // train based on new labeled corpus
      val newTraining = labeled.union(trainingSet).cache()
      val evaluator_model = NaiveBayes.train(newTraining, lambda = 1.0, modelType = "multinomial")
      //evaluate it

      println("size of training "+ newTraining.count())
      println("size of testing "+ testingSet.count())

      val predictionAndLabel = testingSet.map(p => (evaluator_model.predict(p.features), p.label))
      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()

      println("accuracy for month " + (counter + 1) + " -> " + (counter + 2))
      println(accuracy)
    }
  }
}

