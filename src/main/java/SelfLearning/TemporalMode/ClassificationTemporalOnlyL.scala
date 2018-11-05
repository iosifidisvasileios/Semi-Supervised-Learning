package SelfLearning.TemporalMode

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object ClassificationTemporalOnlyL {
 
  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Holdout MNB")

    val sc = new SparkContext(conf)
    print("Hello, world!")
    sc.setLogLevel("ERROR")
    val format = new java.text.SimpleDateFormat("EEE MMM dd HH:mm:ss Z yyyy")

    val htf = new HashingTF(1500000)

    val data = sc.textFile(args(0))
    val evaluationData = sc.textFile(args(1))

    var counter = 0
//    train till november
    for (counter <- 0 to 10) {

      val trainingSet = data.filter { line =>
        val parts = line.split(',')
        var temp = format.parse(parts(1))
        temp.getMonth == counter
      }.map{
        line =>
          val parts = line.split(',')
          var id= 0
          if (parts(3).equals("positive")){
            id = 1
          }
          LabeledPoint(id.toDouble, htf.transform(parts(5).split(' ')))
      }.cache()


      val testingSet = evaluationData.filter { line =>
        val parts = line.split(',')
        var temp = format.parse(parts(1))
        temp.getMonth == counter + 1
      }.map{
        line =>
          val parts = line.split(',')
          var id= 0
          if (parts(3).equals("positive")){
            id = 1
          }
          LabeledPoint(id.toDouble, htf.transform(parts(5).split(' ')))
      }.cache()

      val model = NaiveBayes.train(trainingSet, lambda = 1.0, modelType = "multinomial")

      println("size of training "+ trainingSet.count())
      println("size of testing "+ testingSet.count())

      val predictionAndLabel = testingSet.map(p => (model.predict(p.features), p.label))
      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()

      println("accuracy for month " + (counter + 1) + " -> " + (counter + 2))
      println(accuracy)
    }
  }
}

