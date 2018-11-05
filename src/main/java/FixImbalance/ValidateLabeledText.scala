package FixImbalance

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object ValidateLabeledText {

  //   def conf = new SparkConf().setAppName("App_Name").setMaster("spark://localhost:6066").set("spark.ui.port","8080");
  //   var sc: SparkContext = _


  private[this] val logger = Logger.getLogger(getClass().getName());

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Evaluation of HighProb Labeled Data")//.setMaster("spark://localhost:77").set("spark.ui.port","8080");
    val argLen = args.length

    var sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val data = sc.textFile(args(0))
    val groundTruth = sc.textFile(args(argLen - 3))
    val exportDir = args(argLen - 2)
    val exportOrNot = args(argLen - 1)

    val htf = new HashingTF(1500000)

    var trainingSet = data.map { line =>
      val parts = line.split(',')
      var par = 0.0
      if (parts(3).equals("positive")){
        par = 1.0
      }else if(parts(3).equals("negative")){
        par = 0.0
      }else if (parts(3).equals("neutral")){
        par = parts(parts.length - 1).toDouble
      }
      LabeledPoint(par, htf.transform(parts(5).split(' ')))
    }.cache()

    for (cnt <- 1 until argLen - 3){
      println("fuse ground and labeled " + cnt)

      if (cnt == argLen - 4){
        trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
          val parts = line.split(',')
          var id= 0
          if (parts(3).equals("positive")){
            id = 1
          }

          LabeledPoint(id.toDouble, htf.transform(parts(5).split(' ')))
        }).cache()
      }else{
        trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
          val parts = line.split(',')
          val par = parts(parts.length - 1).toDouble
          LabeledPoint(par, htf.transform(parts(5).split(' ')))
        }).cache()
      }
    }

    println("labeled positives = " + trainingSet.filter(_.label == 1.0).count)
    println("labeled negatives = " + trainingSet.filter(_.label == 0.0).count)

    val testingSet = groundTruth.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(3).equals("positive")){
        id = 1
      }
      LabeledPoint(id.toDouble, htf.transform(parts(5).split(' ')))
    }.cache()

    val model = NaiveBayes.train(trainingSet, lambda = 1.0,  modelType = "multinomial")
    val predictionAndLabel = testingSet.map(p => (model.predict(p.features), p.label))


    if(exportOrNot.equals("true")) {
      val export = groundTruth.map { p =>
        val parts = p.split(',')
        val tweet_id = parts(0).replace("\t", "")

        var label = 0.0
        if (parts(3).equals("positive")) {
          label = 1.0
        }
        (tweet_id, label, model.predict(htf.transform(parts(5).split(' '))))
      }.saveAsTextFile(exportDir)
    }

    val metricsLabel = new MulticlassMetrics(predictionAndLabel)

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()

     // Overall Statistics
    println("Summary Statistics")
    println(s"accuracy = " + accuracy)
    println(s"confusionMatrix = " + metricsLabel.confusionMatrix)
    // Precision by label
    val labels = metricsLabel.labels
    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metricsLabel.fMeasure(l))
    }

    println("Weighted F1 = " + metricsLabel.weightedFMeasure(1.0))

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    val auPRC = metrics.areaUnderPR
    println(s"Area under precision-recall curve = $auPRC")
  }
}
