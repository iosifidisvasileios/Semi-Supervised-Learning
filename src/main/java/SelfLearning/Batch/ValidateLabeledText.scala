package SelfLearning.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.log4j._
import org.apache.spark.mllib.classification.NaiveBayes
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
    val groundTruth = sc.textFile(args(argLen - 1))
    val htf = new HashingTF(1500000)

    var trainingSet = data.map { line =>
      val parts = line.split(',')
      val par = parts(parts.length - 1).toDouble
      LabeledPoint(par, htf.transform(parts(5).split(' ')))
    }.cache()

    for (cnt <- 1 until argLen - 1){
      println("fuse ground and labeled " + cnt)

      if (cnt == argLen - 2 ){
        trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
          val parts = line.split(',')
          var id= 0
          if (parts(3).equals("positive")){
            id = 1
          }

          LabeledPoint(id.toDouble, htf.transform(parts(5).split(' ')))
        }).cache()
      }else{

        //      log.++("fuse ground and labeled " + cnt)
        trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
          val parts = line.split(',')
          val par = parts(parts.length - 1).toDouble
          LabeledPoint(par, htf.transform(parts(5).split(' ')))
        }).cache()
      }
    }

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
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()
    println(accuracy)
    //    log.++("Accuracy : " + accuracy.toString)
    //    sc.parallelize(log).saveAsTextFile("SelfLearn/" + "Accuracy_" + (argLen-2).toString +  ".log")


  }
}
