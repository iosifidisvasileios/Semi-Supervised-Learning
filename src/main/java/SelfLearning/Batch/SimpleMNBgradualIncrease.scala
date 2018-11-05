package SelfLearning.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object SimpleMNBgradualIncrease {



  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Simple MNB Gradual Unlabeled Increase") //.setMaster("spark://localhost:7077").set("spark.ui.port","8080");
    val argLen = args.length

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val htf = new HashingTF(1400000)

    val data = sc.textFile(args(0))
    val evaluationData = sc.textFile(args(1))

    val trainingSet = data.map { line =>
      val parts = line.split('\t')
      LabeledPoint(parts(1).toDouble, htf.transform(parts(3).split(' ')))
    }.cache()

    val testingSet = evaluationData.map { line =>
      val parts = line.split('\t')
      LabeledPoint(parts(1).toDouble, htf.transform(parts(3).split(' ')))
    }.cache()

    val model = NaiveBayes.train(trainingSet, lambda = 1.0,  modelType = "multinomial")

    val labeled = testingSet.map{ item =>
      LabeledPoint(model.predict(item.features) , item.features)
    }.cache()
    // train based on new labeled corpus
    val newTraining = labeled.union(trainingSet).cache()
    val evaluator_model = NaiveBayes.train(newTraining, lambda = 1.0,  modelType = "multinomial")
    //evaluate it
    val predictionAndLabel = testingSet.map(p => (evaluator_model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSet.count()


    println(accuracy)
    println("Positives: " + (newTraining.filter( _.label == 4).count - trainingSet.filter( _.label == 4).count))
    println("Negatives: " + (newTraining.filter( _.label == 0).count - trainingSet.filter( _.label == 0).count))


  }
}
