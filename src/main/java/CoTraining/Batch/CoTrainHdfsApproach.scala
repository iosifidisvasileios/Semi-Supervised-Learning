package CoTraining.Batch

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object CoTrainHdfsApproach {

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Using CoTrain Technique")

    val sc = new SparkContext(conf)
    println("Hello, world!!!")
    sc.setLogLevel("ERROR")

    val htf_F0 = new HashingTF(1500000)
    val htf_F1 = new HashingTF(3500000)


    val argsSize = args.length
    val indexL0 = ((argsSize - 9 + 1)/2 ) - 1
    val indexL1 = (argsSize - 9 + 1)/2

    var dataL0 = sc.textFile(args(0))
    var dataL1 = sc.textFile(args(indexL1))

    for (cnt <- 1 to indexL0){
      dataL0 = dataL0.union(sc.textFile(args(cnt)))
    }

    for (cnt <- indexL1 + 1 to argsSize - 9){
      dataL1 = dataL1.union(sc.textFile(args(cnt)))
    }

    val evaluationData = sc.textFile(args(argsSize - 8))

    val unlabeledL0 = sc.textFile(args(argsSize - 7))
    val unlabeledL1 = sc.textFile(args(argsSize - 6))

    val confidenceLevel = args(argsSize - 5).toDouble

    val testingSet = evaluationData.map { line =>
      val parts = line.split(',')
      var id= 0
      if (parts(3).equals("positive")){
        id = 1
      }
      (id.toDouble, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(parts(6).split(' ')))
    }.cache()

    val trainingSet_L0 = dataL0.map { line =>
      val parts = line.split(',')
      var id= 0.0
      if (parts(3).equals("positive")){
        id = 1.0
      }else if (parts(3).equals("neutral")){
        id = parts(7).toDouble
      }
      LabeledPoint(id, htf_F0.transform(parts(5).split(' ')))
    }.cache()

    val trainingSet_L1 = dataL1.map { line =>
      val parts = line.split(',')
      var id= 0.0
      if (parts(3).equals("positive")){
        id = 1.0
      }else if (parts(3).equals("neutral")){
        id = parts(7).toDouble
      }
      LabeledPoint(id, htf_F1.transform(parts(6).split(' ')))
    }.cache()

    val testingSetSize = testingSet.count()

    val unlabeled_F0 = unlabeledL0.map { line =>
      val parts = line.split(',')
      (line, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(parts(6).split(' ') ))
    }

    val unlabeled_F1 = unlabeledL1.map { line =>
      val parts = line.split(',')
      (line, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(parts(6).split(' ') ))
    }

    val model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
    val model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

    var predictionAndLabel = testingSet.map(p => (model_F0.predict(p._2), p._1))
    var accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSetSize

    println("Accuracy for UniGrams :" + accuracy)

    predictionAndLabel = testingSet.map(p => (model_F1.predict(p._3), p._1))
    accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testingSetSize
    println("Accuracy for BiGrams  :" + accuracy)

    println("size of training UniBrams: " + trainingSet_L0.count())
    println("size of training Bi Grams: " + trainingSet_L1.count())

    println("size of unlabeled UniGrams:" + unlabeled_F0.count())
    println("size of unlabeled Bi Grams:" + unlabeled_F1.count())
    println("*********************************************************")

    // put htf (8) to vector so dont compute them when added to trainingsetL1
    // labeled L0
    unlabeled_F0.filter { item =>
      val output = model_F0.predictProbabilities(item._2)
      output(0) >= confidenceLevel || output(1) >= confidenceLevel
    }.map{ line=>
      line._1 + "," + model_F0.predict(line._2)
    }.saveAsTextFile(args(argsSize - 4))

    // UN labeled L0
    unlabeled_F0.filter { item =>
      val output = model_F0.predictProbabilities(item._2)
      output(0) < confidenceLevel && output(1) < confidenceLevel
    }.map{ line =>
      line._1
    }.saveAsTextFile(args(argsSize - 3))

    // put htf (7) to vector so dont compute them when added to trainingsetL0
    // labeled L1
    unlabeled_F1.filter { item =>
      val output = model_F1.predictProbabilities(item._3)
      output(0) >= confidenceLevel || output(1) >= confidenceLevel
    }.map{ line=>
      line._1 + "," + model_F1.predict(line._3)
    }.saveAsTextFile(args(argsSize - 2))

    //UN labeled L1
    unlabeled_F1.filter { item =>
      val output = model_F1.predictProbabilities(item._3)
      output(0) < confidenceLevel && output(1) < confidenceLevel
    }.map{ line =>
      line._1
    }.saveAsTextFile(args(argsSize - 1))

  }
}
