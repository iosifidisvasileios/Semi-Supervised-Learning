package FixImbalance

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object CoTrainPredict {

  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName("Classification Using CoTrain Technique")

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val htf_F0 = new HashingTF(1500000)
    val htf_F1 = new HashingTF(3500000)


    val argsSize = args.length
    val indexL0 = ((argsSize - 8 + 1)/2 ) - 1
    val indexL1 = (argsSize - 8 + 1)/2

    var dataL0 = sc.textFile(args(0))
    var dataL1 = sc.textFile(args(indexL1))

    for (cnt <- 1 to indexL0){
      dataL0 = dataL0.union(sc.textFile(args(cnt)))
    }

    for (cnt <- indexL1 + 1 to argsSize - 8){
      dataL1 = dataL1.union(sc.textFile(args(cnt)))
    }

    val unlabeledL0 = sc.textFile(args(argsSize - 7))
    val unlabeledL1 = sc.textFile(args(argsSize - 6))

    val confidenceLevel = args(argsSize - 5).toDouble

    val trainingSet_L0 = dataL0.map { line =>
      val parts = line.split(',')
      var id= 0.0
      if (parts(3).equals("positive")){
        id = 1.0
      }else if (parts(3).equals("negative")){
        id = 0.0
      }else if (parts(3).equals("neutral")){
        id = parts(parts.length - 1).toDouble
      }
      LabeledPoint(id, htf_F0.transform(parts(5).split(' ')))
    }.cache()

    val trainingSet_L1 = dataL1.map { line =>
      val parts = line.split(',')
      var id= 0.0
      if (parts(3).equals("positive")){
        id = 1.0
      }else if (parts(3).equals("negative")){
        id = 0.0
      }else if (parts(3).equals("neutral")){
        id = parts(parts.length - 1).toDouble
      }

      val words = parts(5).split(" ")
      var bigrams = ""

      for(i <- 0 until words.length - 1){
        bigrams += words(i) + "_" + words( i + 1) + " "
      }

      LabeledPoint(id, htf_F1.transform(bigrams.trim.split(' ')))
    }.cache()


    val unlabeled_F0 = unlabeledL0.map { line =>
      val parts = line.split(',')

      val words = parts(5).split(" ")
      var bigrams = ""

      for(i <- 0 until words.length - 1){
        bigrams += words(i) + "_" + words( i + 1) + " "
      }
      (line, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(bigrams.trim.split(' ')))
    }

    val unlabeled_F1 = unlabeledL1.map { line =>
      val parts = line.split(',')
      val words = parts(5).split(" ")
      var bigrams = ""

      for(i <- 0 until words.length - 1){
        bigrams += words(i) + "_" + words( i + 1) + " "
      }
      (line, htf_F0.transform(parts(5).split(' ')), htf_F1.transform(bigrams.trim.split(' ')))
    }


    println("size of training UniBrams Positive: " + trainingSet_L0.filter(_.label == 1.0).count())
    println("size of training UniBrams Negative: " + trainingSet_L0.filter(_.label == 0.0).count())

    println("size of training Bi Grams Positive: " + trainingSet_L1.filter(_.label == 1).count())
    println("size of training Bi Grams Negative: " + trainingSet_L1.filter(_.label == 0).count())

    val model_F0 = NaiveBayes.train(trainingSet_L0, lambda = 1.0, modelType = "multinomial")
    val model_F1 = NaiveBayes.train(trainingSet_L1, lambda = 1.0, modelType = "multinomial")

//    println("*********************************************************")

    // put htf (8) to vector so dont compute them when added to trainingsetL1
    // labeled L0
    unlabeled_F0.filter { item =>
      val output = model_F0.predictProbabilities(item._2)
      output(0) >= confidenceLevel || output(1) >= confidenceLevel
    }.map{ line=>
      line._1 + "," + model_F0.predict(line._2).toInt
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
      line._1 + "," + model_F1.predict(line._3).toInt
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

 //TrainingSet Augmented_0
//Augmented_0 Augmented_0 unlabeled unlabeled 0.65 L0_0 U0_0 L1_0 U1_0


//TrainingSet L1_0 Augmented_1
//TrainingSet L0_0 Augmented_0
//Augmented_1 Augmented_0 U0_0 U1_0 0.65 L0_1 U0_1 L1_1 U1_1

 //TrainingSet L1_0 L1_1 Augmented_1
//TrainingSet L0_0 L0_1 Augmented_0
//Augmented_1 Augmented_0 U0_1 U1_1 0.65 L0_2 U0_2 L1_2 U1_2

