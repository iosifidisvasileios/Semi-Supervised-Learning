package FixImbalance

import java.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object SelfLearning {
  val word2vecFromGoogleMapper = new mutable.HashMap[String, ArrayBuffer[String]]()  { }
  var train:RDD[LabeledPoint]= null



  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName("Self Learning Imbalance Fix")
    val argLen = args.length

    val sc = new SparkContext(conf)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    sc.setLogLevel("ERROR")
    val sentiwordSet = new SentiwordLoader().getDictionary

    for (line <- Source.fromFile("glove_300_dim.tsv").getLines) {

      val ab = ArrayBuffer[String]()
      val words = line.split("\t")

      for(i <- 1 until words.length){
        ab += words(i).toLowerCase
      }
      word2vecFromGoogleMapper += (words(0).toLowerCase -> ab)
    }

    val data = sc.textFile(args(0))
    var optionForAugmentation = args(argLen - 1)

    val prediction = args(argLen - 2).toDouble
    val unlabeled = sc.textFile(args(argLen - 5))
    val htf = new HashingTF(1500000)

    var trainingSet = data.map { line =>
      val parts = line.split(',')
      var par = 0

      if(parts(3).equals("positive")){
        par = 1
      }else if(parts(3).equals("negative")){
        par = 0
      }else {
        par = parts(parts.length - 1).toInt
      }
      (par.toDouble, line)

    }.cache()

    for (cnt <- 1 until argLen - 5){
      println("fuse ground and labeled " + cnt)
      trainingSet = trainingSet.union(sc.textFile(args(cnt)).map { line =>
        val parts = line.split(',')
        val par = parts(parts.length - 1).toDouble
        (par, line)
      }).cache()
    }

    var positivesTraining = trainingSet.filter(_._1 == 1)
    var negativesTraining = trainingSet.filter(_._1 == 0)

    var positiveCount = positivesTraining.count
    var negativeCount = negativesTraining.count

    val staticIncreasementNegatives = negativeCount
    println("size of positive in training set = " + positiveCount)
    println("size of negative (Initial) in training set = " + negativeCount)
    val distance = Math.abs((positiveCount.toDouble - negativeCount.toDouble)/(positiveCount.toDouble + negativeCount.toDouble))

    println(optionForAugmentation)
    println("distance = " + distance)

    if (distance <= 0.20){
      println("distance = " + distance)
      optionForAugmentation = "balanced"
    }

    // Augmentation methods
    if (optionForAugmentation.equals("blankout")){
      println(optionForAugmentation)
      var iteration = 0
      while(negativeCount < positiveCount + staticIncreasementNegatives){
        iteration += 1
        blankout_generator_with_filter(negativesTraining, sentiwordSet, 1, iteration)
        negativeCount += staticIncreasementNegatives
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)
      }

      val augmented = sc.textFile("BlankoutAugmentation/*").map { line =>

        val parts = line.split(",")
        val label = parts(0).replace("(","").toDouble
        val text = parts(1).replace(")","")
        LabeledPoint(label, htf.transform(text))
      }.union(negativesTraining.map{line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      })

      train = positivesTraining.map{ line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      }.union(augmented).cache()

    }else if(optionForAugmentation.equals("embeddings")) {

      var iteration = 0
      while (negativeCount < positiveCount) {
        iteration += 1
        wordEmbeddingsGenerator(negativesTraining, word2vecFromGoogleMapper, iteration)
        negativeCount += staticIncreasementNegatives
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)


      }
      val augmented = sc.textFile("AugmentedEmbeddings/*").map { line =>

        val parts = line.split(",")

        val label = parts(0).replace("(","").toDouble
        val text = parts(1).replace(")","")
        LabeledPoint(label, htf.transform(text))
      }.union(negativesTraining.map{line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      })

      train = positivesTraining.map{ line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      }.union(augmented).cache()


    }else if (optionForAugmentation.equals("sampling")){

      positivesTraining = positivesTraining.randomSplit(Array(0.5, 0.5 ))(0)
      positiveCount = positivesTraining.count()

      var iteration = 0
      while(negativeCount < positiveCount){
        iteration += 1
        oversampling(negativesTraining, iteration)
        negativeCount += staticIncreasementNegatives / 2
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)
      }

      val augmented = sc.textFile("OverSampling/*").map { line =>

        val parts = line.split(",")

        val label = parts(0).replace("(","").toDouble
        val text = parts(1).replace(")","")
        LabeledPoint(label, htf.transform(text))
      }.union(negativesTraining.map{line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      })

      train = positivesTraining.map{ line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      }.union(augmented).cache()

    }else if (optionForAugmentation.equals("balanced")) {

      train = trainingSet.map { line=>
        LabeledPoint(line._1, htf.transform(line._2.split(",")(5).split(' ')))
      }

    }

    // label propagation
    println("final positives = " + train.filter(_.label == 1).count() )
    println("final negatives = " + train.filter(_.label == 0).count() )

    val model = NaiveBayes.train(train , lambda = 1.0, modelType = "multinomial")

    val forPrediction = unlabeled.map { line =>
      val parts = line.split(',')
      val text = parts(5).split(' ')
      (line, htf.transform(text))
    }.cache()

    var unlabeldCountCorpus = forPrediction.count
    println("TRAINING SET SIZE     :------------------------------------------------: "  + trainingSet.count())
    println("UnLabel  SET SIZE     :------------------------------------------------: "  + forPrediction.count())

    val lowProbabilitiesSet = forPrediction.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) < prediction && model.predictProbabilities(item._2)(1) < prediction) {
        List(item._1)
      } else {
        None
      }
    }.cache()

    val highProbabilitiesSet = forPrediction.flatMap { item =>
      if (model.predictProbabilities(item._2)(0) >= prediction || model.predictProbabilities(item._2)(1) >= prediction) {
        List(item._1 +","+ model.predict(item._2).toInt)
      } else {
        None
      }
    }.cache()

    var remainingUnlabeled = lowProbabilitiesSet.count()
    println("Unlabeled SET SIZE     :------------------------------------------------: " + remainingUnlabeled )
    println("Labeled SET SIZE     :------------------------------------------------: "  + (unlabeldCountCorpus - remainingUnlabeled))

    highProbabilitiesSet.saveAsTextFile(args(argLen - 4))
    lowProbabilitiesSet.saveAsTextFile(args(argLen - 3))


    if (optionForAugmentation.equals("blankout")) {
      fs.delete(new Path("BlankoutAugmentation/"),true)
    } else if (optionForAugmentation.equals("embeddings")){
      fs.delete(new Path("AugmentedEmbeddings/"),true)
    } else if(optionForAugmentation.equals("sampling")){
      fs.delete(new Path("OverSampling/"),true)
    }
  }



  def blankout_generator_with_filter(negatives: RDD[(Double, String)],
                                     sentiwordSet: util.HashSet[String],
                                     k: Int,
                                     count: Int) : Unit = {
    val filtered = negatives.filter(x => x._2.split(",")(5).split(" ").length - k >= 4)

    filtered.map{ line =>
      val parts = line._2.split(',')
      val number_of_words = parts(5).split(' ').length
      var tempString = ""
      val indexSet = scala.collection.mutable.Set[Int]()

      val sentence = parts(5).split(' ')

      var special_counter = 0

      while(special_counter <= 10 && indexSet.size != k ){
        val r = new java.util.Random

        special_counter += 1
        val random_dropout = r.nextInt(number_of_words)

        if (
          !sentence(random_dropout).startsWith("NOT_") &&
            !sentence(random_dropout).startsWith("not_") &&
            !sentence(random_dropout).endsWith("dont") &&
            !sentence(random_dropout).endsWith("havent") &&
            !sentence(random_dropout).endsWith("cant") &&
            !sentence(random_dropout).endsWith("cannot") &&
            !sentence(random_dropout).endsWith("arent") &&
            !sentence(random_dropout).endsWith("aint") &&
            !sentiwordSet.contains(sentence(random_dropout))){
          indexSet.add(random_dropout)
        }
      }

      if(indexSet.nonEmpty ) {
        var counter = 0
        for (word <- parts(1).split(" ")) {
          if (!indexSet.contains(counter)) {
            tempString += word.toLowerCase + " "
          }
          counter += 1
        }

        (line._1, tempString.trim())
      }else{
        (line._1, "null")
      }

    }.filter(x=> !x._2.equals("null")).saveAsTextFile("BlankoutAugmentation/number_" + count)
  }

  def wordEmbeddingsGenerator(data: RDD[(Double,String)],
                              mapper : mutable.HashMap[String, ArrayBuffer[String]],
                              count : Integer): Unit = {

    data.map { line =>

      val parts = line._2.split(",")
      val text = line._2.split(",")(5)
      var outputStr = ""

      var unigramsSynonymous = ""
      var similarityFlag = false
      val rnd = new scala.util.Random

      for (word <- text.split(" ")){
        if (mapper.contains(word)){
          similarityFlag = true
          unigramsSynonymous += mapper(word)(rnd.nextInt(mapper(word).size)).toLowerCase + " "
        }else{
          unigramsSynonymous += word.toLowerCase  + " "
        }
      }

      if(similarityFlag ){
        outputStr = unigramsSynonymous.trim()
        (line._1, outputStr)

      }else{
        (line._1, "null")
      }
    }.filter(x=> !x._2.equals("null")).saveAsTextFile("AugmentedEmbeddings/number_" + count)
  }


  def oversampling(data: RDD[(Double, String)], count: Int): Unit = {
    val r = scala.util.Random
    data.randomSplit(Array(0.5, 0.5), r.nextLong())(0).map{ line =>
      (line._1, line._2.split(',')(5))
    }.saveAsTextFile("OverSampling/number_" + count)

  }


}
