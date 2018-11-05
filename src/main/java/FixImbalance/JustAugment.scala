package FixImbalance

import java.util

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object JustAugment {
  val word2vecFromGoogleMapper = new mutable.HashMap[String, ArrayBuffer[String]]()  { }
  var train:RDD[LabeledPoint]= null

  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName("Augmentation")
    val argLen = args.length

    val sc = new SparkContext(conf)

    sc.setLogLevel("ERROR")

    for (line <- Source.fromFile("glove_300_dim.tsv").getLines) {

      val ab = ArrayBuffer[String]()
      val words = line.split("\t")

      for(i <- 1 until words.length){
        ab += words(i).toLowerCase
      }
      word2vecFromGoogleMapper += (words(0).toLowerCase -> ab)
    }
    val sentiwordSet = new SentiwordLoader().getDictionary

    val data = sc.textFile(args(0))
    var optionForAugmentation = args(argLen - 1)


    var trainingSet = data.map { line =>
      val parts = line.split(',')
      var par = 0

      if(parts(3).equals("positive")){
        par = 1
      }
      (par.toDouble, line)

    }.cache()

    for (cnt <- 1 until argLen - 1){
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
    val distance = Math.abs((positiveCount.toDouble - negativeCount.toDouble)/(positiveCount.toDouble + negativeCount.toDouble))


    val flagDistanceIsOK = optionForAugmentation

    println("size of positive in training set = " + positiveCount)
    println("size of negative (Initial) in training set = " + negativeCount)
    println("distance = " + distance)

    if (distance <= 0.20){
      println("distance = " + distance)
      optionForAugmentation = "balanced"
    }

    // Augmentation methods
    if (optionForAugmentation.equals("blankout")){
      //      println(optionForAugmentation)
      var iteration = 0

      flushToHDFS(trainingSet,"BlankoutAugmentation/number_" + iteration)

      while(negativeCount < positiveCount  ){
        iteration += 1
        negativeCount += staticIncreasementNegatives
        blankout_generator_with_filter(negativesTraining, sentiwordSet, 1, iteration)
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)
      }

    }else if(optionForAugmentation.equals("embeddings")) {
      //      println(optionForAugmentation)
      var iteration = 0

      flushToHDFS(trainingSet,"AugmentedEmbeddings/number_" + iteration)

      while (negativeCount < positiveCount) {
        iteration += 1
        negativeCount += staticIncreasementNegatives
        wordEmbeddingsGenerator(negativesTraining, word2vecFromGoogleMapper, iteration)
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)

      }

    }else if (optionForAugmentation.equals("sampling")) {

      var iteration = 0
      oversampling(positivesTraining, iteration)
      positiveCount = positivesTraining.count() / 2

      if (positiveCount < negativeCount) {
        println("less positives than negatives. store existing negative instances")
        flushToHDFS(negativesTraining, "OverSampling/number_1")
      }

      while (negativeCount < positiveCount) {
        iteration += 1
        oversampling(negativesTraining, iteration)
        negativeCount += staticIncreasementNegatives / 2
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)
      }

    }else if(optionForAugmentation.equals("undersampling")){
      flushToHDFS(sc.parallelize(positivesTraining.take(negativeCount.toInt)).union(negativesTraining), "UnderSampling")

    }else if(optionForAugmentation.equals("oversampling")){
      var iteration = 0
      negativeCount = 0
      flushToHDFS(positivesTraining, "OverSampling/number_0")

      while (negativeCount < positiveCount) {
        iteration += 1
        oversampling(negativesTraining, iteration)
        negativeCount += staticIncreasementNegatives / 2
        println("iteration = " + iteration + ". Size of negative in training set = " + negativeCount)
      }

    }else if(optionForAugmentation.equals("balanced")){
      if(flagDistanceIsOK.equals("blankout")){
        flushToHDFS(trainingSet,"BlankoutAugmentation/number")
      }else if(flagDistanceIsOK.equals("embeddings")){
        flushToHDFS(trainingSet,"AugmentedEmbeddings/number")
      }else if(flagDistanceIsOK.equals("sampling")){
        flushToHDFS(trainingSet,"OverSampling/number")
      }else if(flagDistanceIsOK.equals("undersampling")){
        flushToHDFS(trainingSet,"UnderSampling")
      }else if(flagDistanceIsOK.equals("oversampling")){
        flushToHDFS(trainingSet,"OverSampling/number")
      }

    }

  }

  def flushToHDFS(data:  RDD[(Double, String)], directory: String): Unit ={
    data.map(_._2).saveAsTextFile(directory)
  }


  def blankout_generator_with_filter(negatives: RDD[(Double, String)],
                                     sentiwordSet: util.HashSet[String],
                                     k: Int,
                                     count: Int) : Unit = {

    negatives
      .filter(x => x._2.split(",")(5).split(" ").length - k >= 4)
      .map{ line =>

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
        var outputString = "null"

        if(indexSet.nonEmpty ) {
          var counter = 0
          for (word <- sentence) {
            if (!indexSet.contains(counter)) {
              tempString += word.toLowerCase + " "
            }
            counter += 1
          }
          outputString = parts(0) + "," + parts(1) + "," + parts(2) + "," + parts(3) + "," +parts(4) + ","  + tempString.trim() + "," + parts(parts.length - 1)

        }
        outputString
      }
      .filter(x=>  !x.equals("null"))
      .saveAsTextFile("BlankoutAugmentation/number_" + count)

    //    filtered2.saveAsTextFile("BlankoutAugmentation/number_" + count)
    //    filtered2.count().toInt
  }

  def wordEmbeddingsGenerator(data: RDD[(Double,String)],
                              mapper : mutable.HashMap[String, ArrayBuffer[String]],
                              count : Integer): Unit = {

    data.map { line =>
      val parts = line._2.split(",")
      val text = line._2.split(",")(5)

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

      var outputString = "null"
      if(similarityFlag ){
        outputString = parts(0) + "," + parts(1) + "," + parts(2) + "," + parts(3) + "," + parts(4) + ","  + unigramsSynonymous.trim() + "," + parts(parts.length - 1)
      }
      outputString

    }.filter(x=> !x.equals("null"))
     .saveAsTextFile("AugmentedEmbeddings/number_" + count)
  }


  def oversampling(data: RDD[(Double, String)], count: Int): Unit = {
    val r = scala.util.Random
    flushToHDFS(data.randomSplit(Array(0.5, 0.5), r.nextLong())(0), "OverSampling/number_" + count)
  }


}
