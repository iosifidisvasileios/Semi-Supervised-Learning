package SelfLearning.Batch

/**
  * Created by iosifidis on 10.10.16.
  */

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.slf4j.LoggerFactory

import scala.util.control.Breaks._


object SelfTest {
  def main(args: Array[String]) {

    val logger = LoggerFactory.getLogger(this.getClass)
    val sc = new SparkContext()
    sc.setLogLevel("ERROR")
    if (args.length != 3){
      logger.error("must provide path to 1: training_set 2: test_set training_set 3: size of hashfunction")
      sys.exit(1)
    }
    println("starting process...")
    val data = sc.textFile(args(0))
    val unlabeled_1 = sc.textFile(args(1))
//    val unlabeled_2 = sc.textFile(args(2))
    val htf = new HashingTF(args(2).toInt)

    var temp_train = data.map { line =>
      val parts = line.split(',')
      var par = 0

      if(parts(5).equals("positive")){
        par = 1
      }
      LabeledPoint(par.toDouble, htf.transform(parts(7).split(' ')))
    }

    println("training set loaded...")

    var xxxModel = NaiveBayes.train(temp_train)

//    var xxxModel = SVMWithSGD.train(temp_train, 10)
    println("classifier trained")

    var test_set = unlabeled_1.map { line =>
      val parts = line.split(',')
      val text = parts(7).split(' ')
      (line, htf.transform(text))
    }

    println("test set loaded...")

    var previousCounter = 0L
    breakable {

      for (counter <- 1 to 5) {
        println("-------------------  This is the_" + counter + " run !!! -----------------")
        var updated_trainCnt = temp_train.count()
        var updated_testCnt = test_set.count()
        println("Updated Train SET SIZE: " + updated_trainCnt)
        println("Updated Testing SET SIZE: " + updated_testCnt)

        val highProbabilitiesSet = test_set.filter { item =>
          val output = xxxModel.predictProbabilities(item._2)
          output(0) > 0.75 || output(1) > 0.75
        }.map(item => List(item._1 + "," + xxxModel.predict(item._2))) //.cache()


        val test_set_2 = test_set.filter { item =>
          val output = xxxModel.predictProbabilities(item._2)
          output(0) <= 0.75 && output(1) <= 0.75
        }.map(item => List(item._1))

        val hiCnt = highProbabilitiesSet.count()
        val lowCnt = test_set_2.count()

        println("HIGH PROBAB SET : " + hiCnt)
        println("LOW PROBAB SET  : " + lowCnt)
        val diff = updated_testCnt - hiCnt - lowCnt
        if (diff != 0) println("ERROR: Test set not correctly split into high low" + diff)
        temp_train = temp_train.union(highProbabilitiesSet.map(x => LabeledPoint(x.head.split(",")(8).toDouble, htf.transform(x.head.split(",")(7).split(" "))))).cache()
        highProbabilitiesSet.unpersist()

        xxxModel = NaiveBayes.train(temp_train)

        test_set.unpersist()
        test_set = null
        test_set = test_set_2.map { line =>
          val parts = line.head.split(',')
          val text = parts(7).split(' ')
          (line.head, htf.transform(text))
        }

        if(lowCnt - previousCounter == 0){
          println("converged at " + counter + ". Now break!!!")
          break
        }
        previousCounter = lowCnt

        //      xxxModel = sc.broadcast(model)
        //      println("HIGH PROBAB SET : "  + highProbabilitiesSet.count())
        //      println("LOW PROBAB SET  : "  + test_set.count())
        //      println("NEW TRAINING SET: "  + temp_train.count())
      }
    }
  }
}