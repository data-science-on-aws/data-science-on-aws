/**
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.linkedin.spark.datasources.tfrecord

import org.apache.hadoop.fs.Path
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SaveMode}

import TestingUtils._

class TFRecordIOSuite extends SharedSparkSessionSuite {

  val exampleSchema = StructType(List(
    StructField("id", IntegerType),
    StructField("IntegerLabel", IntegerType),
    StructField("LongLabel", LongType),
    StructField("FloatLabel", FloatType),
    StructField("DoubleLabel", DoubleType),
    StructField("DecimalLabel", DataTypes.createDecimalType()),
    StructField("StrLabel", StringType),
    StructField("BinaryLabel", BinaryType),
    StructField("IntegerArrayLabel", ArrayType(IntegerType)),
    StructField("LongArrayLabel", ArrayType(LongType)),
    StructField("FloatArrayLabel", ArrayType(FloatType)),
    StructField("DoubleArrayLabel", ArrayType(DoubleType, true)),
    StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType())),
    StructField("StrArrayLabel", ArrayType(StringType, true)),
    StructField("BinaryArrayLabel", ArrayType(BinaryType), true))
  )

  val exampleTestRows: Array[Row] = Array(
    new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, Decimal(1.1), "r1", Array[Byte](0xff.toByte, 0xf0.toByte),
      Seq(1, 2),
      Seq(11L, 12L),
      Seq(1.2F, 2.1F),
      Seq(1.1, 2.2),
      Seq(Decimal(1.1), Decimal(2.2)),
      Seq("str1", "str2"),
      Seq(Array[Byte](0xfa.toByte, 0xfb.toByte), Array[Byte](0xfa.toByte)))),
    new GenericRow(Array[Any](11, 1, 24L, 11.0F, 15.0, Decimal(2.1), "r2", Array[Byte](0xfa.toByte, 0xfb.toByte),
      Seq(3, 4),
      Seq(110L, 120L),
      Seq(1.2F, 2.1F),
      Seq(1.1, 2.2),
      Seq(Decimal(2.1), Decimal(3.2)),
      Seq("str3", "str4"),
      Seq(Array[Byte](0xf1.toByte, 0xf2.toByte), Array[Byte](0xfa.toByte)))),
    new GenericRow(Array[Any](21, 1, 23L, 10.0F, 14.0, Decimal(3.1), "r3", Array[Byte](0xfc.toByte, 0xfd.toByte),
      Seq(5, 6),
      Seq(111L, 112L),
      Seq(1.22F, 2.11F),
      Seq(11.1, 12.2),
      Seq(Decimal(3.1), Decimal(4.2)),
      Seq("str5", "str6"),
      Seq(Array[Byte](0xf4.toByte, 0xf2.toByte), Array[Byte](0xfa.toByte)))))

  val sequenceExampleTestRows: Array[Row] = Array(
    new GenericRow(Array[Any](23L, Seq(Seq(2, 4)), Seq(Seq(-1.1F, 0.1F)), Seq(Seq("r1", "r2")))),
    new GenericRow(Array[Any](24L, Seq(Seq(-1, 0)), Seq(Seq(-1.1F, 0.2F)), Seq(Seq("r3")))))

  val sequenceExampleSchema = StructType(List(
    StructField("id",LongType),
    StructField("IntegerArrayOfArrayLabel", ArrayType(ArrayType(IntegerType))),
    StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
    StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType)))
  ))

  private def createDataFrameForExampleTFRecord() : DataFrame = {
    val rdd = spark.sparkContext.parallelize(exampleTestRows)
    spark.createDataFrame(rdd, exampleSchema)
  }

  private def createDataFrameForSequenceExampleTFRecords() : DataFrame = {
    val rdd = spark.sparkContext.parallelize(sequenceExampleTestRows)
    spark.createDataFrame(rdd, sequenceExampleSchema)
  }

  private def pathExists(pathStr: String): Boolean = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val outputPath = new Path(pathStr)
    val fs = outputPath.getFileSystem(hadoopConf)
    val qualifiedPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)
    fs.exists(qualifiedPath)
  }

  private def getFileCount(pathStr: String): Long = {
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val outputPath = new Path(pathStr)
    val fs = outputPath.getFileSystem(hadoopConf)
    val qualifiedPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)
    fs.getContentSummary(qualifiedPath).getFileCount
  }

  "Spark tfrecord IO" should {
    "Test tfrecord example Read/write " in {

      val path = s"$TF_SANDBOX_DIR/example.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecord").option("recordType", "Example").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecord").option("recordType", "Example").schema(exampleSchema).load(path)

      val actualDf = importedDf.select("id", "IntegerLabel", "LongLabel", "FloatLabel",
        "DoubleLabel", "DecimalLabel", "StrLabel", "BinaryLabel", "IntegerArrayLabel", "LongArrayLabel",
        "FloatArrayLabel", "DoubleArrayLabel", "DecimalArrayLabel", "StrArrayLabel", "BinaryArrayLabel").sort("StrLabel")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows.zip(actualRows).foreach { case (expected: Row, actual: Row) =>
        assert(expected ~== actual, exampleSchema)
      }
    }

    "Test tfrecord partition by id" in {
      val output = s"$TF_SANDBOX_DIR/example-partition-by-id.tfrecord"
      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecord").partitionBy("id").option("recordType", "Example").save(output)
      assert(pathExists(output))
      val partition1Path = s"$output/id=11"
      val partition2Path = s"$output/id=21"
      assert(pathExists(partition1Path))
      assert(pathExists(partition2Path))
      assert(getFileCount(partition1Path) == 2)
      assert(getFileCount(partition2Path) == 1)
    }

    "Test tfrecord read/write SequenceExample" in {

      val path = s"$TF_SANDBOX_DIR/sequenceExample.tfrecord"

      val df: DataFrame = createDataFrameForSequenceExampleTFRecords()
      df.write.format("tfrecord").option("recordType", "SequenceExample").save(path)

      val importedDf: DataFrame = spark.read.format("tfrecord").option("recordType", "SequenceExample").schema(sequenceExampleSchema).load(path)
      val actualDf = importedDf.select("id", "IntegerArrayOfArrayLabel", "FloatArrayOfArrayLabel", "StrArrayOfArrayLabel").sort("id")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      assert(expectedRows === actualRows)
    }

    "Test tfrecord write overwrite mode " in {

      val path = s"$TF_SANDBOX_DIR/example_overwrite.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecord").option("recordType", "Example").save(path)

      df.write.format("tfrecord").mode(SaveMode.Overwrite).option("recordType", "Example").save(path)

      //If schema is not provided. It will automatically infer schema
      val importedDf: DataFrame = spark.read.format("tfrecord").option("recordType", "Example").schema(exampleSchema).load(path)

      val actualDf = importedDf.select("id", "IntegerLabel", "LongLabel", "FloatLabel",
        "DoubleLabel", "DecimalLabel", "StrLabel", "BinaryLabel", "IntegerArrayLabel", "LongArrayLabel",
        "FloatArrayLabel", "DoubleArrayLabel", "DecimalArrayLabel", "StrArrayLabel", "BinaryArrayLabel").sort("StrLabel")

      val expectedRows = df.collect()
      val actualRows = actualDf.collect()

      expectedRows.zip(actualRows).foreach { case (expected: Row, actual: Row) =>
        assert(expected ~== actual, exampleSchema)
      }
    }

    "Test tfrecord write append mode" in {

      val path = s"$TF_SANDBOX_DIR/example_append.tfrecord"

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecord").option("recordType", "Example").save(path)
      df.write.format("tfrecord").mode(SaveMode.Append).option("recordType", "Example").save(path)
    }

    "Test tfrecord write ignore mode" in {

      val path = s"$TF_SANDBOX_DIR/example_ignore.tfrecord"

      val hadoopConf = spark.sparkContext.hadoopConfiguration
      val outputPath = new Path(path)
      val fs = outputPath.getFileSystem(hadoopConf)
      val qualifiedOutputPath = outputPath.makeQualified(fs.getUri, fs.getWorkingDirectory)

      val df: DataFrame = createDataFrameForExampleTFRecord()
      df.write.format("tfrecord").mode(SaveMode.Ignore).option("recordType", "Example").save(path)

      assert(fs.exists(qualifiedOutputPath))
      val timestamp1 = fs.getFileStatus(qualifiedOutputPath).getModificationTime

      df.write.format("tfrecord").mode(SaveMode.Ignore).option("recordType", "Example").save(path)

      val timestamp2 = fs.getFileStatus(qualifiedOutputPath).getModificationTime

      assert(timestamp1 == timestamp2, "SaveMode.Ignore Error: File was overwritten. Timestamps do not match")
    }
  }
}
