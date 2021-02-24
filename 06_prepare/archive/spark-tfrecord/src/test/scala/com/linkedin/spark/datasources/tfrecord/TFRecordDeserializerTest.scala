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

import com.google.protobuf.ByteString
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.{Matchers, WordSpec}
import org.tensorflow.example._
import TestingUtils._


class TFRecordDeserializerTest extends WordSpec with Matchers {
  val intFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(1)).build()
  val longFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(23L)).build()
  val floatFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(10.0F)).build()
  val doubleFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(14.0F)).build()
  val decimalFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(2.5F)).build()
  val longArrFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(-2L).addValue(7L).build()).build()
  val doubleArrFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(1F).addValue(2F).build()).build()
  val decimalArrFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(3F).addValue(5F).build()).build()
  val strFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes)).build()).build()
  val strListFeature =Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r2".getBytes))
    .addValue(ByteString.copyFrom("r3".getBytes)).build()).build()
  val binaryFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r4".getBytes))).build()
  val binaryListFeature = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r5".getBytes))
    .addValue(ByteString.copyFrom("r6".getBytes)).build()).build()

  private def createArray(values: Any*): ArrayData = new GenericArrayData(values.toArray)

  "Deserialize tfrecord to spark internalRow" should {

    "Serialize tfrecord example to spark internalRow" in {
      val schema = StructType(List(
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DecimalLabel", DataTypes.createDecimalType()),
        StructField("LongArrayLabel", ArrayType(LongType)),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType())),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType)),
        StructField("BinaryTypeLabel", BinaryType),
        StructField("BinaryTypeArrayLabel", ArrayType(BinaryType))
      ))

      val expectedInternalRow = InternalRow.fromSeq(
        Array[Any](1, 23L, 10.0F, 14.0, Decimal(2.5d),
          createArray(-2L,7L),
          createArray(1.0, 2.0),
          createArray(Decimal(3.0), Decimal(5.0)),
          UTF8String.fromString("r1"),
          createArray(UTF8String.fromString("r2"), UTF8String.fromString("r3")),
          "r4".getBytes,
          createArray("r5".getBytes(), "r6".getBytes())
        )
      )

      //Build example
      val features = Features.newBuilder()
        .putFeature("IntegerLabel", intFeature)
        .putFeature("LongLabel", longFeature)
        .putFeature("FloatLabel", floatFeature)
        .putFeature("DoubleLabel", doubleFeature)
        .putFeature("DecimalLabel", decimalFeature)
        .putFeature("LongArrayLabel", longArrFeature)
        .putFeature("DoubleArrayLabel", doubleArrFeature)
        .putFeature("DecimalArrayLabel", decimalArrFeature)
        .putFeature("StrLabel", strFeature)
        .putFeature("StrArrayLabel", strListFeature)
        .putFeature("BinaryTypeLabel", binaryFeature)
        .putFeature("BinaryTypeArrayLabel", binaryListFeature)
        .build()
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()
      val deserializer = new TFRecordDeserializer(schema)
      val actualInternalRow = deserializer.deserializeExample(example)

      assert(actualInternalRow ~== (expectedInternalRow,schema))
    }

    "Serialize spark internalRow to tfrecord sequenceExample" in {

      val schema = StructType(List(
        StructField("FloatLabel", FloatType),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))),
        StructField("DecimalArrayOfArrayLabel", ArrayType(ArrayType(DataTypes.createDecimalType()))),
        StructField("StrArrayOfArrayLabel", ArrayType(ArrayType(StringType))),
        StructField("ByteArrayOfArrayLabel", ArrayType(ArrayType(BinaryType)))
      ))

      val expectedInternalRow = InternalRow.fromSeq(
        Array[Any](10.0F,
          createArray(createArray(-2L, 7L)),
          createArray(createArray(10.0F), createArray(1.0F, 2.0F)),
          createArray(createArray(Decimal(3.0), Decimal(5.0))),
          createArray(createArray(UTF8String.fromString("r2"), UTF8String.fromString("r3")),
            createArray(UTF8String.fromString("r1"))),
          createArray(createArray("r5".getBytes, "r6".getBytes), createArray("r4".getBytes))
        )
      )

      //Build sequence example
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val floatFeatureList = FeatureList.newBuilder().addFeature(floatFeature).addFeature(doubleArrFeature).build()
      val decimalFeatureList = FeatureList.newBuilder().addFeature(decimalArrFeature).build()
      val stringFeatureList = FeatureList.newBuilder().addFeature(strListFeature).addFeature(strFeature).build()
      val binaryFeatureList = FeatureList.newBuilder().addFeature(binaryListFeature).addFeature(binaryFeature).build()


      val features = Features.newBuilder()
        .putFeature("FloatLabel", floatFeature)

      val featureLists = FeatureLists.newBuilder()
        .putFeatureList("LongArrayOfArrayLabel", int64FeatureList)
        .putFeatureList("FloatArrayOfArrayLabel", floatFeatureList)
        .putFeatureList("DecimalArrayOfArrayLabel", decimalFeatureList)
        .putFeatureList("StrArrayOfArrayLabel", stringFeatureList)
        .putFeatureList("ByteArrayOfArrayLabel", binaryFeatureList)
        .build()

      val seqExample = SequenceExample.newBuilder()
        .setContext(features)
        .setFeatureLists(featureLists)
        .build()

      val deserializer = new TFRecordDeserializer(schema)
      val actualInternalRow = deserializer.deserializeSequenceExample(seqExample)
      assert(actualInternalRow ~== (expectedInternalRow, schema))
    }

    "Throw an exception for unsupported data types" in {

      val features = Features.newBuilder().putFeature("MapLabel1", intFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("MapLabel2", int64FeatureList)

      intercept[RuntimeException] {
        val example = Example.newBuilder()
          .setFeatures(features)
          .build()
        val schema = StructType(List(StructField("MapLabel1", TimestampType)))
        val deserializer = new TFRecordDeserializer(schema)
        deserializer.deserializeExample(example)
      }

      intercept[RuntimeException] {
        val seqExample = SequenceExample.newBuilder()
          .setContext(features)
          .setFeatureLists(featureLists)
          .build()
        val schema = StructType(List(StructField("MapLabel2", TimestampType)))
        val deserializer = new TFRecordDeserializer(schema)
        deserializer.deserializeSequenceExample(seqExample)
      }
    }

    "Throw an exception for non-nullable data types" in {
      val features = Features.newBuilder().putFeature("FloatLabel", floatFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("LongArrayOfArrayLabel", int64FeatureList)

      intercept[NullPointerException] {
        val example = Example.newBuilder()
          .setFeatures(features)
          .build()
        val schema = StructType(List(StructField("MissingLabel", FloatType, nullable = false)))
        val deserializer = new TFRecordDeserializer(schema)
        deserializer.deserializeExample(example)
      }

      intercept[NullPointerException] {
        val seqExample = SequenceExample.newBuilder()
          .setContext(features)
          .setFeatureLists(featureLists)
          .build()
        val schema = StructType(List(StructField("MissingLabel", ArrayType(ArrayType(LongType)), nullable = false)))
        val deserializer = new TFRecordDeserializer(schema)
        deserializer.deserializeSequenceExample(seqExample)
      }
    }


    "Return null fields for nullable data types" in {
      val features = Features.newBuilder().putFeature("FloatLabel", floatFeature)
      val int64FeatureList = FeatureList.newBuilder().addFeature(longArrFeature).build()
      val featureLists = FeatureLists.newBuilder().putFeatureList("LongArrayOfArrayLabel", int64FeatureList)

      // Deserialize Example
      val schema1 = StructType(List(
        StructField("FloatLabel", FloatType),
        StructField("MissingLabel", FloatType, nullable = true))
      )
      val expectedInternalRow1 = InternalRow.fromSeq(
        Array[Any](10.0F, null)
      )
      val example = Example.newBuilder()
        .setFeatures(features)
        .build()
      val deserializer1 = new TFRecordDeserializer(schema1)
      val actualInternalRow1 = deserializer1.deserializeExample(example)
      assert(actualInternalRow1 ~== (expectedInternalRow1, schema1))

      // Deserialize SequenceExample
      val schema2 = StructType(List(
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("MissingLabel", ArrayType(ArrayType(LongType)), nullable = true))
      )
      val expectedInternalRow2 = InternalRow.fromSeq(
        Array[Any](
          createArray(createArray(-2L, 7L)), null)
      )
      val seqExample = SequenceExample.newBuilder()
        .setContext(features)
        .setFeatureLists(featureLists)
        .build()
      val deserializer2 = new TFRecordDeserializer(schema2)
      val actualInternalRow2 = deserializer2.deserializeSequenceExample(seqExample)
      assert(actualInternalRow2 ~== (expectedInternalRow2, schema2))

    }

    val schema = StructType(Array(
      StructField("LongLabel", LongType))
    )
    val deserializer = new TFRecordDeserializer(schema)

    "Test Int64ListFeature2SeqLong" in {
      val int64List = Int64List.newBuilder().addValue(5L).build()
      val intFeature = Feature.newBuilder().setInt64List(int64List).build()
      assert(deserializer.Int64ListFeature2SeqLong(intFeature).head === 5L)

      // Throw exception if type doesn't match
      intercept[RuntimeException] {
        val floatList = FloatList.newBuilder().addValue(2.5F).build()
        val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
        deserializer.Int64ListFeature2SeqLong(floatFeature)
      }
    }

    "Test floatListFeature2SeqFloat" in {
      val floatList = FloatList.newBuilder().addValue(2.5F).build()
      val floatFeature = Feature.newBuilder().setFloatList(floatList).build()
      assert(deserializer.floatListFeature2SeqFloat(floatFeature).head === 2.5F)

      // Throw exception if type doesn't match
      intercept[RuntimeException] {
        val int64List = Int64List.newBuilder().addValue(5L).build()
        val intFeature = Feature.newBuilder().setInt64List(int64List).build()
        deserializer.floatListFeature2SeqFloat(intFeature)
      }
    }

    "Test bytesListFeature2SeqArrayByte" in {
      val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("str-input".getBytes)).build()
      val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
      assert(deserializer.bytesListFeature2SeqArrayByte(bytesFeature).head === "str-input".getBytes.deep)

      // Throw exception if type doesn't match
      intercept[RuntimeException] {
        val int64List = Int64List.newBuilder().addValue(5L).build()
        val intFeature = Feature.newBuilder().setInt64List(int64List).build()
        deserializer.bytesListFeature2SeqArrayByte(intFeature)
      }
    }

    "Test bytesListFeature2SeqString" in {
      val bytesList = BytesList.newBuilder().addValue(ByteString.copyFrom("alice".getBytes))
        .addValue(ByteString.copyFrom("bob".getBytes)).build()
      val bytesFeature = Feature.newBuilder().setBytesList(bytesList).build()
      assert(deserializer.bytesListFeature2SeqString(bytesFeature) === Seq("alice", "bob"))

      // Throw exception if type doesn't match
      intercept[RuntimeException] {
        val int64List = Int64List.newBuilder().addValue(5L).build()
        val intFeature = Feature.newBuilder().setInt64List(int64List).build()
        deserializer.bytesListFeature2SeqArrayByte(intFeature)
      }
    }
  }
}
