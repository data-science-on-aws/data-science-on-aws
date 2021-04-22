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

import org.apache.spark.sql.catalyst.InternalRow
import org.tensorflow.example._
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.{Matchers, WordSpec}

import scala.collection.JavaConverters._
import TestingUtils._

class TFRecordSerializerTest extends WordSpec with Matchers {

  private def createArray(values: Any*): ArrayData = new GenericArrayData(values.toArray)

  "Serialize spark internalRow to tfrecord" should {

    "Serialize  decimal internalRow to tfrecord example" in {
      val schemaStructType = StructType(Array(
        StructField("DecimalLabel", DataTypes.createDecimalType()),
        StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType()))
      ))
      val serializer = new TFRecordSerializer(schemaStructType)

      val decimalArray = Array(Decimal(4.0), Decimal(8.0))
      val rowArray = Array[Any](Decimal(6.5), ArrayData.toArrayData(decimalArray))
      val internalRow = InternalRow.fromSeq(rowArray)

      //Encode Sql InternalRow to TensorFlow example
      val example = serializer.serializeExample(internalRow)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == rowArray.length)

      assert(featureMap("DecimalLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalLabel").getFloatList.getValue(0) == 6.5F)

      assert(featureMap("DecimalArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== decimalArray.map(_.toFloat))
    }

    "Serialize complex internalRow to tfrecord example" in {
      val schemaStructType = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("LongLabel", LongType),
        StructField("FloatLabel", FloatType),
        StructField("DoubleLabel", DoubleType),
        StructField("DecimalLabel", DataTypes.createDecimalType()),
        StructField("DoubleArrayLabel", ArrayType(DoubleType)),
        StructField("DecimalArrayLabel", ArrayType(DataTypes.createDecimalType())),
        StructField("StrLabel", StringType),
        StructField("StrArrayLabel", ArrayType(StringType)),
        StructField("BinaryLabel", BinaryType),
        StructField("BinaryArrayLabel", ArrayType(BinaryType))
      ))
      val doubleArray = Array(1.1, 111.1, 11111.1)
      val decimalArray = Array(Decimal(4.0), Decimal(8.0))
      val byteArray = Array[Byte](0xde.toByte, 0xad.toByte, 0xbe.toByte, 0xef.toByte)
      val byteArray1 = Array[Byte](-128, 23, 127)

      val rowArray = Array[Any](1, 23L, 10.0F, 14.0, Decimal(6.5),
        ArrayData.toArrayData(doubleArray),
        ArrayData.toArrayData(decimalArray),
        UTF8String.fromString("r1"),
        ArrayData.toArrayData(Array(UTF8String.fromString("r2"), UTF8String.fromString("r3"))),
        byteArray,
        ArrayData.toArrayData(Array(byteArray, byteArray1))
      )

      val internalRow = InternalRow.fromSeq(rowArray)

      val serializer = new TFRecordSerializer(schemaStructType)
      val example = serializer.serializeExample(internalRow)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = example.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == rowArray.length)

      assert(featureMap("IntegerLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 1)

      assert(featureMap("LongLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("LongLabel").getInt64List.getValue(0).toInt == 23)

      assert(featureMap("FloatLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatLabel").getFloatList.getValue(0) == 10.0F)

      assert(featureMap("DoubleLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleLabel").getFloatList.getValue(0) == 14.0F)

      assert(featureMap("DecimalLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalLabel").getFloatList.getValue(0) == 6.5F)

      assert(featureMap("DoubleArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DoubleArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== doubleArray.map(_.toFloat))

      assert(featureMap("DecimalArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("DecimalArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== decimalArray.map(_.toFloat))

      assert(featureMap("StrLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StrLabel").getBytesList.getValue(0).toStringUtf8 == "r1")

      assert(featureMap("StrArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StrArrayLabel").getBytesList.getValueList.asScala.map(_.toStringUtf8) === Seq("r2", "r3"))

      assert(featureMap("BinaryLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("BinaryLabel").getBytesList.getValue(0).toByteArray.deep == byteArray.deep)

      assert(featureMap("BinaryArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      val binaryArrayValue = featureMap("BinaryArrayLabel").getBytesList.getValueList.asScala.map((byteArray) => byteArray.asScala.toArray.map(_.toByte))
      assert(binaryArrayValue.toArray.deep == Array(byteArray, byteArray1).deep)
    }

    "Serialize internalRow to tfrecord sequenceExample" in {

      val schemaStructType = StructType(Array(
        StructField("IntegerLabel", IntegerType),
        StructField("StringArrayLabel", ArrayType(StringType)),
        StructField("LongArrayOfArrayLabel", ArrayType(ArrayType(LongType))),
        StructField("FloatArrayOfArrayLabel", ArrayType(ArrayType(FloatType))) ,
        StructField("DoubleArrayOfArrayLabel", ArrayType(ArrayType(DoubleType))),
        StructField("DecimalArrayOfArrayLabel", ArrayType(ArrayType(DataTypes.createDecimalType()))),
        StructField("StringArrayOfArrayLabel", ArrayType(ArrayType(StringType))),
        StructField("BinaryArrayOfArrayLabel", ArrayType(ArrayType(BinaryType)))
      ))

      val stringList = Array(UTF8String.fromString("r1"), UTF8String.fromString("r2"), UTF8String.fromString(("r3")))
      val longListOfLists = Array(Array(3L, 5L), Array(-8L, 0L))
      val floatListOfLists = Array(Array(1.5F, -6.5F), Array(-8.2F, 0F))
      val doubleListOfLists = Array(Array(3.0), Array(6.0, 9.0))
      val decimalListOfLists = Array(Array(Decimal(2.0), Decimal(4.0)), Array(Decimal(6.0)))
      val stringListOfLists = Array(Array(UTF8String.fromString("r1")),
        Array(UTF8String.fromString("r2"), UTF8String.fromString("r3")),
        Array(UTF8String.fromString("r4")))
      val binaryListOfLists = stringListOfLists.map(stringList => stringList.map(_.getBytes))

      val rowArray = Array[Any](10,
        createArray(UTF8String.fromString("r1"), UTF8String.fromString("r2"), UTF8String.fromString(("r3"))),
        createArray(
          createArray(3L, 5L),
          createArray(-8L, 0L)
        ),
        createArray(
          createArray(1.5F, -6.5F),
          createArray(-8.2F, 0F)
        ),
        createArray(
          createArray(3.0),
          createArray(6.0, 9.0)
        ),
        createArray(
          createArray(Decimal(2.0), Decimal(4.0)),
          createArray(Decimal(6.0))
        ),
        createArray(
          createArray(UTF8String.fromString("r1")),
          createArray(UTF8String.fromString("r2"), UTF8String.fromString("r3")),
          createArray(UTF8String.fromString("r4"))
        ),
        createArray(createArray("r1".getBytes()),
          createArray("r2".getBytes(), "r3".getBytes),
          createArray("r4".getBytes())
        )
      )

      val internalRow = InternalRow.fromSeq(rowArray)

      val serializer = new TFRecordSerializer(schemaStructType)
      val tfexample = serializer.serializeSequenceExample(internalRow)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = tfexample.getContext.getFeatureMap.asScala
      val featureListMap = tfexample.getFeatureLists.getFeatureListMap.asScala

      assert(featureMap.size == 2)
      assert(featureMap("IntegerLabel").getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER)
      assert(featureMap("IntegerLabel").getInt64List.getValue(0).toInt == 10)
      assert(featureMap("StringArrayLabel").getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER)
      assert(featureMap("StringArrayLabel").getBytesList.getValueList.asScala.map(x => UTF8String.fromString(x.toStringUtf8)) === stringList)

      assert(featureListMap.size == 6)
      assert(featureListMap("LongArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getInt64List.getValueList.asScala.toSeq) === longListOfLists)

      assert(featureListMap("FloatArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toFloat).toSeq) ~== floatListOfLists.map{arr => arr.toSeq}.toSeq)
      assert(featureListMap("DoubleArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(_.toDouble).toSeq) ~== doubleListOfLists.map{arr => arr.toSeq}.toSeq)

      assert(featureListMap("DecimalArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getFloatList.getValueList.asScala.map(x => Decimal(x.toDouble)).toSeq) ~== decimalListOfLists.map{arr => arr.toSeq}.toSeq)

      assert(featureListMap("StringArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(x => UTF8String.fromString(x.toStringUtf8)).toSeq) === stringListOfLists)

      assert(featureListMap("BinaryArrayOfArrayLabel").getFeatureList.asScala.map(
        _.getBytesList.getValueList.asScala.map(byteList => byteList.asScala.toSeq)) === binaryListOfLists.map(_.map(_.toSeq)))
    }

    "Throw an exception for non-nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NonNullLabel", ArrayType(FloatType), nullable = false)
      ))

      val internalRow = InternalRow.fromSeq(Array[Any](null))

      val serializer = new TFRecordSerializer(schemaStructType)

      intercept[NullPointerException]{
        serializer.serializeExample(internalRow)
      }

      intercept[NullPointerException]{
        serializer.serializeSequenceExample(internalRow)
      }
    }

    "Omit null fields from Example for nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NullLabel", ArrayType(FloatType), nullable = true),
        StructField("FloatArrayLabel", ArrayType(FloatType))
      ))

      val floatArray = Array(2.5F, 5.0F)
      val internalRow = InternalRow.fromSeq(
        Array[Any](null, createArray(2.5F, 5.0F))
      )

      val serializer = new TFRecordSerializer(schemaStructType)
      val tfexample = serializer.serializeExample(internalRow)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = tfexample.getFeatures.getFeatureMap.asScala
      assert(featureMap.size == 1)
      assert(featureMap("FloatArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== floatArray.toSeq)
    }

    "Omit null fields from SequenceExample for nullable data types" in {
      val schemaStructType = StructType(Array(
        StructField("NullLabel", ArrayType(FloatType), nullable = true),
        StructField("FloatArrayLabel", ArrayType(FloatType))
      ))

      val floatArray = Array(2.5F, 5.0F)
      val internalRow = InternalRow.fromSeq(
        Array[Any](null, createArray(2.5F, 5.0F)))

      val serializer = new TFRecordSerializer(schemaStructType)
      val tfSeqExample = serializer.serializeSequenceExample(internalRow)

      //Verify each Datatype converted to TensorFlow datatypes
      val featureMap = tfSeqExample.getContext.getFeatureMap.asScala
      val featureListMap = tfSeqExample.getFeatureLists.getFeatureListMap.asScala
      assert(featureMap.size == 1)
      assert(featureListMap.isEmpty)
      assert(featureMap("FloatArrayLabel").getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER)
      assert(featureMap("FloatArrayLabel").getFloatList.getValueList.asScala.toSeq.map(_.toFloat) ~== floatArray.toSeq)
    }

    "Throw an exception for unsupported data types" in {

      val schemaStructType = StructType(Array(
        StructField("TimestampLabel", TimestampType)
      ))

      intercept[RuntimeException]{
        new TFRecordSerializer(schemaStructType)
      }
    }

    val schema = StructType(Array(
      StructField("bytesLabel", BinaryType))
    )
    val serializer = new TFRecordSerializer(schema)

    "Test Int64ListFeature" in {
      val longFeature = serializer.Int64ListFeature(Seq(10L))
      val longListFeature = serializer.Int64ListFeature(Seq(3L,5L,6L))

      assert(longFeature.getInt64List.getValueList.asScala.toSeq === Seq(10L))
      assert(longListFeature.getInt64List.getValueList.asScala.toSeq === Seq(3L, 5L, 6L))
    }

    "Test floatListFeature" in {
      val floatFeature = serializer.floatListFeature(Seq(10.1F))
      val floatListFeature = serializer.floatListFeature(Seq(3.1F,5.1F,6.1F))

      assert(floatFeature.getFloatList.getValueList.asScala.toSeq === Seq(10.1F))
      assert(floatListFeature.getFloatList.getValueList.asScala.toSeq === Seq(3.1F,5.1F,6.1F))
    }

    "Test bytesListFeature" in {
      val bytesFeature = serializer.bytesListFeature(Seq(Array(0xff.toByte, 0xd8.toByte)))
      val bytesListFeature = serializer.bytesListFeature(Seq(
        Array(0xff.toByte, 0xd8.toByte),
        Array(0xff.toByte, 0xd9.toByte)))

      assert(bytesFeature.getBytesList.getValueList.asScala.map(_.toByteArray.deep) ===
        Seq(Array(0xff.toByte, 0xd8.toByte).deep))
      assert(bytesListFeature.getBytesList.getValueList.asScala.map(_.toByteArray.deep) ===
        Seq(Array(0xff.toByte, 0xd8.toByte).deep, Array(0xff.toByte, 0xd9.toByte).deep))
    }
  }
}
