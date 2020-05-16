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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.tensorflow.example._
import com.google.protobuf.ByteString

class InferSchemaSuite extends SharedSparkSessionSuite {

  val longFeature = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(Int.MaxValue + 10L)).build()
  val floatFeature = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(10.0F).build()).build()
  val strFeature = Feature.newBuilder().setBytesList(
    BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes))).build()

  val longList = Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(-2L).addValue(20L).build()).build()
  val floatList = Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(2.5F).addValue(7F).build()).build()
  val strList = Feature.newBuilder().setBytesList(BytesList.newBuilder().addValue(ByteString.copyFrom("r1".getBytes))
    .addValue(ByteString.copyFrom("r2".getBytes)).build()).build()

  "InferSchema" should {

    "Infer schema from Example records" in {
      //Build example1
      val features1 = Features.newBuilder()
        .putFeature("LongFeature", longFeature)
        .putFeature("FloatFeature", floatFeature)
        .putFeature("StrFeature", strFeature)
        .putFeature("LongList", longFeature)
        .putFeature("FloatList", floatFeature)
        .putFeature("StrList", strFeature)
        .putFeature("MixedTypeList", longList)
        .build()
      val example1 = Example.newBuilder()
        .setFeatures(features1)
        .build()

      //Example2 contains subset of features in example1 to test behavior with missing features
      val features2 = Features.newBuilder()
        .putFeature("StrFeature", strFeature)
        .putFeature("LongList", longList)
        .putFeature("FloatList", floatList)
        .putFeature("StrList", strList)
        .putFeature("MixedTypeList", floatList)
        .build()
      val example2 = Example.newBuilder()
        .setFeatures(features2)
        .build()

      val exampleRdd: RDD[Example] = spark.sparkContext.parallelize(List(example1, example2))
      val inferredSchema = TensorFlowInferSchema(exampleRdd)

      //Verify each TensorFlow Datatype is inferred as one of our Datatype
      assert(inferredSchema.fields.length == 7)
      val schemaMap = inferredSchema.map(f => (f.name, f.dataType)).toMap
      assert(schemaMap("LongFeature") === LongType)
      assert(schemaMap("FloatFeature") === FloatType)
      assert(schemaMap("StrFeature") === StringType)
      assert(schemaMap("LongList") ===  ArrayType(LongType))
      assert(schemaMap("FloatList") ===  ArrayType(FloatType))
      assert(schemaMap("StrList") === ArrayType(StringType))
      assert(schemaMap("MixedTypeList") === ArrayType(FloatType))
    }

    "Infer schema from SequenceExample records" in {

      //Build sequence example1
      val features1 = Features.newBuilder()
        .putFeature("FloatFeature", floatFeature)

      val longFeatureList1 = FeatureList.newBuilder().addFeature(longFeature).addFeature(longList).build()
      val floatFeatureList1 = FeatureList.newBuilder().addFeature(floatFeature).addFeature(floatList).build()
      val strFeatureList1 = FeatureList.newBuilder().addFeature(strFeature).build()
      val mixedFeatureList1 = FeatureList.newBuilder().addFeature(floatFeature).addFeature(strList).build()

      val featureLists1 = FeatureLists.newBuilder()
        .putFeatureList("LongListOfLists", longFeatureList1)
        .putFeatureList("FloatListOfLists", floatFeatureList1)
        .putFeatureList("StringListOfLists", strFeatureList1)
        .putFeatureList("MixedListOfLists", mixedFeatureList1)
        .build()

      val seqExample1 = SequenceExample.newBuilder()
        .setContext(features1)
        .setFeatureLists(featureLists1)
        .build()

      //SequenceExample2 contains subset of features in example1 to test behavior with missing features
      val longFeatureList2 = FeatureList.newBuilder().addFeature(longList).build()
      val floatFeatureList2 = FeatureList.newBuilder().addFeature(floatFeature).build()
      val strFeatureList2 = FeatureList.newBuilder().addFeature(strFeature).build() //test both string lists of length=1
      val mixedFeatureList2 = FeatureList.newBuilder().addFeature(longFeature).addFeature(strFeature).build()

      val featureLists2 = FeatureLists.newBuilder()
        .putFeatureList("LongListOfLists", longFeatureList2)
        .putFeatureList("FloatListOfLists", floatFeatureList2)
        .putFeatureList("StringListOfLists", strFeatureList2)
        .putFeatureList("MixedListOfLists", mixedFeatureList2)
        .build()

      val seqExample2 = SequenceExample.newBuilder()
        .setFeatureLists(featureLists2)
        .build()

      val seqExampleRdd: RDD[SequenceExample] = spark.sparkContext.parallelize(List(seqExample1, seqExample2))
      val inferredSchema = TensorFlowInferSchema(seqExampleRdd)

      //Verify each TensorFlow Datatype is inferred as one of our Datatype
      assert(inferredSchema.fields.length == 5)
      val schemaMap = inferredSchema.map(f => (f.name, f.dataType)).toMap
      assert(schemaMap("FloatFeature") === FloatType)
      assert(schemaMap("LongListOfLists") === ArrayType(ArrayType(LongType)))
      assert(schemaMap("FloatListOfLists") === ArrayType(ArrayType(FloatType)))
      assert(schemaMap("StringListOfLists") === ArrayType(ArrayType(StringType)))
      assert(schemaMap("MixedListOfLists") === ArrayType(ArrayType(StringType)))
    }
  }

  "Throw an exception for unsupported record types" in {
    intercept[Exception]  {
      val rdd: RDD[Long] = spark.sparkContext.parallelize(List(5L, 6L))
      TensorFlowInferSchema(rdd)
    }

  }
}

