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
import org.tensorflow.example.{FeatureList, SequenceExample, Example, Feature}
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.runtime.universe._

object TensorFlowInferSchema {

  /**
   * Similar to the JSON schema inference.
   * [[org.apache.spark.sql.execution.datasources.json.InferSchema]]
   *     1. Infer type of each row
   *     2. Merge row types to find common type
   *     3. Replace any null types with string type
   */
  def apply[T : TypeTag](rdd: RDD[T]): StructType = {
    val startType: mutable.Map[String, DataType] = mutable.Map.empty[String, DataType]

    val rootTypes: mutable.Map[String, DataType] = typeOf[T] match {
      case t if t =:= typeOf[Example] => {
        rdd.asInstanceOf[RDD[Example]].aggregate(startType)(inferExampleRowType, mergeFieldTypes)
      }
      case t if t =:= typeOf[SequenceExample] => {
        rdd.asInstanceOf[RDD[SequenceExample]].aggregate(startType)(inferSequenceExampleRowType, mergeFieldTypes)
      }
      case _ =>  throw new IllegalArgumentException(s"Unsupported recordType: recordType can be Example or SequenceExample")
    }

    val columnsList = rootTypes.map {
      case (featureName, featureType) =>
        if (featureType == null) {
          StructField(featureName, StringType)
        }
        else {
          StructField(featureName, featureType)
        }
    }
    StructType(columnsList.toSeq)
  }

  private def inferSequenceExampleRowType(schemaSoFar: mutable.Map[String, DataType],
                                          next: SequenceExample): mutable.Map[String, DataType] = {
    val featureMap = next.getContext.getFeatureMap.asScala
    val updatedSchema = inferFeatureTypes(schemaSoFar, featureMap)

    val featureListMap = next.getFeatureLists.getFeatureListMap.asScala
    inferFeatureListTypes(updatedSchema, featureListMap)
  }

  private def inferExampleRowType(schemaSoFar: mutable.Map[String, DataType],
                                  next: Example): mutable.Map[String, DataType] = {
    val featureMap = next.getFeatures.getFeatureMap.asScala
    inferFeatureTypes(schemaSoFar, featureMap)
  }

  private def inferFeatureTypes(schemaSoFar: mutable.Map[String, DataType],
                                featureMap: mutable.Map[String, Feature]): mutable.Map[String, DataType] = {
    featureMap.foreach {
      case (featureName, feature) => {
        val currentType = inferField(feature)
        if (schemaSoFar.contains(featureName)) {
          val updatedType = findTightestCommonType(schemaSoFar(featureName), currentType)
          schemaSoFar(featureName) = updatedType.orNull
        }
        else {
          schemaSoFar += (featureName -> currentType)
        }
      }
    }
    schemaSoFar
  }

  def inferFeatureListTypes(schemaSoFar: mutable.Map[String, DataType],
                            featureListMap: mutable.Map[String, FeatureList]): mutable.Map[String, DataType] = {
    featureListMap.foreach {
      case (featureName, featureList) => {
        val featureType = featureList.getFeatureList.asScala.map(f => inferField(f))
          .reduceLeft((a, b) => findTightestCommonType(a, b).orNull)
        val currentType = featureType match {
          case ArrayType(_, _) => ArrayType(featureType)
          case _ => ArrayType(ArrayType(featureType))
        }
        if (schemaSoFar.contains(featureName)) {
          val updatedType = findTightestCommonType(schemaSoFar(featureName), currentType)
          schemaSoFar(featureName) = updatedType.orNull
        }
        else {
          schemaSoFar += (featureName -> currentType)
        }
      }
    }
    schemaSoFar
  }

  private def mergeFieldTypes(first: mutable.Map[String, DataType],
                              second: mutable.Map[String, DataType]): mutable.Map[String, DataType] = {
    //Merge two maps and do the comparison.
    val mutMap = collection.mutable.Map[String, DataType]((first.keySet ++ second.keySet)
      .map(key => (key, findTightestCommonType(first.getOrElse(key, null), second.getOrElse(key, null)).get))
      .toSeq: _*)
    mutMap
  }

  /**
   * Infer Feature datatype based on field number
   */
  private def inferField(feature: Feature): DataType = {
    feature.getKindCase.getNumber match {
      case Feature.BYTES_LIST_FIELD_NUMBER => {
        parseBytesList(feature)
      }
      case Feature.INT64_LIST_FIELD_NUMBER => {
        parseInt64List(feature)
      }
      case Feature.FLOAT_LIST_FIELD_NUMBER => {
        parseFloatList(feature)
      }
      case _ => throw new RuntimeException("unsupported type ...")
    }
  }

  private def parseBytesList(feature: Feature): DataType = {
    val length = feature.getBytesList.getValueCount

    if (length == 0) {
      null
    }
    else if (length > 1) {
      ArrayType(StringType)
    }
    else {
      StringType
    }
  }

  private def parseInt64List(feature: Feature): DataType = {
    val int64List = feature.getInt64List.getValueList.asScala.toArray
    val length = int64List.length

    if (length == 0) {
      null
    }
    else if (length > 1) {
      ArrayType(LongType)
    }
    else {
      LongType
    }
  }

  private def parseFloatList(feature: Feature): DataType = {
    val floatList = feature.getFloatList.getValueList.asScala.toArray
    val length = floatList.length
    if (length == 0) {
      null
    }
    else if (length > 1) {
      ArrayType(FloatType)
    }
    else {
      FloatType
    }
  }

  /**
   * Copied from internal Spark api
   * [[org.apache.spark.sql.catalyst.analysis.HiveTypeCoercion]]
   */
  private def getNumericPrecedence(dataType: DataType): Int = {
    dataType match {
      case LongType => 1
      case FloatType => 2
      case StringType => 3
      case ArrayType(LongType, _) => 4
      case ArrayType(FloatType, _) => 5
      case ArrayType(StringType, _) => 6
      case ArrayType(ArrayType(LongType, _), _) => 7
      case ArrayType(ArrayType(FloatType, _), _) => 8
      case ArrayType(ArrayType(StringType, _), _) => 9
      case _ => throw new RuntimeException("Unable to get the precedence for given datatype...")
    }
  }

  /**
   * Copied from internal Spark api
   * [[org.apache.spark.sql.catalyst.analysis.HiveTypeCoercion]]
   */
  private def findTightestCommonType(tt1: DataType, tt2: DataType) : Option[DataType] = {
    val currType = (tt1, tt2) match {
      case (t1, t2) if t1 == t2 => Some(t1)
      case (null, t2) => Some(t2)
      case (t1, null) => Some(t1)

      // Promote types based on numeric precedence
      case (t1, t2) =>
        val t1Precedence = getNumericPrecedence(t1)
        val t2Precedence = getNumericPrecedence(t2)
        val newType = if (t1Precedence > t2Precedence) t1 else t2
        Some(newType)
      case _ => None
    }
    currType
  }
}

