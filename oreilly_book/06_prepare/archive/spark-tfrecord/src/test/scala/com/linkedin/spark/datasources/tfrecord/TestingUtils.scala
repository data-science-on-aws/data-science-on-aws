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

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericRowWithSchema, SpecializedGetters}
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.scalatest.Matchers

object TestingUtils extends Matchers {

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class FloatArrayWithAlmostEquals(val left: Seq[Float]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Float], epsilon : Float = 1E-6F): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a === (b +- epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class DoubleArrayWithAlmostEquals(val left: Seq[Double]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Double], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a === (b +- epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two decimal values using absolute tolerance.
   */
  implicit class DecimalArrayWithAlmostEquals(val left: Seq[Decimal]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Decimal], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a.toDouble === (b.toDouble +- epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class FloatMatrixWithAlmostEquals(val left: Seq[Seq[Float]]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Seq[Float]], epsilon : Float = 1E-6F): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a ~== (b, epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two double values using absolute tolerance.
   */
  implicit class DoubleMatrixWithAlmostEquals(val left: Seq[Seq[Double]]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Seq[Double]], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a ~== (b, epsilon) }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two decimal values using absolute tolerance.
   */
  implicit class DecimalMatrixWithAlmostEquals(val left: Seq[Seq[Decimal]]) {

    /**
     * When the difference of two values are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Seq[Seq[Decimal]], epsilon : Double = 1E-6): Boolean = {
      if (left.size === right.size) {
        (left zip right) forall { case (a, b) => a ~== (b, epsilon) }
      }
      else false
    }
  }

  /**
    * Implicit class for comparing two internalRows using absolute tolerance.
    */
  implicit class InternalRowWithAlmostEquals(val left: InternalRow) {

    private type valueCompare = (SpecializedGetters, SpecializedGetters, Int) => Boolean
    private def newValueCompare(
      dataType: DataType,
      epsilon : Float = 1E-6F): valueCompare = dataType match {
      case NullType => (left, right, ordinal) =>
        left.get(ordinal, null) == right.get(ordinal, null)

      case IntegerType => (left, right, ordinal) =>
        left.getInt(ordinal) === right.getInt(ordinal)

      case LongType => (left, right, ordinal) =>
        left.getLong(ordinal) === right.getLong(ordinal)

      case FloatType => (left, right, ordinal) =>
        left.getFloat(ordinal) === (right.getFloat(ordinal) +- epsilon)

      case DoubleType => (left, right, ordinal) =>
        left.getDouble(ordinal) === (right.getDouble(ordinal) +- epsilon)

      case DecimalType() => (left, right, ordinal) =>
        left.getDecimal(ordinal, DecimalType.USER_DEFAULT.precision, DecimalType.USER_DEFAULT.scale).toDouble ===
          (right.getDecimal(ordinal, DecimalType.USER_DEFAULT.precision, DecimalType.USER_DEFAULT.scale).toDouble
            +- epsilon)

      case StringType => (left, right, ordinal) =>
        left.getUTF8String(ordinal).getBytes === right.getUTF8String(ordinal).getBytes

      case BinaryType => (left, right, ordinal) =>
        left.getBinary(ordinal) === right.getBinary(ordinal)

      case ArrayType(elementType, _) => (left, right, ordinal) =>
        if (left.get(ordinal, null) == null || right.get(ordinal, null) == null ){
          left.get(ordinal, null) == right.get(ordinal, null)
        } else {
            val leftArray = left.getArray(ordinal)
            val rightArray = right.getArray(ordinal)
            if (leftArray.numElements == rightArray.numElements) {
              val len = leftArray.numElements()
              val elementValueCompare = newValueCompare(elementType)
              var result = true
              var idx: Int = 0
              while (idx < len && result) {
                result = elementValueCompare(leftArray, rightArray, idx)
                idx += 1
              }
              result
            } else false
          }
      case _ => throw new RuntimeException(s"Cannot convert field to unsupported data type ${dataType}")
    }

    /**
      * When all fields in row with given schema are equal or are within eps, returns true; otherwise, returns false.
      */
    def ~==(right: InternalRow, schema: StructType, epsilon : Float = 1E-6F): Boolean = {
      if (schema != null && schema.fields.size == left.numFields && schema.fields.size == right.numFields) {
        schema.fields.map(_.dataType).zipWithIndex.forall {
          case (dataType, idx) =>
          val valueCompare = newValueCompare(dataType)
          valueCompare(left, right, idx)
        }
      }
      else false
    }
  }

  /**
   * Implicit class for comparing two rows using absolute tolerance.
   */
  implicit class RowWithAlmostEquals(val left: Row) {

    /**
     * When all fields in row with given schema are equal or are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Row, schema: StructType): Boolean = {
      if (schema != null && schema.fields.size == left.size && schema.fields.size == right.size) {
        val leftRowWithSchema = new GenericRowWithSchema(left.toSeq.toArray, schema)
        val rightRowWithSchema = new GenericRowWithSchema(right.toSeq.toArray, schema)
        leftRowWithSchema ~== rightRowWithSchema
      }
      else false
    }

    /**
     * When all fields in row are equal or are within eps, returns true; otherwise, returns false.
     */
    def ~==(right: Row, epsilon : Float = 1E-6F): Boolean = {
      if (left.size === right.size) {
        val leftDataTypes = left.schema.fields.map(_.dataType)
        val rightDataTypes = right.schema.fields.map(_.dataType)

        (leftDataTypes zip rightDataTypes).zipWithIndex.forall {
          case (x, i) if left.get(i) == null || right.get(i) == null =>
            left.get(i) == null && right.get(i) == null

          case ((FloatType, FloatType), i) =>
            left.getFloat(i) === (right.getFloat(i) +- epsilon)

          case ((DoubleType, DoubleType), i) =>
            left.getDouble(i) === (right.getDouble(i) +- epsilon)

          case ((BinaryType, BinaryType), i) =>
            left.getAs[Array[Byte]](i).toSeq === right.getAs[Array[Byte]](i).toSeq

          case ((ArrayType(FloatType,_), ArrayType(FloatType,_)), i) =>
            val leftArray = ArrayData.toArrayData(left.get(i)).toFloatArray().toSeq
            val rightArray = ArrayData.toArrayData(right.get(i)).toFloatArray().toSeq
            leftArray ~== (rightArray, epsilon)

          case ((ArrayType(DoubleType,_), ArrayType(DoubleType,_)), i) =>
            val leftArray = ArrayData.toArrayData(left.get(i)).toDoubleArray().toSeq
            val rightArray = ArrayData.toArrayData(right.get(i)).toDoubleArray().toSeq
            leftArray ~== (rightArray, epsilon)

          case ((ArrayType(BinaryType,_), ArrayType(BinaryType,_)), i) =>
            val leftArray = ArrayData.toArrayData(left.get(i)).toArray[Array[Byte]](BinaryType).map(_.toSeq).toSeq
            val rightArray = ArrayData.toArrayData(right.get(i)).toArray[Array[Byte]](BinaryType).map(_.toSeq).toSeq
            leftArray === rightArray

          case ((ArrayType(ArrayType(FloatType,_),_), ArrayType(ArrayType(FloatType,_),_)), i) =>
            val leftArrays = ArrayData.toArrayData(left.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toFloatArray().toSeq
            }
            val rightArrays = ArrayData.toArrayData(right.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toFloatArray().toSeq
            }
            leftArrays ~== (rightArrays, epsilon)

          case ((ArrayType(ArrayType(DoubleType,_),_), ArrayType(ArrayType(DoubleType,_),_)), i) =>
            val leftArrays = ArrayData.toArrayData(left.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toDoubleArray().toSeq
            }
            val rightArrays = ArrayData.toArrayData(right.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toDoubleArray().toSeq
            }
            leftArrays ~== (rightArrays, epsilon)

          case ((ArrayType(ArrayType(BinaryType,_),_), ArrayType(ArrayType(BinaryType,_),_)), i) =>
            val leftArrays = ArrayData.toArrayData(left.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toArray[Array[Byte]](BinaryType).map(_.toSeq).toSeq
            }
            val rightArrays = ArrayData.toArrayData(right.get(i)).array.toSeq.map {arr =>
              ArrayData.toArrayData(arr).toArray[Array[Byte]](BinaryType).map(_.toSeq).toSeq
            }
            leftArrays === rightArrays

          case((a,b), i) => left.get(i) === right.get(i)
        }
      }
      else false
    }
  }
}