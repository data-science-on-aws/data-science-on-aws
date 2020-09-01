package com.linkedin.spark.datasources.tfrecord

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types.{DecimalType, DoubleType, _}
import org.tensorflow.example._
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters
import com.google.protobuf.ByteString

/**
 * Creates a TFRecord serializer to serialize Spark InternalRow to Tfrecord example or sequenceExample
 */
class TFRecordSerializer(dataSchema: StructType) {

  private val featureConverters = dataSchema.map(_.dataType).map(newFeatureConverter(_)).toArray

  def serializeExample(row: InternalRow): Example = {
    val features = Features.newBuilder()
    val example = Example.newBuilder()
    for (idx <- featureConverters.indices) {
      val structField = dataSchema(idx)
      if (!row.isNullAt(idx)) {
        val feature = featureConverters(idx)(row, idx).asInstanceOf[Feature]
        features.putFeature(structField.name, feature)
      }
      else if (!dataSchema(idx).nullable) {
        throw new NullPointerException(s"${structField.name} does not allow null values")
      }
    }
    example.setFeatures(features.build())
    example.build()
  }

  def serializeSequenceExample(row: InternalRow): SequenceExample = {
    val features = Features.newBuilder()
    val featureLists = FeatureLists.newBuilder()
    val sequenceExample = SequenceExample.newBuilder()
    for (idx <- featureConverters.indices) {
      val structField = dataSchema(idx)
      if (!row.isNullAt(idx)) {
        structField.dataType match {
          case ArrayType(ArrayType(_, _), _) =>
            val featureList = featureConverters(idx)(row, idx).asInstanceOf[FeatureList]
            featureLists.putFeatureList(structField.name, featureList)
          case _ =>
            val feature = featureConverters(idx)(row, idx).asInstanceOf[Feature]
            features.putFeature(structField.name, feature)
        }
      }
      else if (!dataSchema(idx).nullable) {
        throw new NullPointerException(s"${structField.name} does not allow null values")
      }
    }
    sequenceExample.setContext(features.build())
    sequenceExample.setFeatureLists(featureLists.build())
    sequenceExample.build()
  }

  private type FeatureConverter = (SpecializedGetters, Int) => Any
  private type arrayElementConverter = (SpecializedGetters, Int) => Any

  /**
    * Creates a converter to convert Catalyst data at the given ordinal to TFrecord Feature.
    */
  private def newFeatureConverter(
    dataType: DataType): FeatureConverter = dataType match {
    case NullType => (getter, ordinal) => null

    case IntegerType => (getter, ordinal) =>
      val value = getter.getInt(ordinal)
      Int64ListFeature(Seq(value.toLong))

    case LongType => (getter, ordinal) =>
      val value = getter.getLong(ordinal)
      Int64ListFeature(Seq(value))

    case FloatType  => (getter, ordinal) =>
      val value = getter.getFloat(ordinal)
      floatListFeature(Seq(value))

    case DoubleType  => (getter, ordinal) =>
      val value = getter.getDouble(ordinal)
      floatListFeature(Seq(value.toFloat))

    case  DecimalType() => (getter, ordinal) =>
      val value = getter.getDecimal(ordinal, DecimalType.USER_DEFAULT.precision, DecimalType.USER_DEFAULT.scale)
      floatListFeature(Seq(value.toFloat))

    case StringType => (getter, ordinal) =>
      val value = getter.getUTF8String(ordinal).getBytes
      bytesListFeature(Seq(value))

    case BinaryType => (getter, ordinal) =>
      val value = getter.getBinary(ordinal)
      bytesListFeature(Seq(value))

    case ArrayType(elementType, containsNull) => (getter, ordinal) =>
      val arrayData = getter.getArray(ordinal)
      val featureOrFeatureList = elementType match {
        case IntegerType =>
          Int64ListFeature(arrayData.toIntArray().map(_.toLong))

        case LongType =>
          Int64ListFeature(arrayData.toLongArray())

        case FloatType =>
          floatListFeature(arrayData.toFloatArray())

        case DoubleType =>
          floatListFeature(arrayData.toDoubleArray().map(_.toFloat))

        case  DecimalType() =>
          val elementConverter = arrayElementConverter(elementType)
          val len = arrayData.numElements()
          val result = new Array[Decimal](len)
          for (idx <- 0 until len) {
            if (containsNull && arrayData.isNullAt(idx)) {
              result(idx) = null
            } else result(idx) = elementConverter(arrayData, idx).asInstanceOf[Decimal]
          }
          floatListFeature(result.map(_.toFloat))

        case StringType | BinaryType =>
          val elementConverter = arrayElementConverter(elementType)
          val len = arrayData.numElements()
          val result = new Array[Array[Byte]](len)
          for (idx <- 0 until len) {
            if (containsNull && arrayData.isNullAt(idx)) {
              result(idx) = null
            } else result(idx) = elementConverter(arrayData, idx).asInstanceOf[Array[Byte]]
          }
          bytesListFeature(result)

        // 2-dimensional array to TensorFlow "FeatureList"
        case ArrayType(_, _) =>
          val elementConverter = newFeatureConverter(elementType)
          val featureList = FeatureList.newBuilder()
          for (idx <- 0 until arrayData.numElements) {
            val feature = elementConverter(arrayData, idx).asInstanceOf[Feature]
            featureList.addFeature(feature)
          }
          featureList.build()

        case _ => throw new RuntimeException(s"Array element data type ${dataType} is unsupported")
      }
      featureOrFeatureList

    case _ => throw new RuntimeException(s"Cannot convert field to unsupported data type ${dataType}")
  }

  private def arrayElementConverter(
    dataType: DataType): arrayElementConverter = dataType match {
    case NullType => null

    case IntegerType => (getter, ordinal) =>
      getter.getInt(ordinal)

    case LongType => (getter, ordinal) =>
      getter.getLong(ordinal)

    case FloatType => (getter, ordinal) =>
      getter.getFloat(ordinal)

    case DoubleType => (getter, ordinal) =>
      getter.getDouble(ordinal)

    case DecimalType() => (getter, ordinal) =>
      getter.getDecimal(ordinal, DecimalType.USER_DEFAULT.precision, DecimalType.USER_DEFAULT.scale)

    case StringType => (getter, ordinal) =>
      getter.getUTF8String(ordinal).getBytes

    case BinaryType => (getter, ordinal) =>
      getter.getBinary(ordinal)

    case _ => throw new RuntimeException(s"Cannot convert field to unsupported data type ${dataType}")
  }

  def Int64ListFeature(value: Seq[Long]): Feature = {
    val intListBuilder = Int64List.newBuilder()
    value.foreach {x =>
      intListBuilder.addValue(x)
    }
    val int64List = intListBuilder.build()
    Feature.newBuilder().setInt64List(int64List).build()
  }

  def floatListFeature(value: Seq[Float]): Feature = {
    val floatListBuilder = FloatList.newBuilder()
    value.foreach {x =>
      floatListBuilder.addValue(x)
    }
    val floatList = floatListBuilder.build()
    Feature.newBuilder().setFloatList(floatList).build()
  }

  def bytesListFeature(value: Seq[Array[Byte]]): Feature = {
    val bytesListBuilder = BytesList.newBuilder()
    value.foreach {x =>
      bytesListBuilder.addValue(ByteString.copyFrom(x))
    }
    val bytesList = bytesListBuilder.build()
    Feature.newBuilder().setBytesList(bytesList).build()
  }
}
