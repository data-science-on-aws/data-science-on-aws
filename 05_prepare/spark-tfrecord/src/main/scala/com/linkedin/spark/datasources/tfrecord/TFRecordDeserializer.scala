package com.linkedin.spark.datasources.tfrecord

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{SpecializedGetters, SpecificInternalRow, UnsafeArrayData}
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.sql.types.{DecimalType, DoubleType, _}
import org.apache.spark.unsafe.types.UTF8String
import org.tensorflow.example._

import scala.collection.JavaConverters._

/**
 * Creates a TFRecord deserializer to deserialize Tfrecord example or sequenceExample to Spark InternalRow
 */
class TFRecordDeserializer(dataSchema: StructType) {

  private val resultRow = new SpecificInternalRow(dataSchema.map(_.dataType))

  def deserializeExample(example: Example): InternalRow = {
    val featureMap = example.getFeatures.getFeatureMap.asScala
    dataSchema.zipWithIndex.foreach {
      case (field, index) =>
        val feature = featureMap.get(field.name)
        feature match {
          case Some(ft) =>
            val featureWriter = newFeatureWriter(field.dataType, new RowUpdater(resultRow))
            featureWriter(index, ft)
          case None => if (!field.nullable) throw new NullPointerException(s"Field ${field.name} does not allow null values")
        }
    }
    resultRow
  }

  def deserializeSequenceExample(sequenceExample: SequenceExample): InternalRow = {

    val featureMap = sequenceExample.getContext.getFeatureMap.asScala
    val featureListMap = sequenceExample.getFeatureLists.getFeatureListMap.asScala

    dataSchema.zipWithIndex.foreach {
      case (field, index) =>
        val feature = featureMap.get(field.name)
        feature match {
          case Some(ft) =>
            val featureWriter = newFeatureWriter(field.dataType, new RowUpdater(resultRow))
            featureWriter(index, ft)
          case None =>
            val featureList = featureListMap.get(field.name)
            featureList match {
              case Some(ftList) =>
                val featureListWriter = newFeatureListWriter(field.dataType, new RowUpdater(resultRow))
                featureListWriter(index, ftList)
              case None => if (!field.nullable) throw new NullPointerException(s"Field ${field.name}  does not allow null values")
            }
        }
    }
    resultRow
  }

  private type arrayElementConverter = (SpecializedGetters, Int) => Any

  /**
   * Creates a writer to write Tfrecord Feature values to Catalyst data structure at the given ordinal.
   */
  private def newFeatureWriter(
    dataType: DataType, updater: CatalystDataUpdater): (Int, Feature) => Unit =
    dataType match {
      case NullType => (ordinal, _) =>
        updater.setNullAt(ordinal)

      case IntegerType => (ordinal, feature) =>
        updater.setInt(ordinal, Int64ListFeature2SeqLong(feature).head.toInt)

      case LongType => (ordinal, feature) =>
        updater.setLong(ordinal, Int64ListFeature2SeqLong(feature).head)

      case FloatType => (ordinal, feature) =>
        updater.setFloat(ordinal, floatListFeature2SeqFloat(feature).head.toFloat)

      case DoubleType => (ordinal, feature) =>
        updater.setDouble(ordinal, floatListFeature2SeqFloat(feature).head.toDouble)

      case DecimalType() => (ordinal, feature) =>
        updater.setDecimal(ordinal, Decimal(floatListFeature2SeqFloat(feature).head.toDouble))

      case StringType => (ordinal, feature) =>
        val value = bytesListFeature2SeqString(feature).head
        updater.set(ordinal, UTF8String.fromString(value))

      case BinaryType => (ordinal, feature) =>
        val value = bytesListFeature2SeqArrayByte(feature).head
        updater.set(ordinal, value)

      case ArrayType(elementType, _) => (ordinal, feature) =>

        elementType match {
          case IntegerType | LongType | FloatType | DoubleType | DecimalType() | StringType | BinaryType =>
            val valueList = elementType match {
              case IntegerType  => Int64ListFeature2SeqLong(feature).map(_.toInt)
              case LongType => Int64ListFeature2SeqLong(feature)
              case FloatType => floatListFeature2SeqFloat(feature).map(_.toFloat)
              case DoubleType => floatListFeature2SeqFloat(feature).map(_.toDouble)
              case DecimalType() => floatListFeature2SeqFloat(feature).map(x => Decimal(x.toDouble))
              case StringType => bytesListFeature2SeqString(feature)
              case BinaryType => bytesListFeature2SeqArrayByte(feature)
            }
            val len = valueList.length
            val result = createArrayData(elementType, len)
            val elementUpdater = new ArrayDataUpdater(result)
            val elementConverter = newArrayElementWriter(elementType, elementUpdater)
            for (idx <- 0 until len) {
              elementConverter(idx, valueList(idx))
            }
            updater.set(ordinal, result)

          case _ => throw new scala.RuntimeException(s"Cannot convert Array type to unsupported data type ${elementType}")
        }

      case _ =>
        throw new UnsupportedOperationException(s"$dataType is not supported yet.")
    }

  /**
   * Creates a writer to write Tfrecord FeatureList values to Catalyst data structure at the given ordinal.
   */
  private def newFeatureListWriter(
    dataType: DataType, updater: CatalystDataUpdater): (Int, FeatureList) => Unit =
    dataType match {
      case ArrayType(elementType, _) => (ordinal, featureList) =>
        val ftList = featureList.getFeatureList.asScala
        val len = ftList.length
        val resultArray = createArrayData(elementType, len)
        val elementUpdater = new ArrayDataUpdater(resultArray)
        val elementConverter = newFeatureWriter(elementType, elementUpdater)
        for (idx <- 0 until len) {
          elementConverter(idx, ftList(idx))
        }
        updater.set(ordinal, resultArray)
      case _ => throw new scala.RuntimeException(s"Cannot convert FeatureList to unsupported data type ${dataType}")
    }

  /**
   * Creates a writer to write Tfrecord Feature array element to Catalyst data structure at the given ordinal.
   */
  private def newArrayElementWriter(
    dataType: DataType, updater: CatalystDataUpdater): (Int, Any) => Unit =
    dataType match {
      case NullType => null

      case IntegerType => (ordinal, value) =>
        updater.setInt(ordinal, value.asInstanceOf[Int])

      case LongType => (ordinal, value) =>
        updater.setLong(ordinal, value.asInstanceOf[Long])

      case FloatType => (ordinal, value) =>
        updater.setFloat(ordinal, value.asInstanceOf[Float])

      case DoubleType => (ordinal, value) =>
        updater.setDouble(ordinal, value.asInstanceOf[Double])

      case DecimalType() => (ordinal, value) =>
        updater.setDecimal(ordinal, value.asInstanceOf[Decimal])

      case StringType => (ordinal, value) =>
        updater.set(ordinal, UTF8String.fromString(value.asInstanceOf[String]))

      case BinaryType => (ordinal, value) =>
        updater.set(ordinal, value.asInstanceOf[Array[Byte]])

      case _ => throw new RuntimeException(s"Cannot convert array element to unsupported data type ${dataType}")
  }

  def Int64ListFeature2SeqLong(feature: Feature): Seq[Long] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.INT64_LIST_FIELD_NUMBER, "Feature must be of type Int64List")
    try {
      feature.getInt64List.getValueList.asScala.toSeq.map(_.toLong)
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to long.", ex)
    }
  }

  def floatListFeature2SeqFloat(feature: Feature): Seq[java.lang.Float] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.FLOAT_LIST_FIELD_NUMBER, "Feature must be of type FloatList")
    try {
      val array = feature.getFloatList.getValueList.asScala.toSeq
      array
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to Float.", ex)
    }
  }

  def bytesListFeature2SeqArrayByte(feature: Feature): Seq[Array[Byte]] = {
    require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
    try {
      feature.getBytesList.getValueList.asScala.map((byteArray) => byteArray.asScala.toArray.map(_.toByte))
    }
    catch {
      case ex: Exception =>
        throw new RuntimeException(s"Cannot convert feature to byte array.", ex)
    }
  }

  def bytesListFeature2SeqString(feature: Feature): Seq[String] = {
      require(feature != null && feature.getKindCase.getNumber == Feature.BYTES_LIST_FIELD_NUMBER, "Feature must be of type ByteList")
      try {
        val array = feature.getBytesList.getValueList.asScala.toSeq
        array.map(_.toStringUtf8)
      }
      catch {
        case ex: Exception =>
          throw new RuntimeException(s"Cannot convert feature to String array.", ex)
      }
  }

  private def createArrayData(elementType: DataType, length: Int): ArrayData = elementType match {
    case BooleanType => new GenericArrayData(new Array[Boolean](length))
    case ByteType => new GenericArrayData(new Array[Byte](length))
    case ShortType => new GenericArrayData(new Array[Short](length))
    case IntegerType => new GenericArrayData(new Array[Int](length))
    case LongType => new GenericArrayData(new Array[Long](length))
    case FloatType => new GenericArrayData(new Array[Float](length))
    case DoubleType => new GenericArrayData(new Array[Double](length))
    case _ => new GenericArrayData(new Array[Any](length))
  }

  /**
    * A base interface for updating values inside catalyst data structure like `InternalRow` and
    * `ArrayData`.
    */
  sealed trait CatalystDataUpdater {
    def set(ordinal: Int, value: Any): Unit
    def setNullAt(ordinal: Int): Unit = set(ordinal, null)
    def setBoolean(ordinal: Int, value: Boolean): Unit = set(ordinal, value)
    def setByte(ordinal: Int, value: Byte): Unit = set(ordinal, value)
    def setShort(ordinal: Int, value: Short): Unit = set(ordinal, value)
    def setInt(ordinal: Int, value: Int): Unit = set(ordinal, value)
    def setLong(ordinal: Int, value: Long): Unit = set(ordinal, value)
    def setDouble(ordinal: Int, value: Double): Unit = set(ordinal, value)
    def setFloat(ordinal: Int, value: Float): Unit = set(ordinal, value)
    def setDecimal(ordinal: Int, value: Decimal): Unit = set(ordinal, value)
  }

  final class RowUpdater(row: InternalRow) extends CatalystDataUpdater {
    override def setNullAt(ordinal: Int): Unit = row.setNullAt(ordinal)
    override def set(ordinal: Int, value: Any): Unit = row.update(ordinal, value)
    override def setBoolean(ordinal: Int, value: Boolean): Unit = row.setBoolean(ordinal, value)
    override def setByte(ordinal: Int, value: Byte): Unit = row.setByte(ordinal, value)
    override def setShort(ordinal: Int, value: Short): Unit = row.setShort(ordinal, value)
    override def setInt(ordinal: Int, value: Int): Unit = row.setInt(ordinal, value)
    override def setLong(ordinal: Int, value: Long): Unit = row.setLong(ordinal, value)
    override def setDouble(ordinal: Int, value: Double): Unit = row.setDouble(ordinal, value)
    override def setFloat(ordinal: Int, value: Float): Unit = row.setFloat(ordinal, value)
    override def setDecimal(ordinal: Int, value: Decimal): Unit =
      row.setDecimal(ordinal, value, value.precision)
  }

  final class ArrayDataUpdater(array: ArrayData) extends CatalystDataUpdater {
    override def setNullAt(ordinal: Int): Unit = array.setNullAt(ordinal)
    override def set(ordinal: Int, value: Any): Unit = array.update(ordinal, value)
    override def setBoolean(ordinal: Int, value: Boolean): Unit = array.setBoolean(ordinal, value)
    override def setByte(ordinal: Int, value: Byte): Unit = array.setByte(ordinal, value)
    override def setShort(ordinal: Int, value: Short): Unit = array.setShort(ordinal, value)
    override def setInt(ordinal: Int, value: Int): Unit = array.setInt(ordinal, value)
    override def setLong(ordinal: Int, value: Long): Unit = array.setLong(ordinal, value)
    override def setDouble(ordinal: Int, value: Double): Unit = array.setDouble(ordinal, value)
    override def setFloat(ordinal: Int, value: Float): Unit = array.setFloat(ordinal, value)
    override def setDecimal(ordinal: Int, value: Decimal): Unit = array.update(ordinal, value)
  }
}
