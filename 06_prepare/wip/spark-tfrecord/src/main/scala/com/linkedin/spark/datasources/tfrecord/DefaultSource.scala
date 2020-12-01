package com.linkedin.spark.datasources.tfrecord

import java.io.{DataInputStream, DataOutputStream, IOException, ObjectInputStream, ObjectOutputStream, Serializable}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.io.SequenceFile.CompressionType
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.slf4j.LoggerFactory
import org.tensorflow.example.{Example, SequenceExample}
import org.tensorflow.hadoop.io.TFRecordFileInputFormat

import scala.util.control.NonFatal

class DefaultSource extends FileFormat with DataSourceRegister {
  override val shortName: String = "tfrecord"

  override def isSplitable(
      sparkSession: SparkSession,
      options: Map[String, String],
      path: Path): Boolean = false

  override def inferSchema(
      sparkSession: SparkSession,
      options: Map[String, String],
      files: Seq[FileStatus]): Option[StructType] = {
    val recordType = options.getOrElse("recordType", "Example")
    // use the first file
    val rdd = sparkSession.sparkContext.newAPIHadoopFile(files(0).getPath.toString,
      classOf[TFRecordFileInputFormat], classOf[BytesWritable], classOf[NullWritable])
    val finalSchema = recordType match {
      case "Example" =>
        val exampleRdd = rdd.map{case (bytesWritable, nullWritable) =>
          Example.parseFrom(bytesWritable.getBytes)
        }
        TensorFlowInferSchema(exampleRdd)
      case "SequenceExample" =>
        val sequenceExampleRdd = rdd.map{case (bytesWritable, nullWritable) =>
          SequenceExample.parseFrom(bytesWritable.getBytes)
        }
        TensorFlowInferSchema(sequenceExampleRdd)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported recordType ${recordType}: recordType can be Example or SequenceExample")
    }
    Some(finalSchema)
  }

  override def prepareWrite(
      sparkSession: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType): OutputWriterFactory = {
    val conf = job.getConfiguration
    val codec = options.getOrElse("codec", "")
    if (!codec.isEmpty) {
      conf.set("mapreduce.output.fileoutputformat.compress", "true")
      conf.set("mapreduce.output.fileoutputformat.compress.type", CompressionType.BLOCK.toString)
      conf.set("mapreduce.output.fileoutputformat.compress.codec", codec)
      conf.set("mapreduce.map.output.compress", "true")
      conf.set("mapreduce.map.output.compress.codec", codec)
    }

    new OutputWriterFactory {
      override def newInstance(
          path: String,
          dataSchema: StructType,
          context: TaskAttemptContext): OutputWriter = {
        new TFRecordOutputWriter(path, options, dataSchema, context)
      }

      override def getFileExtension(context: TaskAttemptContext): String = {
        ".tfrecord" + CodecStreams.getCompressionExtension(context)
      }
    }
  }

  override def buildReader(
      sparkSession: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration): PartitionedFile => Iterator[InternalRow] = {
    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    (file: PartitionedFile) => {
      TFRecordFileReader.readFile(
        broadcastedHadoopConf.value.value,
        options,
        file,
        requiredSchema)
    }
  }

  override def toString: String = "TFRECORD"

  override def hashCode(): Int = getClass.hashCode()

  override def equals(other: Any): Boolean = other.isInstanceOf[DefaultSource]
}

private [tfrecord] class SerializableConfiguration(@transient var value: Configuration)
  extends Serializable with KryoSerializable {
  @transient private[tfrecord] lazy val log = LoggerFactory.getLogger(getClass)

  private def writeObject(out: ObjectOutputStream): Unit = tryOrIOException {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream): Unit = tryOrIOException {
    value = new Configuration(false)
    value.readFields(in)
  }

  private def tryOrIOException[T](block: => T): T = {
    try {
      block
    } catch {
      case e: IOException =>
        log.error("Exception encountered", e)
        throw e
      case NonFatal(e) =>
        log.error("Exception encountered", e)
        throw new IOException(e)
    }
  }

  def write(kryo: Kryo, out: Output): Unit = {
    val dos = new DataOutputStream(out)
    value.write(dos)
    dos.flush()
  }

  def read(kryo: Kryo, in: Input): Unit = {
    value = new Configuration(false)
    value.readFields(new DataInputStream(in))
  }
}
