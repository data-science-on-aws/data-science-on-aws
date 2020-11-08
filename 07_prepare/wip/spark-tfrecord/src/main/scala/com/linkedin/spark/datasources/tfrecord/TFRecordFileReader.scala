package com.linkedin.spark.datasources.tfrecord

import java.net.URI
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.{JobID, TaskAttemptID, TaskID, TaskType}
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.TaskContext
import org.apache.spark.sql.types.StructType
import org.tensorflow.example.{Example, SequenceExample}
import org.tensorflow.hadoop.io.TFRecordFileInputFormat

object TFRecordFileReader {
  def readFile(
    conf: Configuration,
    options: Map[String, String],
    file: PartitionedFile,
    schema: StructType): Iterator[InternalRow] = {

    val recordType = options.getOrElse("recordType", "Example")

    val inputSplit = new FileSplit(
      new Path(new URI(file.filePath)),
      file.start,
      file.length,
      // The locality is decided by `getPreferredLocations` in `FileScanRDD`.
      Array.empty)
    val attemptId = new TaskAttemptID(new TaskID(new JobID(), TaskType.MAP, 0), 0)
    val hadoopAttemptContext = new TaskAttemptContextImpl(conf, attemptId)
    val recordReader = new TFRecordFileInputFormat().createRecordReader(inputSplit, hadoopAttemptContext)

    // Ensure that the reader is closed even if the task fails or doesn't consume the entire
    // iterator of records.
    Option(TaskContext.get()).foreach { taskContext =>
      taskContext.addTaskCompletionListener { _ =>
        recordReader.close()
      }
    }

    recordReader.initialize(inputSplit, hadoopAttemptContext)

    val deserializer = new TFRecordDeserializer(schema)

    new Iterator[InternalRow] {
      private[this] var havePair = false
      private[this] var finished = false
      override def hasNext: Boolean = {
        if (!finished && !havePair) {
          finished = !recordReader.nextKeyValue
          if (finished) {
            // Close and release the reader here; close() will also be called when the task
            // completes, but for tasks that read from many files, it helps to release the
            // resources early.
            recordReader.close()
          }
          havePair = !finished
        }
        !finished
      }

      override def next(): InternalRow = {
        if (!hasNext) {
          throw new java.util.NoSuchElementException("End of stream")
        }
        havePair = false
        val bytesWritable = recordReader.getCurrentKey
        recordType match {
          case "Example" =>
            val example = Example.parseFrom(bytesWritable.getBytes)
            deserializer.deserializeExample(example)
          case "SequenceExample" =>
            val sequenceExample = SequenceExample.parseFrom(bytesWritable.getBytes)
            deserializer.deserializeSequenceExample(sequenceExample)
          case _ =>
            throw new IllegalArgumentException(s"Unsupported recordType ${recordType}: recordType can be Example or SequenceExample")
        }
      }
    }
  }
}