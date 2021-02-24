# Spark-TFRecord

A library for reading and writing [Tensorflow TFRecord](https://www.tensorflow.org/how_tos/reading_data/) data from [Apache Spark](http://spark.apache.org/).
The implementation is based on [Spark Tensorflow Connector](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector), but it is rewritten in Spark FileFormat trait to provide the partitioning function.

## Prerequisites

[Apache Spark 2.0 (or later)](http://spark.apache.org/)

## Building the library
Build the library using Maven 3.3.9 or newer as shown below:

```sh
# Build Spark-TFRecord
git clone https://github.com/linkedin/spark-tfrecord.git
cd spark-tfrecord
mvn clean install
# one can specific the spark version and tensorflow hadoop version
# mvn clean install -Dspark.version=2.2.1 -Dtensorflow.hadoop.version=1.15.0
```

After installation (or deployment), the package can be used with the following dependency:

```xml
<dependency>
  <groupId>com.linkedin.sparktfrecord</groupId>
  <artifactId>spark-tfrecord_2.11</artifactId>
  <version>0.1.1</version>
</dependency>
```

## Using Spark Shell
Run this library in Spark using the `--jars` command line option in `spark-shell`, `pyspark` or `spark-submit`. For example:

```sh
$SPARK_HOME/bin/spark-shell --jars target/spark-tfrecord_2.11-0.1.1.jar
```

## Features
This library allows reading TensorFlow records in local or distributed filesystem as [Spark DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html).
When reading TensorFlow records into Spark DataFrame, the API accepts several options:
* `load`: input path to TensorFlow records. Similar to Spark can accept standard Hadoop globbing expressions.
* `schema`: schema of TensorFlow records. Optional schema defined using Spark StructType. If not provided, the schema is inferred from TensorFlow records.
* `recordType`: input format of TensorFlow records. By default it is Example. Possible values are:
  * `Example`: TensorFlow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
  * `SequenceExample`: TensorFlow [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records

When writing Spark DataFrame to TensorFlow records, the API accepts several options:
* `save`: output path to TensorFlow records. Output path to TensorFlow records on local or distributed filesystem.
compression. While reading compressed TensorFlow records, `codec` can be inferred automatically, so this option is not required for reading.
* `recordType`: output format of TensorFlow records. By default it is Example. Possible values are:
  * `Example`: TensorFlow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records
  * `SequenceExample`: TensorFlow [SequenceExample](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) records

The writer support partitionBy operation. So the following command will partition the output by "partitionColumn".
```
df.write.mode(SaveMode.Overwrite).partitionBy("partitionColumn").format("tfrecord").option("recordType", "Example").save(output_dir)
```
Note we use `format("tfrecord")` instead `format("tfrecords")`. So if you migrate from Spark-Tensorflow-Connector, make sure this is changed accordingly.

## Schema inference
This library supports automatic schema inference when reading TensorFlow records into Spark DataFrames.
Schema inference is expensive since it requires an extra pass through the data.

The schema inference rules are described in the table below:

| TFRecordType             | Feature Type  | Inferred Spark Data Type  |
| ------------------------ |:--------------|:--------------------------|
| Example, SequenceExample | Int64List     | LongType if all lists have length=1, else ArrayType(LongType) |
| Example, SequenceExample | FloatList     | FloatType if all lists have length=1, else ArrayType(FloatType) |
| Example, SequenceExample | BytesList     | StringType if all lists have length=1, else ArrayType(StringType) |
| SequenceExample          | FeatureList of Int64List | ArrayType(ArrayType(LongType)) |
| SequenceExample          | FeatureList of FloatList | ArrayType(ArrayType(FloatType)) |
| SequenceExample          | FeatureList of BytesList | ArrayType(ArrayType(StringType)) |

## Supported data types

The supported Spark data types are listed in the table below:

| Type            | Spark DataTypes                          |
| --------------- |:------------------------------------------|
| Scalar          | IntegerType, LongType, FloatType, DoubleType, DecimalType, StringType, BinaryType |
| Array           | VectorType, ArrayType of IntegerType, LongType, FloatType, DoubleType, DecimalType, BinaryType, or StringType |
| Array of Arrays | ArrayType of ArrayType of IntegerType, LongType, FloatType, DoubleType, DecimalType, BinaryType, or StringType |

## Usage Examples

### Python API

#### TF record Import/export

Run PySpark with the spark_connector in the jars argument as shown below:

`$SPARK_HOME/bin/pyspark --jars target/spark-tfrecord_2.11-0.1.1.jar`

The following Python code snippet demonstrates usage on test data.

```
from pyspark.sql.types import *

path = "test-output.tfrecord"

fields = [StructField("id", IntegerType()), StructField("IntegerCol", IntegerType()),
          StructField("LongCol", LongType()), StructField("FloatCol", FloatType()),
          StructField("DoubleCol", DoubleType()), StructField("VectorCol", ArrayType(DoubleType(), True)),
          StructField("StringCol", StringType())]
schema = StructType(fields)
test_rows = [[11, 1, 23, 10.0, 14.0, [1.0, 2.0], "r1"], [21, 2, 24, 12.0, 15.0, [2.0, 2.0], "r2"]]
rdd = spark.sparkContext.parallelize(test_rows)
df = spark.createDataFrame(rdd, schema)
df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(path)
df = spark.read.format("tfrecord").option("recordType", "Example").load(path)
df.show()
```

### Scala API
Run Spark shell with the spark_connector in the jars argument as shown below:
```sh
$SPARK_HOME/bin/spark-shell --jars target/spark-tfrecord_2.11-0.1.1.jar
```

The following Scala code snippet demonstrates usage on test data.

```scala
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.types._

val path = "test-output.tfrecord"
val testRows: Array[Row] = Array(
new GenericRow(Array[Any](11, 1, 23L, 10.0F, 14.0, List(1.0, 2.0), "r1")),
new GenericRow(Array[Any](21, 2, 24L, 12.0F, 15.0, List(2.0, 2.0), "r2")))
val schema = StructType(List(StructField("id", IntegerType),
                             StructField("IntegerCol", IntegerType),
                             StructField("LongCol", LongType),
                             StructField("FloatCol", FloatType),
                             StructField("DoubleCol", DoubleType),
                             StructField("VectorCol", ArrayType(DoubleType, true)),
                             StructField("StringCol", StringType)))

val rdd = spark.sparkContext.parallelize(testRows)

//Save DataFrame as TFRecords
val df: DataFrame = spark.createDataFrame(rdd, schema)
df.write.format("tfrecord").option("recordType", "Example").save(path)

//Read TFRecords into DataFrame.
//The DataFrame schema is inferred from the TFRecords if no custom schema is provided.
val importedDf1: DataFrame = spark.read.format("tfrecord").option("recordType", "Example").load(path)
importedDf1.show()

//Read TFRecords into DataFrame using custom schema
val importedDf2: DataFrame = spark.read.format("tfrecord").schema(schema).load(path)
importedDf2.show()
```

#### Use partitionBy
The following example shows to how to use partitionBy, which is not supported by [Spark Tensorflow Connector](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)

```scala

// launch spark-shell with the following command:
// SPARK_HOME/bin/spark-shell --jar target/spark-tfrecord_2.11-0.1.1.jar

import org.apache.spark.sql.SaveMode

val df = Seq((8, "bat"),(8, "abc"), (1, "xyz"), (2, "aaa")).toDF("number", "word")
df.show

// scala> df.show
// +------+----+
// |number|word|
// +------+----+
// |     8| bat|
// |     8| abc|
// |     1| xyz|
// |     2| aaa|
// +------+----+

val tf_output_dir = "/tmp/tfrecord-test"

// dump the tfrecords to files.
df.repartition(3, col("number")).write.mode(SaveMode.Overwrite).partitionBy("number").format("tfrecord").option("recordType", "Example").save(tf_output_dir)

// ls /tmp/tfrecord-test
// _SUCCESS        number=1        number=2        number=8

// read back the tfrecords from files.
val new_df = spark.read.format("tfrecord").option("recordType", "Example").load(tf_output_dir)
new_df.show

// scala> new_df.show
// +----+------+
// |word|number|
// +----+------+
// | bat|     8|
// | abc|     8|
// | xyz|     1|
// | aaa|     2|
```
## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE.md](LICENSE.md) file for details
