FROM openjdk:8-jre-slim

RUN apt-get update
RUN apt-get install -y curl unzip python3 python3-setuptools python3-pip python-dev python3-dev python-psutil
RUN pip3 install py4j psutil==5.6.5 numpy==1.17.4
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed
ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Install Hadoop
ENV HADOOP_VERSION 3.1.3
ENV HADOOP_HOME /usr/hadoop-$HADOOP_VERSION
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
RUN curl -sL --retry 3 \
  "http://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /usr/ \
 && rm -rf $HADOOP_HOME/share/doc \
 && chown -R root:root $HADOOP_HOME

# Install Spark
ENV SPARK_VERSION 2.4.6
#ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-hadoop-3.1.3
ENV SPARK_HOME /usr/spark-${SPARK_VERSION}
ENV SPARK_DIST_CLASSPATH="$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
#RUN curl -sL --retry 3 \
#  "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
#  | gunzip \
#  | tar x -C /usr/ \
# && mv /usr/$SPARK_PACKAGE $SPARK_HOME \
# && chown -R root:root $SPARK_HOME

RUN curl --retry 3 -O https://s3.amazonaws.com/fluxcapacitor.com/packages/spark-2.4.6-hadoop-3.1.3.tar.gz

RUN mkdir -p /usr/$SPARK_PACKAGE/ \
  && tar -C /usr/$SPARK_PACKAGE/ -xzv -f spark-2.4.6-hadoop-3.1.3.tar.gz -C /usr/$SPARK_PACKAGE/ \
 && mv /usr/$SPARK_PACKAGE $SPARK_HOME \
 && chown -R root:root $SPARK_HOME

RUN echo $SPARK_HOME && ls -al $SPARK_HOME

# Point Spark at proper python binary
ENV PYSPARK_PYTHON=/usr/bin/python3

# Setup Spark/Yarn/HDFS user as root
ENV PATH="/usr/bin:/opt/program:${PATH}"
ENV YARN_RESOURCEMANAGER_USER="root"
ENV YARN_NODEMANAGER_USER="root"
ENV HDFS_NAMENODE_USER="root"
ENV HDFS_DATANODE_USER="root"
ENV HDFS_SECONDARYNAMENODE_USER="root"

# Set up bootstrapping program and Spark configuration
COPY program /opt/program
RUN chmod +x /opt/program/submit
COPY hadoop-config /opt/hadoop-config

COPY jars /usr/jars

WORKDIR $SPARK_HOME

ENTRYPOINT ["/opt/program/submit"]
