# https://github.com/linkedin/spark-tfrecord

curl -O https://downloads.apache.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
tar -xvzf apache-maven-3.6.3-bin.tar.gz

./apache-maven-3.6.3/bin/mvn clean install -Dspark.version=2.4.5

cp target/spark-tfrecord_2.11-0.1.1.jar ../container/jars/ 

echo "Copied jar to container/jars."

echo ""

echo "Don't ferget to rebuild the Docker image."
