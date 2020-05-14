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

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.SharedSparkSession
import org.junit.{After, Before}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}


trait BaseSuite extends WordSpecLike with Matchers with BeforeAndAfterAll

class SharedSparkSessionSuite extends SharedSparkSession with BaseSuite {
  val TF_SANDBOX_DIR = "tf-sandbox"
  val file = new File(TF_SANDBOX_DIR)

  @Before
  override def beforeAll() = {
    super.setUp()
    FileUtils.deleteQuietly(file)
    file.mkdirs()
  }

  @After
  override def afterAll() = {
    FileUtils.deleteQuietly(file)
    super.tearDown()
  }
}

