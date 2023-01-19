## this is a full fledged machine learning pipeline for MDP 

import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import keyword_only  # Note: use pyspark.ml.util.keyword_only if Spark < 2.0
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName('Test').getOrCreate()

## reads the actions.csv file and loads it into spark_dataframe.
df_spark_actions = spark.read.option('header', 'true').csv('actions.csv')
df_spark_actions.show()

## reades the states.csv and loads it into a spark_dataframe
df_spark_states = spark.read.option('header', 'true').csv('states.csv')
df_spark_states.show() 

df_spark_tf = spark.read.option('header', 'true').option("inferSchema", 'true').csv('tf_b.csv')
df_spark_tf.show()

df_spark_tf = spark.read.option('header', 'true').option("inferSchema", 'true').csv('tf_b.csv')
pd_frame_tf =df_spark_tf.toPandas()

""" 
This part converts a spark dataframe object to pandas and
makes the transition function, orderering required to make TF object
"""

length = len(df_spark_tf.dtypes) 
order = {}

arr = [[0 for i in range(length)] for i in range(length)]


for i in range(length):
  order[df_spark_tf.dtypes[i][0]] = i
  for j in range(length):
    arr[i][j]=pd_frame_tf[df_spark_tf.dtypes[i][0]][j]


@udf(returnType=StringType())
def convert_states(s):
  return str(s)

@udf(returnType=StringType())
def convert_actions(s):
  return str(s)

# df_spark_states.select(convert_states("States")).show()


class StateTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  """
    Custom Transformer wrapper class for convert_states()
  """ 

  input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
  output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)

  @keyword_only
  def __init__(self, input_col: str = "input", output_col: str = "output"):
    super(StateTransformer, self).__init__()
    self._setDefault(input_col=None, output_col=None)
    kwargs = self._input_kwargs
    self.set_params(**kwargs)

  @keyword_only
  def set_params(self, input_col: str = "input", output_col: str = "output"):
    kwargs = self._input_kwargs
    self._set(**kwargs)
  
  def get_input_col(self):
    return self.getOrDefault(self.input_col)

  def get_output_col(self):
    return self.getOrDefault(self.output_col)

  def _transform(self, df: DataFrame):
    input_col = self.get_input_col()
    output_col = self.get_output_col()

    return df.withColumn(output_col, convert_states(input_col))

class ActionTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  """
    Custom Transformer wrapper class for convert_states()
  """ 

  input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
  output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)

  @keyword_only
  def __init__(self, input_col: str = "input", output_col: str = "output"):
    super(StateTransformer, self).__init__()
    self._setDefault(input_col=None, output_col=None)
    kwargs = self._input_kwargs
    self.set_params(**kwargs)

  @keyword_only
  def set_params(self, input_col: str = "input", output_col: str = "output"):
    kwargs = self._input_kwargs
    self._set(**kwargs)
  
  def get_input_col(self):
    return self.getOrDefault(self.input_col)

  def get_output_col(self):
    return self.getOrDefault(self.output_col)

  def _transform(self, df: DataFrame):
    input_col = self.get_input_col()
    output_col = self.get_output_col()

    return df.withColumn(output_col, convert_actions(input_col))







