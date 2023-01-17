## this is a full fledged machine learning pipeline for MDP 

import pyspark
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Test').getOrCreate()

## reads the actions.csv file and loads it into spark_dataframe.
df_spark_actions = spark.read.option('header', 'true').csv('actions.csv')
df_spark_actions.show()

## reades the states.csv and loads it into a spark_dataframe
df_spark_states = spark.read.option('header', 'true').csv('states.csv')
df_spark_states.show() 

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def convert_states(s):
  return str(s)

@udf(returnType=StringType())
def convert_actions(s):
  return str(s)

df_spark_states.select(convert_states("States")).show()


from pyspark import keyword_only  # Note: use pyspark.ml.util.keyword_only if Spark < 2.0
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class StateTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
  """
    Custom Transformer wrapper class for convert_states()
  """ 
  


